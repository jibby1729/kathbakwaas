"""Local, bounded persistence for completed translations (never prompt tensors)."""
from collections import OrderedDict
import json
import logging
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


class TranslationCacheStore:
    def __init__(self, path, signature):
        self.connection = None
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path, timeout=0.2)
            self.connection.execute('PRAGMA journal_mode=WAL')
            with self.connection:
                self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (signature TEXT NOT NULL)')
                self.connection.execute('''CREATE TABLE IF NOT EXISTS translations (
                    english TEXT PRIMARY KEY, result TEXT NOT NULL, recency INTEGER NOT NULL)''')
                self.connection.execute('CREATE INDEX IF NOT EXISTS translation_recency ON translations(recency)')
                previous = self.connection.execute('SELECT signature FROM metadata').fetchone()
                if previous != (signature,):
                    self.connection.execute('DELETE FROM translations')
                    self.connection.execute('DELETE FROM metadata')
                    self.connection.execute('INSERT INTO metadata VALUES (?)', (signature,))
        except (OSError, sqlite3.Error):
            self._disable()

    def _disable(self):
        logger.warning('Translation cache persistence unavailable; using RAM only.', exc_info=True)
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def load(self, limit):
        cache = OrderedDict()
        if self.connection is None:
            return cache
        try:
            with self.connection:
                self._trim(limit)
                for text, payload in self.connection.execute(
                    'SELECT english, result FROM translations ORDER BY recency'
                ).fetchall():
                    try:
                        result = json.loads(payload)
                        valid = (isinstance(result, dict) and isinstance(result.get('translation'), str)
                                 and isinstance(result.get('draft'), str) and 0 < len(text) <= 500)
                    except (ValueError, TypeError):
                        valid = False
                    if valid:
                        cache[text] = result
                    else:
                        self.connection.execute('DELETE FROM translations WHERE english=?', (text,))
        except sqlite3.Error:
            self._disable()
        return cache

    def _trim(self, limit):
        self.connection.execute('''DELETE FROM translations WHERE english NOT IN
            (SELECT english FROM translations ORDER BY recency DESC LIMIT ?)''', (limit,))

    def put(self, text, result, limit):
        if self.connection is None:
            return
        try:
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
            with self.connection:
                self.connection.execute('''INSERT OR REPLACE INTO translations VALUES
                    (?, ?, (SELECT COALESCE(MAX(recency), 0) + 1 FROM translations))''', (text, payload))
                self._trim(limit)
        except (sqlite3.Error, ValueError, TypeError):
            self._disable()

    def touch(self, text):
        if self.connection is None:
            return
        try:
            with self.connection:
                self.connection.execute('''UPDATE translations SET recency=
                    (SELECT COALESCE(MAX(recency), 0) + 1 FROM translations) WHERE english=?''', (text,))
        except sqlite3.Error:
            self._disable()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def cache_signature(root):
    """Invalidate saved results when pipeline source or pinned model revisions change."""
    import hashlib
    import os
    digest = hashlib.sha256(b'completed-translations-v1')
    # engine.py includes model revisions, prompt, precision, decoding and restoration settings.
    for name in ('engine.py', 'fast_execution.py', 'execution.py', 'prefix_cache.py'):
        digest.update((Path(root) / 'app' / name).read_bytes())
    for name in ('KOSHUR_OPTIMIZE_EXECUTION', 'KOSHUR_NATIVE_DECODE', 'KOSHUR_PERFORMANCE_CORES'):
        digest.update(f'{name}={os.getenv(name, "1")}'.encode())
    return digest.hexdigest()
