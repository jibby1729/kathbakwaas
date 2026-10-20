"""Local aggregate traffic counts. Never persist input text or raw IP addresses."""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import ipaddress
from pathlib import Path
import secrets
import sqlite3


def cloudflare_ip(request):
    # The gateway binds only to loopback and disables Uvicorn proxy-header rewriting.
    # Only its local tunnel peer supplies Cloudflare's visitor header. XFF is ignored.
    if not request.client or request.client.host not in ('127.0.0.1', '::1'):
        return None
    if not request.headers.get('cf-ray'):
        return None
    try:
        address = ipaddress.ip_address(request.headers.get('cf-connecting-ip', ''))
        return str(getattr(address, 'ipv4_mapped', None) or address)
    except ValueError:
        return None


def secret_file(path):
    try:
        with path.open('xb') as file:
            file.write(secrets.token_bytes(32))
    except FileExistsError:
        pass
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError('Invalid analytics key length')
    return value


class TrafficStats:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = secret_file(self.path.with_suffix('.key'))
        self.check_token = secret_file(self.path.with_suffix('.check-key')).hex()
        self.db = sqlite3.connect(self.path, timeout=0.1)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA secure_delete=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counters (
                day TEXT, kind TEXT, status INTEGER, cached INTEGER, is_check INTEGER,
                count INTEGER NOT NULL, total_ms REAL NOT NULL,
                PRIMARY KEY(day, kind, status, cached, is_check));
            CREATE TABLE IF NOT EXISTS visitors (
                fingerprint TEXT, is_check INTEGER, first_seen TEXT, last_seen TEXT,
                PRIMARY KEY(fingerprint, is_check));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, time TEXT, kind TEXT, status INTEGER,
                cached INTEGER, is_check INTEGER, elapsed_ms REAL);
        ''')
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('enabled_at', ?)", (now,))
        self.db.commit()

    def record(self, kind, status, address, elapsed_ms, cached=None, check_token=''):
        if not address:
            return
        now = datetime.now(timezone.utc)
        stamp, day = now.isoformat(), now.date().isoformat()
        check = int(hmac.compare_digest(check_token, self.check_token))
        cache = int(cached) if isinstance(cached, bool) else -1
        fingerprint = hmac.new(self.key, address.encode('ascii'), hashlib.sha256).hexdigest()
        with self.db:
            self.db.execute('''INSERT INTO counters VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(day,kind,status,cached,is_check)
                DO UPDATE SET count=count+1, total_ms=total_ms+excluded.total_ms''',
                (day, kind, status, cache, check, elapsed_ms))
            if status == 200:
                self.db.execute('''INSERT INTO visitors VALUES (?, ?, ?, ?)
                    ON CONFLICT(fingerprint,is_check) DO UPDATE SET last_seen=excluded.last_seen''',
                    (fingerprint, check, stamp, stamp))
            self.db.execute('INSERT INTO events(time,kind,status,cached,is_check,elapsed_ms) VALUES (?,?,?,?,?,?)',
                            (stamp, kind, status, cache, check, elapsed_ms))
            # Keep recent diagnostics bounded; lifetime aggregate counts persist.
            self.db.execute('DELETE FROM events WHERE time < ? OR id <= (SELECT MAX(id)-10000 FROM events)',
                            ((now - timedelta(days=7)).isoformat(),))

    def close(self):
        self.db.close()


def summary(path, checks=False):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT kind,status,cached,SUM(count) FROM counters WHERE is_check=? GROUP BY kind,status,cached',
                          (int(checks),)).fetchall()
        def count(kind, status=None, cached=None):
            return sum(n for k, s, c, n in rows if k == kind and (status is None or s == status)
                       and (cached is None or c == cached))
        return {
            'enabled_at_utc': db.execute("SELECT value FROM metadata WHERE key='enabled_at'").fetchone()[0],
            'verification_traffic': checks,
            'page_loads': count('page', 200),
            'approximate_unique_ips': db.execute('SELECT COUNT(*) FROM visitors WHERE is_check=?', (int(checks),)).fetchone()[0],
            'successful_translations': count('translation', 200),
            'cached_translations': count('translation', 200, 1),
            'fresh_translations': count('translation', 200, 0),
            'translation_attempts': count('translation'),
            'busy_translations': count('translation', 429),
            'cancelled_translations': count('translation', 499),
            'server_error_translations': sum(n for k, s, _, n in rows if k == 'translation' and s >= 500),
        }
