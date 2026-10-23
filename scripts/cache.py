"""List or export saved translations locally. Does not contact the running server."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path(os.getenv(
        'KOSHUR_CACHE_DB', ROOT / '.hosting' / 'translations.sqlite3')))
    parser.add_argument('--json', action='store_true', help='Print English, draft and marked text as JSON.')
    parser.add_argument('--export', type=Path, help='Write a UTF-8 JSON snapshot to this local file.')
    args = parser.parse_args()
    if not args.database.is_file():
        parser.exit(1, 'No saved cache yet. Start the updated server to enable persistence.\n')
    try:
        with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('SELECT english, result FROM translations ORDER BY recency DESC').fetchall()
        entries = []
        for english, payload in rows:
            result = json.loads(payload)
            entries.append({'english': english, 'translation': result['translation'], 'draft': result['draft']})
    except (sqlite3.Error, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Could not read saved cache: {error}\n')
    data = {'count': len(entries), 'order': 'most recently used first', 'translations': entries}
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Exported {len(entries)} translations to {args.export}')
    elif args.json:
        sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        sys.stdout.reconfigure(encoding='utf-8')
        print(f'{len(entries)} saved translations (most recently used first)')
        for index, entry in enumerate(entries, 1):
            print(f"\n{index}. {entry['english']}\n   {entry['translation']}")


if __name__ == '__main__':
    main()
