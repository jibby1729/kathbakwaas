"""Read local traffic counts; no public analytics endpoint or network requests."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.analytics import summary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checks', action='store_true', help='Show verification requests instead of visitor traffic.')
parser.add_argument('--json', action='store_true')
parser.add_argument('--database', type=Path, default=ROOT / '.hosting' / 'traffic.sqlite3')
args = parser.parse_args()
if not args.database.exists():
    sys.exit('Traffic counting has not started yet.')
data = summary(args.database, args.checks)
if args.json:
    print(json.dumps(data, indent=2))
else:
    from datetime import datetime
    print('Kath Bath traffic')
    print('Counting since:', datetime.fromisoformat(data['enabled_at_utc']).astimezone().strftime('%Y-%m-%d %I:%M %p %Z'))
    for label, key in [('Page loads', 'page_loads'), ('Approximate visitors (distinct IPs)', 'approximate_unique_ips'),
                       ('Successful translations', 'successful_translations'), ('  Fresh', 'fresh_translations'),
                       ('  Cached', 'cached_translations'), ('Busy translation responses', 'busy_translations'),
                       ('Cancelled translation requests', 'cancelled_translations'), ('Translation server errors', 'server_error_translations')]:
        print(f'{label}: {data[key]}')
    print('Verification traffic only.' if args.checks else 'Verification requests excluded. IPs are not individual people; previews/bots can count.')
    print('Prefill, assets, health polling and browser-only saved answers are not translations in this count.')
