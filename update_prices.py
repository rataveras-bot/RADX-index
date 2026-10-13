"""Nightly market-data update, run by .github/workflows/update-prices.yml.

Fetches every market source defined in app.MARKETS and upserts it into data/market_data.csv,
then writes data/market_meta.json with the time and per-source status. Exits non-zero only if
nothing at all could be fetched, so one flaky source doesn't fail the job.
"""
import sys

import app

log = app.update_market_data()
for row in log:
    print(f"{row['market']:<10} {row['source']:<28} {row['status']:<8} {row.get('last', '')} {row.get('value', '')}")
ok = sum(r["status"] == "ok" for r in log)
print(f"{ok}/{len(log)} sources updated")
sys.exit(0 if ok else 1)
