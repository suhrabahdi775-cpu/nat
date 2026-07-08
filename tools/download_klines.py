"""
Binance Klines Downloader for Backtesting

Downloads historical klines to CSV for use with backtest/run_backtest.py.

Two sources:
- vision (default): https://data.binance.vision public monthly dumps.
  This is a static CDN and is typically reachable even where the trading
  API (fapi.binance.com) is geo-restricted.
- api: live REST endpoint (requires an unrestricted network).

Usage:
    python tools/download_klines.py --symbol BTCUSDT --interval 15m --months 2026-01 2026-06
    python tools/download_klines.py --symbol BTCUSDT --interval 1h --months 2026-01 2026-06
    python tools/download_klines.py --source api --symbol BTCUSDT --interval 15m --days 30
"""

import argparse
import csv
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API_URL = "https://fapi.binance.com/fapi/v1/klines"
VISION_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
MAX_LIMIT = 1500


def _month_range(start: str, end: str):
    """Yield YYYY-MM strings from start to end inclusive."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def _parse_ts(raw: str) -> datetime:
    """Parse a Binance kline open_time (ms or µs epoch)."""
    ts = int(raw)
    if ts > 10**14:  # microseconds
        ts //= 1000
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def download_vision(symbol: str, interval: str, start_month: str, end_month: str, out_path: Path):
    """Download monthly kline dumps from data.binance.vision."""
    rows = []
    for month in _month_range(start_month, end_month):
        url = VISION_URL.format(symbol=symbol, interval=interval, month=month)
        print(f"  fetching {month}...", end=" ", flush=True)
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            print("not available (skipped)")
            continue
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                count = 0
                for line in csv.reader(text):
                    if not line or not line[0].strip():
                        continue
                    if not line[0].isdigit():  # header row in newer dumps
                        continue
                    ts = _parse_ts(line[0])
                    rows.append([
                        ts.strftime("%Y-%m-%d %H:%M:%S"),
                        line[1], line[2], line[3], line[4], line[5],
                    ])
                    count += 1
        print(f"{count} bars")

    if not rows:
        raise SystemExit("No data downloaded - check symbol/interval/months")

    rows.sort(key=lambda r: r[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} bars to {out_path} ({rows[0][0]} → {rows[-1][0]})")


def download(symbol: str, interval: str, days: int, out_path: Path):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            API_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": MAX_LIMIT,
            },
            timeout=15,
        )
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break

        for k in klines:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
            rows.append([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                k[1], k[2], k[3], k[4], k[5],
            ])

        cursor = klines[-1][0] + 1
        print(f"  fetched {len(rows)} bars (up to {rows[-1][0]})")
        time.sleep(0.3)  # stay well under rate limits

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} bars to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Binance futures klines")
    parser.add_argument("--source", choices=["vision", "api"], default="vision",
                        help="vision = public dumps CDN (works under geo-block); api = live REST")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m", help="e.g. 15m, 1h")
    parser.add_argument("--days", type=int, default=30, help="(api source only)")
    parser.add_argument("--months", nargs=2, metavar=("START", "END"),
                        default=["2026-01", "2026-06"],
                        help="(vision source) YYYY-MM range, e.g. 2026-01 2026-06")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(
        f"data/{args.symbol}_{args.interval}.csv"
    )

    if args.source == "vision":
        print(f"Downloading {args.symbol} {args.interval} monthly dumps "
              f"{args.months[0]} → {args.months[1]} from data.binance.vision...")
        download_vision(args.symbol, args.interval, args.months[0], args.months[1], out)
    else:
        print(f"Downloading {args.symbol} {args.interval} klines for {args.days} days...")
        download(args.symbol, args.interval, args.days, out)


if __name__ == "__main__":
    main()
