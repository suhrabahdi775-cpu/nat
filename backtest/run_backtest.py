"""
Backtest Harness for the DeepSeek AI Strategy

Runs the EXACT same strategy class used in live trading against historical
klines, with the rule-based analyzer replacing the DeepSeek API (deterministic,
free, offline). This validates the execution pipeline end-to-end: sizing,
bracket SL/TP geometry, reversals, trailing stops, partial TPs, cooldowns,
and the HTF filter.

Usage:
    python backtest/run_backtest.py --csv data/BTCUSDT_15m.csv
    python backtest/run_backtest.py --csv data/BTCUSDT_15m.csv --htf-csv data/BTCUSDT_1h.csv

    # Measure REAL DeepSeek signals (API calls, cached to disk for replay):
    python backtest/run_backtest.py --csv data/BTCUSDT_15m.csv --htf-csv data/BTCUSDT_1h.csv \
        --analyzer deepseek --cache-file data/deepseek_cache.json \
        --start 2026-06-24 --end 2026-07-01 --max-api-calls 700

CSV format (from tools/download_klines.py):
    timestamp,open,high,low,close,volume
    2025-01-01 00:00:00,93576.0,93650.1,93400.0,93521.3,412.5
"""

import argparse
import sys
from decimal import Decimal
from pathlib import Path

# Make project modules importable when run from repo root or backtest/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from backtest.instrument import btcusdt_perp_real_fees

from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig


def load_bars_from_csv(csv_path: str, bar_type: BarType, instrument, start=None, end=None):
    """Load a klines CSV into Nautilus Bar objects, optionally date-sliced."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index < pd.Timestamp(end)]

    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    return wrangler.process(df)


def main():
    parser = argparse.ArgumentParser(description="Backtest the DeepSeek AI strategy")
    parser.add_argument("--csv", required=True, help="15m klines CSV path")
    parser.add_argument("--htf-csv", default=None, help="1h klines CSV path (optional, enables HTF filter)")
    parser.add_argument("--equity", type=float, default=500.0, help="Starting USDT balance")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--base-position", type=float, default=100.0, help="Base position notional (USDT)")
    parser.add_argument("--risk-pct", type=float, default=None,
                        help="Override risk_per_trade_pct (e.g. 0.02 = 2%% of equity per trade)")
    parser.add_argument("--max-ratio", type=float, default=None,
                        help="Override max_position_ratio (margin cap as fraction of equity)")
    parser.add_argument("--partial-tp", choices=["on", "off"], default=None,
                        help="on = laddered partial TPs; off = single full-size bracket TP")
    parser.add_argument("--log-level", default="ERROR", help="Engine log level (ERROR keeps output readable)")
    parser.add_argument("--start", default=None, help="Window start date, e.g. 2026-06-24")
    parser.add_argument("--end", default=None, help="Window end date (exclusive)")
    parser.add_argument("--analyzer", choices=["rule", "deepseek"], default="rule",
                        help="rule = offline baseline; deepseek = REAL API signals (cached)")
    parser.add_argument("--cache-file", default="data/deepseek_cache.json",
                        help="Signal cache path for --analyzer deepseek")
    parser.add_argument("--max-api-calls", type=int, default=300,
                        help="Spend cap per run for --analyzer deepseek")
    args = parser.parse_args()

    if args.analyzer == "deepseek":
        from dotenv import load_dotenv
        load_dotenv()
        import os
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise SystemExit("DEEPSEEK_API_KEY not found in environment/.env")

    # --- Engine ---
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level=args.log_level),
        )
    )

    # --- Venue: Binance Futures-like margin account, netting OMS ---
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(args.equity, USDT)],
    )

    # --- Instrument (matches the live instrument ID exactly) ---
    instrument = btcusdt_perp_real_fees()
    engine.add_instrument(instrument)

    # --- Data ---
    # Warmup padding: indicators need ~50x15m bars and the HTF filter needs
    # ~50x1h bars before trading can start; pull the window start back so
    # the requested range is fully tradable.
    warmup_start = None
    if args.start:
        warmup_start = str(pd.Timestamp(args.start) - pd.Timedelta(hours=60))

    bar_type = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    bars = load_bars_from_csv(args.csv, bar_type, instrument, warmup_start, args.end)
    print(f"Loaded {len(bars)} x 15m bars from {args.csv}")
    engine.add_data(bars)

    enable_htf = args.htf_csv is not None
    if enable_htf:
        htf_bar_type = BarType.from_str("BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL")
        htf_bars = load_bars_from_csv(args.htf_csv, htf_bar_type, instrument, warmup_start, args.end)
        print(f"Loaded {len(htf_bars)} x 1h bars from {args.htf_csv}")
        engine.add_data(htf_bars)

    # --- Strategy: same class as live; analyzer per --analyzer ---
    config = DeepSeekAIStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        equity=args.equity,
        leverage=args.leverage,
        base_usdt_amount=args.base_position,
        use_rule_based_analyzer=(args.analyzer == "rule"),
        deepseek_cache_file=(args.cache_file if args.analyzer == "deepseek" else ""),
        deepseek_max_api_calls=args.max_api_calls,
        prefetch_bars=False,            # history comes from the engine
        analysis_source="websocket",    # bars come from the backtest engine, not REST
        use_order_emulation=False,      # bars-only data starves the emulator
        sentiment_enabled=False,        # no live API calls in backtest
        enable_telegram=False,
        enable_htf_filter=enable_htf,
        analyze_on_bar_close=True,
        use_account_balance=True,
        **({"risk_per_trade_pct": args.risk_pct} if args.risk_pct is not None else {}),
        **({"max_position_ratio": args.max_ratio} if args.max_ratio is not None else {}),
        **({"enable_partial_tp": args.partial_tp == "on"} if args.partial_tp is not None else {}),
    )
    strategy = DeepSeekAIStrategy(config=config)
    engine.add_strategy(strategy)

    # --- Run ---
    engine.run()

    if hasattr(strategy.deepseek, "stats"):
        print(f"\n{strategy.deepseek.stats()}")

    # --- Report ---
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    account = engine.trader.generate_account_report(BINANCE)
    if not account.empty:
        final_balance = account.iloc[-1]
        print(f"Final account state:\n{final_balance}\n")

    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    print(f"Orders filled: {len(fills)}")
    print(f"Positions: {len(positions)}")

    if not positions.empty and "realized_pnl" in positions.columns:
        pnls = positions["realized_pnl"].apply(
            lambda x: float(str(x).split(" ")[0]) if x is not None else 0.0
        )
        wins = (pnls > 0).sum()
        losses = (pnls < 0).sum()
        total = pnls.sum()
        win_rate = wins / max(wins + losses, 1) * 100
        # Max drawdown on the cumulative realized-PnL curve (equity proxy)
        equity_curve = args.equity + pnls.cumsum()
        running_peak = equity_curve.cummax()
        drawdown = (equity_curve - running_peak) / running_peak
        max_dd = drawdown.min() * 100  # most negative
        # Worst consecutive losing streak
        streak = worst = 0
        for p in pnls:
            streak = streak + 1 if p < 0 else 0
            worst = max(worst, streak)
        print(f"Win/Loss: {wins}/{losses} ({win_rate:.1f}% win rate)")
        print(f"Total realized PnL: {total:+.2f} USDT ({total/args.equity*100:+.1f}% of ${args.equity:.0f})")
        print(f"Max drawdown: {max_dd:.1f}% | Worst losing streak: {worst}")
        if losses > 0 and wins > 0:
            avg_win = pnls[pnls > 0].mean()
            avg_loss = abs(pnls[pnls < 0].mean())
            print(f"Avg win: {avg_win:.2f} | Avg loss: {avg_loss:.2f} "
                  f"| Realized R:R: {avg_win/avg_loss:.2f}")

    # Persist full reports for inspection
    out_dir = Path("logs/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    fills.to_csv(out_dir / "fills.csv")
    positions.to_csv(out_dir / "positions.csv")
    account.to_csv(out_dir / "account.csv")
    print(f"\nDetailed reports written to {out_dir}/")

    engine.dispose()


if __name__ == "__main__":
    main()
