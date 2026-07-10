"""
Parameter Sweep with Train/Test Split

Sweeps risk-parameter combinations on a TRAIN window, then validates the
best combination on an untouched TEST window. This guards against
curve-fitting: a combination is only trustworthy if it holds up on data
it was not selected on.

Usage:
    python backtest/sweep.py --csv data/BTCUSDT_15m.csv --htf-csv data/BTCUSDT_1h.csv \
        --train-end 2026-05-01

Note: uses the rule-based analyzer (deterministic, offline). Results measure
the RISK/EXECUTION layer, not DeepSeek signal quality.
"""

import argparse
import itertools
import sys
from pathlib import Path

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
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig

# The grid to sweep. Keep it SMALL - every added combination multiplies
# runtime and increases the chance of overfitting the train window.
#
# NOTE: at low-volatility regimes (15m ATR ~0.1% of price) the ATR multiplier
# is dominated by the min_sl_pct clamp - sweeping it changes nothing. Sweep
# the binding parameters instead: the SL floor itself and the exit timing.
GRID = {
    "tp_mode": ["r_multiple"],
    "tp_r_multiple": [1.0, 1.5],
    "htf_strict_alignment": [False, True],
}


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def run_one(bars_15m, bars_1h, equity, base_position, **params) -> dict:
    """Run a single backtest; return summary metrics."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("SWEEP-001"),
            logging=LoggingConfig(log_level="ERROR", print_config=False),
        )
    )
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(equity, USDT)],
    )
    engine.add_instrument(TestInstrumentProvider.btcusdt_perp_binance())
    engine.add_data(bars_15m)
    if bars_1h:
        engine.add_data(bars_1h)

    config = DeepSeekAIStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        equity=equity,
        leverage=10.0,
        base_usdt_amount=base_position,
        use_rule_based_analyzer=True,
        prefetch_bars=False,
        analysis_source="websocket",
        use_order_emulation=False,
        sentiment_enabled=False,
        enable_telegram=False,
        enable_htf_filter=bool(bars_1h),
        analyze_on_bar_close=True,
        use_account_balance=True,
        **params,
    )
    engine.add_strategy(DeepSeekAIStrategy(config=config))
    engine.run()

    positions = engine.trader.generate_positions_report()
    result = {"trades": len(positions), "pnl": 0.0, "win_rate": 0.0}
    if not positions.empty and "realized_pnl" in positions.columns:
        pnls = positions["realized_pnl"].apply(
            lambda x: float(str(x).split(" ")[0]) if x is not None else 0.0
        )
        wins = int((pnls > 0).sum())
        losses = int((pnls < 0).sum())
        result["pnl"] = float(pnls.sum())
        result["win_rate"] = wins / max(wins + losses, 1) * 100

    engine.dispose()
    return result


def child_run(args) -> None:
    """Run ONE combo in this process and print a parseable RESULT line.

    Each combo runs in its own subprocess because NautilusTrader's Rust
    logging can panic when many engines are created in a single process.
    """
    import json

    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    bt_15m = BarType.from_str("BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    bt_1h = BarType.from_str("BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL")

    df_15m = load_df(args.csv)
    df_1h = load_df(args.htf_csv) if args.htf_csv else None
    split = pd.Timestamp(args.train_end)

    if args.window == "train":
        df_15m = df_15m[df_15m.index < split]
        df_1h = df_1h[df_1h.index < split] if df_1h is not None else None
    else:
        df_15m = df_15m[df_15m.index >= split]
        df_1h = df_1h[df_1h.index >= split] if df_1h is not None else None

    bars_15m = BarDataWrangler(bar_type=bt_15m, instrument=instrument).process(df_15m)
    bars_1h = (
        BarDataWrangler(bar_type=bt_1h, instrument=instrument).process(df_1h)
        if df_1h is not None else []
    )

    params = json.loads(args.params)
    result = run_one(bars_15m, bars_1h, args.equity, args.base_position, **params)
    print(f"RESULT {json.dumps(result)}")


def spawn(args, window: str, params: dict) -> dict:
    """Run one combo in a subprocess; parse its RESULT line."""
    import json
    import subprocess

    cmd = [
        sys.executable, __file__,
        "--csv", args.csv,
        "--train-end", args.train_end,
        "--equity", str(args.equity),
        "--base-position", str(args.base_position),
        "--window", window,
        "--params", json.dumps(params),
    ]
    if args.htf_csv:
        cmd += ["--htf-csv", args.htf_csv]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(
        f"Combo {params} on {window} produced no result. "
        f"stderr tail: {proc.stderr[-500:]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep with train/test split")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--htf-csv", default=None)
    parser.add_argument("--train-end", required=True,
                        help="Train/test split date, e.g. 2026-05-01")
    parser.add_argument("--equity", type=float, default=500.0)
    parser.add_argument("--base-position", type=float, default=100.0)
    # Internal: set when running as a single-combo child process
    parser.add_argument("--window", choices=["train", "test"], default=None)
    parser.add_argument("--params", default=None)
    args = parser.parse_args()

    if args.window and args.params:
        child_run(args)
        return

    # --- Sweep on TRAIN (one subprocess per combo) ---
    keys = list(GRID.keys())
    results = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        r = spawn(args, "train", params)
        results.append((params, r))
        print(f"TRAIN {params} -> PnL {r['pnl']:+8.2f} | {r['trades']:3d} trades | "
              f"win {r['win_rate']:.1f}%", flush=True)

    # --- Validate best on TEST ---
    best_params, best_train = max(results, key=lambda x: x[1]["pnl"])
    default_params = {}  # empty = strategy config defaults

    print(f"\nBest on train: {best_params} (PnL {best_train['pnl']:+.2f})")
    test_best = spawn(args, "test", best_params)
    test_default = spawn(args, "test", default_params)

    print(f"TEST  best    {best_params} -> PnL {test_best['pnl']:+8.2f} | "
          f"{test_best['trades']:3d} trades | win {test_best['win_rate']:.1f}%")
    print(f"TEST  default (config defaults) -> PnL {test_default['pnl']:+8.2f} | "
          f"{test_default['trades']:3d} trades | win {test_default['win_rate']:.1f}%")

    print(
        "\n⚠️  Only adopt the 'best' parameters if they ALSO outperform on TEST.\n"
        "    If they win on train but lose on test, that's curve-fitting - keep defaults."
    )


if __name__ == "__main__":
    main()
