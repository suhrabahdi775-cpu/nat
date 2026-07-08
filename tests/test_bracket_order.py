"""
Bracket order SL/TP geometry tests.

The original tests in this file stubbed NautilusTrader internals
(strategy.cache, strategy.log) which are read-only C-level attributes when
the real framework is installed - they could never run in this environment.

The SL/TP computation now lives in DeepSeekAIStrategy._compute_sl_tp (a pure
method), tested here through the real constructor. Order-submission plumbing
is covered end-to-end by backtest/run_backtest.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig


def _make_strategy(**overrides) -> DeepSeekAIStrategy:
    params = dict(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        equity=500.0,
        leverage=10.0,
        base_usdt_amount=100.0,
        use_rule_based_analyzer=True,
        prefetch_bars=False,
        use_account_balance=False,
        enable_telegram=False,
        sentiment_enabled=False,
    )
    params.update(overrides)
    return DeepSeekAIStrategy(config=DeepSeekAIStrategyConfig(**params))


def test_bracket_sl_tp_uses_atr_when_available() -> None:
    """With a valid ATR the SL distance must equal atr_sl_multiplier × ATR."""
    s = _make_strategy()
    entry = 1000.0
    atr = 5.0  # 1.5×5 = 7.5 = 0.75% of entry, inside [0.3%, 1.5%] clamps
    sl, tp = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="HIGH", atr=atr)
    assert abs((entry - sl) - 7.5) < 1e-9
    assert tp > entry


def test_bracket_sl_tp_falls_back_without_atr() -> None:
    """Without ATR (0.0), the SL must use the 1% fallback distance."""
    s = _make_strategy()
    entry = 1000.0
    sl, tp = s._compute_sl_tp(is_buy=False, entry_price=entry, confidence="MEDIUM", atr=0.0)
    assert abs((sl - entry) - 10.0) < 1e-9  # 1% of 1000, SELL: SL above
    assert tp < entry


def test_bracket_tp_confidence_ordering() -> None:
    """Higher confidence must produce a further TP (larger target)."""
    s = _make_strategy()
    entry = 1000.0
    _, tp_high = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="HIGH", atr=5.0)
    _, tp_low = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="LOW", atr=5.0)
    assert tp_high >= tp_low


if __name__ == "__main__":
    test_bracket_sl_tp_uses_atr_when_available()
    test_bracket_sl_tp_falls_back_without_atr()
    test_bracket_tp_confidence_ordering()
    print("All bracket geometry tests passed")
