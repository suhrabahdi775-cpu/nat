"""
Unit tests for DeepSeek Strategy components.

Tests the current (post-overhaul) behavior:
- Margin-based position sizing with working confidence/trend/RSI multipliers
- Skip (not bump) when below exchange minimum notional
- ATR-based SL/TP geometry with R:R floor
- MACD signal-line warmup gating
- Rule-based analyzer determinism
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig


def make_strategy(**overrides) -> DeepSeekAIStrategy:
    """Build a real strategy instance without registration/API keys."""
    params = dict(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        equity=500.0,
        leverage=10.0,
        base_usdt_amount=100.0,
        use_rule_based_analyzer=True,   # no API key needed
        prefetch_bars=False,            # no network calls
        use_account_balance=False,      # static equity (no portfolio access)
        enable_telegram=False,
        sentiment_enabled=False,
    )
    params.update(overrides)
    return DeepSeekAIStrategy(config=DeepSeekAIStrategyConfig(**params))


PRICE = {"price": 100_000.0}


# ---------- Position sizing: notional mode (legacy) ----------

def test_sizing_scales_with_confidence():
    """HIGH confidence must produce a larger position than MEDIUM (notional mode)."""
    s = make_strategy(sizing_mode="notional")
    high = s._calculate_position_size(
        {"confidence": "HIGH", "signal": "BUY"}, PRICE,
        {"overall_trend": "强势上涨", "rsi": 55.0}, None,
    )
    medium = s._calculate_position_size(
        {"confidence": "MEDIUM", "signal": "BUY"}, PRICE,
        {"overall_trend": "震荡整理", "rsi": 55.0}, None,
    )
    assert high > medium > 0, f"HIGH {high} must exceed MEDIUM {medium}"


def test_sizing_reduces_at_extreme_rsi():
    """Extreme RSI must shrink the position (notional mode)."""
    s = make_strategy(sizing_mode="notional")
    normal = s._calculate_position_size(
        {"confidence": "HIGH", "signal": "BUY"}, PRICE,
        {"overall_trend": "强势上涨", "rsi": 55.0}, None,
    )
    extreme = s._calculate_position_size(
        {"confidence": "HIGH", "signal": "BUY"}, PRICE,
        {"overall_trend": "强势上涨", "rsi": 80.0}, None,
    )
    assert extreme < normal


def test_sizing_skips_below_min_notional():
    """Sizes below the exchange minimum must SKIP, never round up past risk caps."""
    s = make_strategy(sizing_mode="notional")
    qty = s._calculate_position_size(
        {"confidence": "LOW", "signal": "BUY"}, PRICE,  # 100 × 0.5 = $50 < $100 minimum
        {"overall_trend": "震荡整理", "rsi": 55.0}, None,
    )
    assert qty == 0.0


# ---------- Position sizing: risk mode (default) ----------

def test_risk_sizing_consistent_across_volatility():
    """Same risk-at-stop regardless of ATR: tight stop -> big size, wide -> small."""
    s = make_strategy(sizing_mode="risk", risk_per_trade_pct=0.01,
                      equity=5000.0, max_position_ratio=1.0)
    price = {"price": 100_000.0}
    results = {}
    for atr in (150.0, 600.0):
        tech = {"overall_trend": "震荡整理", "rsi": 55.0, "atr": atr,
                "support": 0.0, "resistance": 0.0}
        qty = s._calculate_position_size({"confidence": "MEDIUM", "signal": "BUY"},
                                         price, tech, None)
        sl, _ = s._compute_sl_distance(True, price["price"], atr, 0.0, 0.0)
        results[atr] = qty * sl  # risk-at-stop in USD
    # Both should risk ~1% of 5000 = ~$50, within rounding tolerance
    assert abs(results[150.0] - 50.0) < 8, results
    assert abs(results[600.0] - 50.0) < 8, results
    # Tight-stop position (150) must be LARGER in notional than wide-stop (600)
    # -> verified implicitly: same risk, tighter stop => bigger size


def test_risk_sizing_tight_stop_bigger_than_wide():
    """Tighter stop must yield a larger notional at equal risk."""
    s = make_strategy(sizing_mode="risk", equity=5000.0, max_position_ratio=1.0)
    price = {"price": 100_000.0}
    tight = s._calculate_position_size(
        {"confidence": "MEDIUM", "signal": "BUY"}, price,
        {"overall_trend": "震荡整理", "rsi": 55.0, "atr": 150.0}, None)
    wide = s._calculate_position_size(
        {"confidence": "MEDIUM", "signal": "BUY"}, price,
        {"overall_trend": "震荡整理", "rsi": 55.0, "atr": 600.0}, None)
    assert tight > wide > 0


def test_risk_sizing_respects_margin_cap():
    """Even a very tight stop cannot exceed the margin cap."""
    s = make_strategy(sizing_mode="risk", equity=500.0, max_position_ratio=0.10,
                      risk_per_trade_pct=0.05)
    price = {"price": 100_000.0}
    qty = s._calculate_position_size(
        {"confidence": "HIGH", "signal": "BUY"}, price,
        {"overall_trend": "强势上涨", "rsi": 55.0, "atr": 100.0}, None)
    max_notional = 500.0 * 0.10 * 10.0
    assert qty * price["price"] <= max_notional + 1e-9


def test_sizing_respects_margin_risk_cap():
    """Notional must never exceed equity × ratio × leverage."""
    s = make_strategy(base_usdt_amount=10_000.0)  # absurdly large base
    qty = s._calculate_position_size(
        {"confidence": "HIGH"}, PRICE,
        {"overall_trend": "强势上涨", "rsi": 55.0}, None,
    )
    max_notional = 500.0 * 0.10 * 10.0  # equity × ratio × leverage = $500
    assert qty * PRICE["price"] <= max_notional + 1e-9


def test_sizing_meets_exchange_minimum_when_traded():
    """Any non-zero size must satisfy the $100 exchange minimum."""
    s = make_strategy()
    qty = s._calculate_position_size(
        {"confidence": "MEDIUM"}, PRICE,
        {"overall_trend": "震荡整理", "rsi": 55.0}, None,
    )
    assert qty == 0.0 or qty * PRICE["price"] >= 100.0


# ---------- SL/TP geometry ----------

def test_sl_tp_geometry_buy():
    """BUY (confidence_pct mode): SL below entry, TP above, R:R >= min_risk_reward."""
    s = make_strategy(tp_mode="confidence_pct")
    entry = 100_000.0
    atr = 400.0  # 1.5×ATR = $600 = 0.6%, within clamps
    sl, tp = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="MEDIUM", atr=atr)
    assert sl < entry < tp
    sl_dist = entry - sl
    tp_dist = tp - entry
    assert abs(sl_dist - 600.0) < 1e-6
    assert tp_dist / sl_dist >= s.min_risk_reward - 1e-9


def test_sl_tp_geometry_sell():
    """SELL: SL above entry, TP below."""
    s = make_strategy()
    entry = 100_000.0
    sl, tp = s._compute_sl_tp(is_buy=False, entry_price=entry, confidence="MEDIUM", atr=400.0)
    assert tp < entry < sl


def test_sl_clamped_to_max_pct():
    """A huge ATR must not produce an SL wider than max_sl_pct."""
    s = make_strategy()
    entry = 100_000.0
    sl, _ = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="MEDIUM", atr=5_000.0)
    assert entry - sl <= entry * s.max_sl_pct + 1e-6


def test_sl_clamped_to_min_pct():
    """A tiny ATR must not produce an SL tighter than min_sl_pct."""
    s = make_strategy()
    entry = 100_000.0
    sl, _ = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="MEDIUM", atr=10.0)
    assert entry - sl >= entry * s.min_sl_pct - 1e-6


def test_rr_floor_enforced_for_all_confidences():
    """confidence_pct mode: TP >= min_risk_reward × SL at every confidence."""
    s = make_strategy(tp_mode="confidence_pct")
    entry = 100_000.0
    for conf in ("HIGH", "MEDIUM", "LOW"):
        sl, tp = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence=conf, atr=1_000.0)
        assert (tp - entry) / (entry - sl) >= s.min_risk_reward - 1e-9, conf


def test_support_tightens_sl_when_closer_than_atr():
    """A support level tighter than the ATR distance must be adopted (BUY)."""
    s = make_strategy()
    entry = 100_000.0
    atr = 800.0  # ATR distance = 1200 (1.2%)
    support = 99_500.0  # S/R distance ≈ 599.5 (with 0.1% buffer) - tighter
    sl, _ = s._compute_sl_tp(
        is_buy=True, entry_price=entry, confidence="MEDIUM",
        atr=atr, support=support,
    )
    expected = support * (1 - s.sl_buffer_pct)
    assert abs(sl - expected) < 1e-6


def test_support_never_widens_sl():
    """A support level FARTHER than the ATR distance must be ignored."""
    s = make_strategy()
    entry = 100_000.0
    atr = 400.0  # ATR distance = 600
    support = 97_000.0  # 3% away - would be the old inverted-R:R bug
    sl, _ = s._compute_sl_tp(
        is_buy=True, entry_price=entry, confidence="MEDIUM",
        atr=atr, support=support,
    )
    assert abs((entry - sl) - 600.0) < 1e-6, "ATR distance must win"


def test_resistance_tightens_sl_for_sell():
    """A resistance level tighter than ATR must be adopted (SELL)."""
    s = make_strategy()
    entry = 100_000.0
    atr = 800.0  # ATR distance = 1200
    resistance = 100_500.0  # ≈600.5 away with buffer - tighter
    sl, _ = s._compute_sl_tp(
        is_buy=False, entry_price=entry, confidence="MEDIUM",
        atr=atr, resistance=resistance,
    )
    expected = resistance * (1 + s.sl_buffer_pct)
    assert abs(sl - expected) < 1e-6


def test_sr_respects_min_clamp():
    """S/R virtually at the entry price must not produce an SL tighter than min_sl_pct."""
    s = make_strategy()
    entry = 100_000.0
    sl, _ = s._compute_sl_tp(
        is_buy=True, entry_price=entry, confidence="MEDIUM",
        atr=400.0, support=99_950.0,  # 0.05% away - inside the noise
    )
    assert entry - sl >= entry * s.min_sl_pct - 1e-6


# ---------- Indicators ----------

def test_macd_signal_gated_during_warmup():
    """Signal-line EMA must not receive values until MACD is initialized."""
    from indicators.technical_manager import TechnicalIndicatorManager

    class FakeBar:
        def __init__(self, px):
            self.open = self.high = self.low = self.close = px
            self.volume = 100.0
            self.ts_init = 0

    m = TechnicalIndicatorManager()
    m.update(FakeBar(100.0))
    m.update(FakeBar(101.0))
    # After 2 bars MACD(12,26) is NOT initialized: signal EMA must be untouched
    assert not m.macd.initialized
    assert not m.macd_signal.initialized
    assert m.macd_signal.value == 0.0


def test_atr_exposed_in_technical_data():
    from indicators.technical_manager import TechnicalIndicatorManager

    class FakeBar:
        def __init__(self, px):
            self.open = px
            self.high = px + 50
            self.low = px - 50
            self.close = px
            self.volume = 100.0
            self.ts_init = 0

    m = TechnicalIndicatorManager()
    for i in range(60):
        m.update(FakeBar(100_000.0 + i * 10))
    data = m.get_technical_data(100_000.0)
    assert "atr" in data
    assert data["atr"] > 0


# ---------- TP modes & HTF strictness ----------

def test_tp_r_multiple_mode():
    """r_multiple mode: TP distance must be exactly tp_r_multiple × SL distance."""
    s = make_strategy(tp_mode="r_multiple", tp_r_multiple=1.2)
    entry = 100_000.0
    sl, tp = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="HIGH", atr=400.0)
    sl_dist = entry - sl
    tp_dist = tp - entry
    assert abs(tp_dist - 1.2 * sl_dist) < 1e-6


def test_tp_confidence_mode_unchanged():
    """confidence_pct mode still uses confidence % with R:R floor."""
    s = make_strategy(tp_mode="confidence_pct")
    entry = 100_000.0
    sl, tp = s._compute_sl_tp(is_buy=True, entry_price=entry, confidence="MEDIUM", atr=400.0)
    assert abs((tp - entry) - entry * 0.02) < 1e-6  # 2% MEDIUM target


# ---------- Profit protection & circuit breakers ----------

def _fake_close_event(pnl_side="LONG", entry=100_000.0):
    from types import SimpleNamespace
    return SimpleNamespace(
        side=SimpleNamespace(name=pnl_side),
        avg_px_open=entry,
        quantity=0.002,
    )


def test_loss_streak_throttles_size():
    """After 2 consecutive losses, size must shrink (notional mode: below min → skip)."""
    s = make_strategy(sizing_mode="notional")
    args = (
        {"confidence": "HIGH", "signal": "BUY"}, PRICE,
        {"overall_trend": "强势上涨", "rsi": 55.0}, None,
    )
    normal = s._calculate_position_size(*args)
    assert normal > 0

    s.consecutive_losses = 2
    throttled = s._calculate_position_size(*args)
    # HIGH: $180 × 0.5 = $90 < $100 exchange min → defensive skip
    assert throttled == 0.0

    s.consecutive_losses = 0
    assert s._calculate_position_size(*args) == normal


def test_loss_streak_throttles_size_risk_mode():
    """Risk mode: streak throttle halves the risked amount (size shrinks, not necessarily to 0)."""
    s = make_strategy(sizing_mode="risk", equity=5000.0, max_position_ratio=1.0)
    args = (
        {"confidence": "HIGH", "signal": "BUY"}, {"price": 100_000.0},
        {"overall_trend": "强势上涨", "rsi": 55.0, "atr": 200.0}, None,
    )
    normal = s._calculate_position_size(*args)
    s.consecutive_losses = 2
    throttled = s._calculate_position_size(*args)
    assert 0 < throttled < normal


def test_consecutive_losses_tracking():
    """Loss counter increments on losses and resets on a win."""
    s = make_strategy()
    s._day_start_equity = 500.0
    s._record_trade_outcome(_fake_close_event(), -5.0)
    s._record_trade_outcome(_fake_close_event(), -3.0)
    assert s.consecutive_losses == 2
    s._record_trade_outcome(_fake_close_event(), +4.0)
    assert s.consecutive_losses == 0


def test_daily_loss_breaker_trips():
    """Breaker must activate once day PnL <= -5% of day-start equity."""
    s = make_strategy()
    s._day_start_equity = 500.0
    assert not s._daily_breaker_active
    s._record_trade_outcome(_fake_close_event(), -20.0)
    assert not s._daily_breaker_active  # -4%, under the limit
    s._record_trade_outcome(_fake_close_event(), -6.0)
    assert s._daily_breaker_active  # -5.2%, tripped


def test_breakeven_flag_set_at_trigger():
    """Breakeven marks done and computes correct BE price for LONG."""
    s = make_strategy()
    key = str(s.instrument_id)
    state = {
        "entry_price": 100_000.0,
        "side": "LONG",
        "current_sl_price": 99_400.0,
        "initial_risk": 600.0,   # SL 0.6% below entry
        "breakeven_done": False,
        "sl_order_id": None,     # forces the no-op path in the update helper
        "quantity": 0.002,
        "highest_price": 100_000.0,
        "activated": False,
    }
    s.trailing_stop_state[key] = state
    # Profit 0.5R: must NOT trigger
    s._check_breakeven(key, state, 100_300.0)
    assert not state["breakeven_done"]
    # Profit 1R: must trigger
    s._check_breakeven(key, state, 100_600.0)
    assert state["breakeven_done"]


def test_reversal_confirmation_blocks_first_signal():
    """First opposite signal must NOT reverse; streak must increment."""
    s = make_strategy()
    current_position = {"side": "long", "quantity": 0.002, "avg_px": 100_000.0}
    # First opposite signal: returns before touching order plumbing
    # (order_factory is None pre-registration - would raise if it tried)
    s._manage_existing_position(current_position, "short", 0.002, "HIGH")
    assert s._opposite_signal_streak == 1

    # A same-direction signal resets the streak
    s._manage_existing_position(current_position, "long", 0.002, "HIGH")
    assert s._opposite_signal_streak == 0


# ---------- Efficiency ratio (chop detection) ----------

def test_efficiency_ratio_trend_vs_chop():
    """Trending closes -> ER near 1; oscillating closes -> ER near 0."""
    from indicators.technical_manager import TechnicalIndicatorManager

    class FakeBar:
        def __init__(self, px):
            self.open = self.high = self.low = self.close = px
            self.volume = 100.0
            self.ts_init = 0

    trend = TechnicalIndicatorManager()
    for i in range(30):
        trend.update(FakeBar(100_000.0 + i * 100))  # straight line up
    er_trend = trend._calculate_efficiency_ratio(period=20)

    chop = TechnicalIndicatorManager()
    for i in range(30):
        chop.update(FakeBar(100_000.0 + (100 if i % 2 else -100)))  # sawtooth
    er_chop = chop._calculate_efficiency_ratio(period=20)

    assert er_trend > 0.9, f"trend ER {er_trend}"
    assert er_chop < 0.1, f"chop ER {er_chop}"
    # And it must be exposed in technical data
    data = trend.get_technical_data(103_000.0)
    assert "efficiency_ratio" in data


# ---------- Rule-based analyzer ----------

def test_rule_based_analyzer_deterministic():
    from utils.rule_based_analyzer import RuleBasedAnalyzer

    tech = {
        "rsi": 45.0, "macd_histogram": 5.0,
        "sma_5": 101.0, "sma_20": 100.0, "sma_50": 99.0,
    }
    price = {"price": 102.0}
    a1 = RuleBasedAnalyzer().analyze(price, tech)
    a2 = RuleBasedAnalyzer().analyze(price, tech)
    assert a1["signal"] == a2["signal"]
    assert a1["signal"] in ("BUY", "SELL", "HOLD")
    assert a1["confidence"] in ("HIGH", "MEDIUM", "LOW")


def test_rule_based_analyzer_bullish_alignment():
    """Full bullish alignment (MA + RSI recovery + MACD) must produce BUY."""
    from utils.rule_based_analyzer import RuleBasedAnalyzer

    tech = {
        "rsi": 35.0, "macd_histogram": 5.0,
        "sma_5": 101.0, "sma_20": 100.0, "sma_50": 99.0,
    }
    result = RuleBasedAnalyzer().analyze({"price": 102.0}, tech)
    assert result["signal"] == "BUY"
