"""
Rule-Based Analyzer (drop-in replacement for DeepSeekAnalyzer)

Generates deterministic trading signals from the same technical data the AI
sees, using the same decision framework described in the DeepSeek prompt
(MA alignment + RSI + MACD, 2-of-3 style confirmation).

Purpose: backtesting. It lets the full execution/risk/sizing pipeline be
validated offline without API calls, costs, or non-determinism. It is NOT
intended to be a profitable strategy by itself - it is a harness.
"""

from datetime import datetime
from typing import Any, Dict, Optional


class RuleBasedAnalyzer:
    """
    Deterministic signal generator mirroring the DeepSeekAnalyzer interface.
    """

    def __init__(self):
        self.signal_history = []

    def analyze(
        self,
        price_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None,
        current_position: Optional[Dict[str, Any]] = None,
        trade_history: Optional[list] = None,
        funding_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a signal from technical conditions (no API call)."""
        price = float(price_data['price'])
        rsi = float(technical_data.get('rsi', 50.0))
        macd_hist = float(technical_data.get('macd_histogram', 0.0))
        sma_5 = technical_data.get('sma_5')
        sma_20 = technical_data.get('sma_20')
        sma_50 = technical_data.get('sma_50')

        bull_votes = 0
        bear_votes = 0

        # Vote 1: MA alignment
        if sma_5 and sma_20 and sma_50:
            if price > sma_5 > sma_20 > sma_50:
                bull_votes += 1
            elif price < sma_5 < sma_20 < sma_50:
                bear_votes += 1

        # Vote 2: RSI momentum (recovering from oversold / falling from overbought)
        if 40 <= rsi <= 60:
            pass  # neutral zone - no vote
        elif rsi < 40:
            bull_votes += 1  # oversold recovery zone
        elif rsi > 60:
            bear_votes += 1  # overbought exhaustion zone

        # Vote 3: MACD histogram direction
        if macd_hist > 0:
            bull_votes += 1
        elif macd_hist < 0:
            bear_votes += 1

        # Optional vote: sentiment alignment
        if sentiment_data:
            net = sentiment_data.get('net_sentiment', 0.0)
            if net > 0.1:
                bull_votes += 1
            elif net < -0.1:
                bear_votes += 1

        # Decide: require a 2-vote lead for a directional signal
        if bull_votes >= 2 and bull_votes - bear_votes >= 2:
            signal = "BUY"
            confidence = "HIGH" if bull_votes >= 3 else "MEDIUM"
        elif bear_votes >= 2 and bear_votes - bull_votes >= 2:
            signal = "SELL"
            confidence = "HIGH" if bear_votes >= 3 else "MEDIUM"
        else:
            signal = "HOLD"
            confidence = "LOW"

        signal_data = {
            "signal": signal,
            "confidence": confidence,
            "reason": (
                f"Rule-based: bull_votes={bull_votes}, bear_votes={bear_votes}, "
                f"rsi={rsi:.1f}, macd_hist={macd_hist:.4f}"
            ),
            "trend_strength": "MODERATE",
            "risk_assessment": "MEDIUM",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.signal_history.append(signal_data)
        if len(self.signal_history) > 30:
            self.signal_history.pop(0)

        return signal_data
