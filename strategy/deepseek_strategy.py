"""
DeepSeek AI Strategy for NautilusTrader

AI-powered cryptocurrency trading strategy using DeepSeek for decision making,
technical indicators for market analysis, and sentiment data for validation.
"""

import os
import asyncio
import threading
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce, PositionSide, PriceType, TriggerType, OrderType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.position import Position
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.indicators import SimpleMovingAverage
from datetime import timedelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.technical_manager import TechnicalIndicatorManager
from utils.deepseek_client import DeepSeekAnalyzer
from utils.sentiment_client import SentimentDataFetcher
# OCOManager no longer needed - using NautilusTrader's built-in bracket orders


class DeepSeekAIStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for DeepSeek AI Strategy."""

    # Instrument
    instrument_id: str
    bar_type: str

    # Capital
    equity: float = 10000.0
    leverage: float = 10.0

    # Position sizing
    base_usdt_amount: float = 100.0
    high_confidence_multiplier: float = 1.5
    medium_confidence_multiplier: float = 1.0
    low_confidence_multiplier: float = 0.5
    max_position_ratio: float = 0.10
    trend_strength_multiplier: float = 1.2
    min_trade_amount: float = 0.001

    # Technical indicators
    sma_periods: Tuple[int, ...] = (5, 20, 50)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    bb_period: int = 20
    bb_std: float = 2.0

    # AI configuration
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_temperature: float = 0.1
    deepseek_max_retries: int = 2

    # Sentiment
    sentiment_enabled: bool = True
    sentiment_lookback_hours: int = 4
    sentiment_timeframe: str = "15m"  # Sentiment data timeframe (should match or be compatible with bar_type)

    # Risk management
    min_confidence_to_trade: str = "MEDIUM"
    allow_reversals: bool = True
    require_high_confidence_for_reversal: bool = False
    rsi_extreme_threshold_upper: float = 75.0
    rsi_extreme_threshold_lower: float = 25.0
    rsi_extreme_multiplier: float = 0.7
    
    # Stop Loss & Take Profit
    enable_auto_sl_tp: bool = True
    sl_use_support_resistance: bool = True
    sl_buffer_pct: float = 0.001
    tp_high_confidence_pct: float = 0.03
    tp_medium_confidence_pct: float = 0.02
    tp_low_confidence_pct: float = 0.01

    # Position sizing mode:
    # "risk"     - size so a stop-out loses risk_per_trade_pct × equity,
    #              scaled by confidence/trend/RSI/streak. Notional adapts to
    #              the ATR stop distance (tight stop -> bigger position).
    # "notional" - legacy: fixed base_usdt_amount × multipliers.
    sizing_mode: str = "risk"
    risk_per_trade_pct: float = 0.01  # 1% of equity risked at the stop (MEDIUM conf)

    # ATR-based adaptive SL/TP (preferred over support/resistance when ATR ready)
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    min_sl_pct: float = 0.003   # SL never tighter than 0.3% of entry
    max_sl_pct: float = 0.015   # SL never wider than 1.5% of entry
    min_risk_reward: float = 1.5  # TP distance >= this multiple of SL distance

    # Higher-timeframe trend filter (1h) - blocks counter-trend entries
    enable_htf_filter: bool = True
    htf_sma_fast: int = 20
    htf_sma_slow: int = 50

    # Cooldown after a losing trade (bars of primary timeframe)
    loss_cooldown_bars: int = 2

    # Breakeven stop: once profit >= trigger_r × initial risk, move SL to
    # entry ± buffer so the trade can no longer lose
    enable_breakeven_stop: bool = True
    breakeven_trigger_r: float = 1.0
    breakeven_buffer_pct: float = 0.0005  # covers round-trip fees

    # Daily loss circuit breaker: halt NEW entries after realized losses
    # exceed this fraction of day-start equity (UTC day). 0 disables.
    daily_loss_limit_pct: float = 0.05

    # Loss-streak size throttle: after N consecutive losses, scale position
    # size by the multiplier until the next winning trade. 0 disables.
    loss_streak_threshold: int = 2
    loss_streak_multiplier: float = 0.5
    # Streak decay: reset the streak after this many bars WITHOUT any closed
    # trade. Prevents a deadlock on small accounts where the throttled size
    # falls below the exchange minimum - no trade can occur, so no win can
    # ever reset the streak. 96 bars = 24h on 15m.
    loss_streak_reset_bars: int = 96

    # Stagnant position exit: close positions older than this many bars
    # that never activated the trailing stop (never reached +1%). 0 disables.
    max_position_age_bars: int = 24  # 6h on 15m bars

    # Reversal confirmation: require this many CONSECUTIVE opposite signals
    # before reversing a position (cuts whipsaw churn). 1 = immediate.
    reversal_confirmation_signals: int = 2

    # Dead-market filter: skip NEW entries when ATR as a fraction of price
    # is below this (chop - fees eat any edge). 0 disables.
    min_atr_pct_to_trade: float = 0.001

    # Chop filter: skip NEW entries when the Kaufman efficiency ratio is
    # below this. ATR measures volatility, not direction - a violent ranging
    # market passes the ATR filter but whipsaws trend entries. 0 disables.
    min_efficiency_ratio: float = 0.25

    # Stale-signal guard (live REST mode only): DeepSeek takes 30-70s to
    # answer; with sub-0.5% targets an adverse move during the call ruins
    # the trade geometry before entry. Re-check the price after the AI
    # returns and skip the entry if it moved against the signal by more
    # than this fraction. 0 disables.
    max_signal_staleness_pct: float = 0.0015

    # Take-profit mode:
    # "confidence_pct" - TP = confidence-based % (1-3%) with min_risk_reward floor
    # "r_multiple"     - TP = tp_r_multiple × SL distance (closer targets,
    #                    higher win rate, smaller average win)
    # Defaults re-validated July 2026 under REAL Binance fees (taker 0.05%,
    # not the 0.018% the test instrument models): 1R targets cannot clear
    # the 0.10% round-trip cost (breakeven win rate 66.7%); 2R targets +
    # the efficiency-ratio chop filter were positive on BOTH train and test
    # windows. The earlier 1R recommendation was an artifact of under-
    # modeled fees.
    tp_mode: str = "r_multiple"
    tp_r_multiple: float = 2.0

    # HTF strict alignment: entries must be WITH the 1h trend (BUY only in
    # UPTREND, SELL only in DOWNTREND; NEUTRAL blocks new entries).
    # False merely blocks counter-trend entries.
    htf_strict_alignment: bool = True

    # Flatten on strategy stop. IMPORTANT: SL/TP are emulated (local to the
    # bot) - when the bot is offline an open position has NO exchange-side
    # protection, so holding through a shutdown is unprotected exposure.
    close_positions_on_stop: bool = True

    # Sizing source: use live account balance when available
    use_account_balance: bool = True

    # Analysis trigger: on bar close (fresh data) vs wall-clock timer
    analyze_on_bar_close: bool = True

    # Data source for analysis:
    # "rest"      - poll Binance REST klines on a timer and rebuild indicators
    #               each cycle. RELIABLE: the kline WebSocket push via the
    #               nautilus Binance adapter has been observed to silently
    #               stop delivering bars (bot ran 16h, subscribed, received
    #               zero bars -> zero trades). REST always returns fresh bars.
    # "websocket" - event-driven on live bars (subject to that flakiness).
    analysis_source: str = "rest"

    # Analyzer: rule-based (deterministic, no API) for backtesting
    use_rule_based_analyzer: bool = False

    # Cached DeepSeek mode (backtests): real API signals recorded to this
    # file, replayed deterministically on later runs. Empty = live client.
    deepseek_cache_file: str = ""
    deepseek_max_api_calls: int = 0  # cap per run; 0 = unlimited

    # Pre-fetch history via REST on start (disable for backtesting)
    prefetch_bars: bool = True

    # Order emulation: needed LIVE (Binance futures lacks native OCO+OTO
    # brackets; the emulator triggers off streaming trade ticks). Disable in
    # BACKTESTS: with bars-only data the emulator never sees a tick, so
    # emulated SL/TP would never trigger - the simulated exchange handles
    # bracket contingencies natively instead.
    use_order_emulation: bool = True
    
    # OCO (One-Cancels-the-Other)
    enable_oco: bool = True
    oco_redis_host: str = "localhost"
    oco_redis_port: int = 6379
    oco_redis_db: int = 0
    oco_redis_password: Optional[str] = None
    oco_group_ttl_hours: int = 24
    
    # Trailing Stop Loss
    enable_trailing_stop: bool = True
    trailing_activation_pct: float = 0.01
    trailing_distance_pct: float = 0.005
    trailing_update_threshold_pct: float = 0.002
    
    # Partial Take Profit
    enable_partial_tp: bool = True
    partial_tp_levels: Tuple[Dict[str, float], ...] = (
        {"profit_pct": 0.02, "position_pct": 0.5},
        {"profit_pct": 0.04, "position_pct": 0.5},
    )
    
    # Telegram Notifications
    enable_telegram: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_notify_signals: bool = True
    telegram_notify_fills: bool = True
    telegram_notify_positions: bool = True
    telegram_notify_errors: bool = True

    # Execution
    position_adjustment_threshold: float = 0.001

    # Timing
    timer_interval_sec: int = 900


class DeepSeekAIStrategy(Strategy):
    """
    DeepSeek AI-powered trading strategy.

    Combines AI decision making, technical analysis, and sentiment data
    for intelligent cryptocurrency trading on Binance Futures.
    """

    def __init__(self, config: DeepSeekAIStrategyConfig):
        """
        Initialize DeepSeek AI strategy.

        Parameters
        ----------
        config : DeepSeekAIStrategyConfig
            Strategy configuration
        """
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)

        # Position sizing config
        self.equity = config.equity
        self.leverage = config.leverage
        self.base_usdt = config.base_usdt_amount
        self.position_config = {
            'high_confidence_multiplier': config.high_confidence_multiplier,
            'medium_confidence_multiplier': config.medium_confidence_multiplier,
            'low_confidence_multiplier': config.low_confidence_multiplier,
            'max_position_ratio': config.max_position_ratio,
            'trend_strength_multiplier': config.trend_strength_multiplier,
            'min_trade_amount': config.min_trade_amount,
            'adjustment_threshold': config.position_adjustment_threshold,
        }

        # Risk management
        self.min_confidence = config.min_confidence_to_trade
        self.allow_reversals = config.allow_reversals
        self.require_high_conf_reversal = config.require_high_confidence_for_reversal
        self.rsi_extreme_upper = config.rsi_extreme_threshold_upper
        self.rsi_extreme_lower = config.rsi_extreme_threshold_lower
        self.rsi_extreme_mult = config.rsi_extreme_multiplier
        
        # Stop Loss & Take Profit
        self.enable_auto_sl_tp = config.enable_auto_sl_tp
        self.sl_use_support_resistance = config.sl_use_support_resistance
        self.sl_buffer_pct = config.sl_buffer_pct
        self.tp_pct_config = {
            'HIGH': config.tp_high_confidence_pct,
            'MEDIUM': config.tp_medium_confidence_pct,
            'LOW': config.tp_low_confidence_pct,
        }

        # Sizing mode
        self.sizing_mode = config.sizing_mode
        self.risk_per_trade_pct = config.risk_per_trade_pct

        # ATR-based SL/TP parameters
        self.atr_sl_multiplier = config.atr_sl_multiplier
        self.min_sl_pct = config.min_sl_pct
        self.max_sl_pct = config.max_sl_pct
        self.min_risk_reward = config.min_risk_reward

        # Higher-timeframe (1h) trend filter state
        self.enable_htf_filter = config.enable_htf_filter
        self.htf_bar_type: Optional[BarType] = None
        if self.enable_htf_filter:
            self.htf_bar_type = BarType.from_str(
                f"{config.instrument_id}-1-HOUR-LAST-EXTERNAL"
            )
        self.htf_sma_fast = SimpleMovingAverage(config.htf_sma_fast)
        self.htf_sma_slow = SimpleMovingAverage(config.htf_sma_slow)
        self.htf_last_close: Optional[float] = None

        # Cooldown after losing trades
        self.loss_cooldown_bars = config.loss_cooldown_bars
        self.cooldown_until_bar: int = 0

        # Breakeven stop
        self.enable_breakeven_stop = config.enable_breakeven_stop
        self.breakeven_trigger_r = config.breakeven_trigger_r
        self.breakeven_buffer_pct = config.breakeven_buffer_pct

        # Daily loss circuit breaker
        self.daily_loss_limit_pct = config.daily_loss_limit_pct
        self._daily_breaker_active = False
        self._current_day = None
        self._day_start_equity: Optional[float] = None
        self._day_realized_pnl = 0.0

        # Loss-streak throttle
        self.loss_streak_threshold = config.loss_streak_threshold
        self.loss_streak_multiplier = config.loss_streak_multiplier
        self.loss_streak_reset_bars = config.loss_streak_reset_bars
        self.consecutive_losses = 0
        self._last_trade_close_bar = 0

        # Stagnant position exit
        self.max_position_age_bars = config.max_position_age_bars

        # Reversal confirmation
        self.reversal_confirmation_signals = config.reversal_confirmation_signals
        self._opposite_signal_streak = 0

        # Dead-market filter
        self.min_atr_pct_to_trade = config.min_atr_pct_to_trade

        # Chop filter (efficiency ratio)
        self.min_efficiency_ratio = config.min_efficiency_ratio

        # Stale-signal guard
        self.max_signal_staleness_pct = config.max_signal_staleness_pct

        # TP mode
        self.tp_mode = config.tp_mode
        self.tp_r_multiple = config.tp_r_multiple

        # HTF strict alignment
        self.htf_strict_alignment = config.htf_strict_alignment

        # Sizing source
        self.use_account_balance = config.use_account_balance

        # Analysis trigger mode / data source
        self.analyze_on_bar_close = config.analyze_on_bar_close
        self.analysis_source = config.analysis_source

        # Track SL/TP prices for the live position (used to re-sync
        # protection orders after adds/reduces/partial fills)
        self.position_protection: Dict[str, float] = {}
        self._pending_protection_refresh = False
        self._reversal_in_progress = False

        # Signal snapshot taken at ENTRY time - the feedback loop must label
        # trades with the signal that opened them, not whatever signal was
        # current when they closed (a reversal-closed trade would otherwise
        # be mislabeled with the reversing signal)
        self._entry_signal_snapshot: Optional[Dict[str, str]] = None

        # Closed-trade outcomes fed back to the AI (persisted across restarts)
        self.trade_history: List[Dict[str, Any]] = []
        self.max_trade_history = 10
        self._trade_history_path = os.path.join("logs", "trade_history.json")
        self._load_trade_history()
        
        # Store latest signal, technical, and price data for SL/TP calculation
        self.latest_signal_data: Optional[Dict[str, Any]] = None
        self.latest_technical_data: Optional[Dict[str, Any]] = None
        self.latest_price_data: Optional[Dict[str, Any]] = None

        # OCO (One-Cancels-the-Other) - Now handled by NautilusTrader's bracket orders
        # No need for manual OCO manager anymore
        self.enable_oco = config.enable_oco  # Keep for config compatibility
        self.oco_manager = None  # Deprecated: bracket orders handle OCO automatically
        
        # Trailing Stop Loss
        self.enable_trailing_stop = config.enable_trailing_stop
        self.trailing_activation_pct = config.trailing_activation_pct
        self.trailing_distance_pct = config.trailing_distance_pct
        self.trailing_update_threshold_pct = config.trailing_update_threshold_pct
        
        # Partial Take Profit
        self.enable_partial_tp = config.enable_partial_tp
        self.partial_tp_levels = list(config.partial_tp_levels)

        # Track trailing stop state for each position
        self.trailing_stop_state: Dict[str, Dict[str, Any]] = {}
        # Format: {
        #   "instrument_id": {
        #       "entry_price": float,
        #       "highest_price": float (for LONG) or "lowest_price": float (for SHORT),
        #       "current_sl_price": float,
        #       "sl_order_id": str,
        #       "activated": bool,
        #       "side": str (LONG/SHORT)
        #   }
        # }

        # Technical indicators manager
        self._sma_periods = config.sma_periods if config.sma_periods else [5, 20, 50]
        self.indicator_manager = self._build_indicator_manager()

        # Analyzer: rule-based (offline), cached DeepSeek (backtest
        # measurement), or live DeepSeek
        if config.use_rule_based_analyzer:
            from utils.rule_based_analyzer import RuleBasedAnalyzer
            self.deepseek = RuleBasedAnalyzer()
        elif config.deepseek_cache_file:
            from utils.cached_analyzer import CachedDeepSeekAnalyzer
            api_key = config.deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DeepSeek API key not provided")
            self.deepseek = CachedDeepSeekAnalyzer(
                api_key=api_key,
                cache_path=config.deepseek_cache_file,
                model=config.deepseek_model,
                temperature=config.deepseek_temperature,
                max_retries=config.deepseek_max_retries,
                max_api_calls=config.deepseek_max_api_calls,
                nautilus_logger=self.log,
            )
        else:
            api_key = config.deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DeepSeek API key not provided")

            self.deepseek = DeepSeekAnalyzer(
                api_key=api_key,
                model=config.deepseek_model,
                temperature=config.deepseek_temperature,
                max_retries=config.deepseek_max_retries,
                nautilus_logger=self.log,
            )
        
        # Telegram Bot
        self.telegram_bot = None
        self.enable_telegram = config.enable_telegram
        if self.enable_telegram:
            try:
                from utils.telegram_bot import TelegramBot
                
                bot_token = config.telegram_bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
                chat_id = config.telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
                
                if bot_token and chat_id:
                    self.telegram_bot = TelegramBot(
                        token=bot_token,
                        chat_id=chat_id,
                        logger=self.log,
                        enabled=True
                    )
                    # Store notification preferences
                    self.telegram_notify_signals = config.telegram_notify_signals
                    self.telegram_notify_fills = config.telegram_notify_fills
                    self.telegram_notify_positions = config.telegram_notify_positions
                    self.telegram_notify_errors = config.telegram_notify_errors
                    
                    self.log.info("✅ Telegram Bot initialized successfully")
                    
                    # Initialize command handler for remote control
                    try:
                        from utils.telegram_command_handler import TelegramCommandHandler
                        import threading
                        
                        # Create callback function for commands
                        def command_callback(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
                            """Callback function for Telegram commands."""
                            return self.handle_telegram_command(command, args)
                        
                        # Initialize command handler
                        allowed_chat_ids = [chat_id]  # Only allow the configured chat ID
                        self.telegram_command_handler = TelegramCommandHandler(
                            token=bot_token,
                            allowed_chat_ids=allowed_chat_ids,
                            strategy_callback=command_callback,
                            logger=self.log
                        )
                        
                        # Start command handler in background thread
                        def run_command_handler():
                            """Run command handler in background thread."""
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                # Start polling (this will run indefinitely via idle())
                                loop.run_until_complete(self.telegram_command_handler.start_polling())
                            except Exception as e:
                                self.log.error(f"❌ Command handler thread error: {e}")
                        
                        # Start background thread for command listening
                        command_thread = threading.Thread(
                            target=run_command_handler,
                            daemon=True,
                            name="TelegramCommandHandler"
                        )
                        command_thread.start()
                        self.log.info("✅ Telegram Command Handler started in background thread")
                        
                    except ImportError:
                        self.log.warning("⚠️ Telegram command handler not available")
                        self.telegram_command_handler = None
                    except Exception as e:
                        self.log.error(f"❌ Failed to initialize command handler: {e}")
                        self.telegram_command_handler = None
                    
                else:
                    self.log.warning("⚠️ Telegram enabled but token/chat_id not configured")
                    self.enable_telegram = False
            except ImportError:
                self.log.warning("⚠️ Telegram bot not available (python-telegram-bot not installed)")
                self.enable_telegram = False
            except Exception as e:
                self.log.error(f"❌ Failed to initialize Telegram Bot: {e}")
                self.enable_telegram = False
        
        # Strategy control state for remote commands
        self.is_trading_paused = False
        self.strategy_start_time = None

        # Sentiment data fetcher
        self.sentiment_enabled = config.sentiment_enabled
        if self.sentiment_enabled:
            # Use sentiment_timeframe from config, or derive from bar_type if not specified
            sentiment_tf = config.sentiment_timeframe
            if not sentiment_tf or sentiment_tf == "":
                # Extract timeframe from bar_type (e.g., "1-MINUTE" -> "1m")
                bar_str = str(self.bar_type)
                if "1-MINUTE" in bar_str:
                    sentiment_tf = "1m"
                elif "5-MINUTE" in bar_str:
                    sentiment_tf = "5m"
                elif "15-MINUTE" in bar_str:
                    sentiment_tf = "15m"
                elif "1-HOUR" in bar_str:
                    sentiment_tf = "1h"
                else:
                    sentiment_tf = "15m"  # Default fallback
            
            self.sentiment_fetcher = SentimentDataFetcher(
                lookback_hours=config.sentiment_lookback_hours,
                timeframe=sentiment_tf,
                timeout=5.0,
                logger=self.log,
            )
            self.log.info(f"Sentiment fetcher initialized with timeframe: {sentiment_tf}")
        else:
            self.sentiment_fetcher = None

        # State tracking
        self.instrument: Optional[Instrument] = None
        self.last_signal: Optional[Dict[str, Any]] = None
        self.bars_received = 0

        self.log.info(f"DeepSeek AI Strategy initialized for {self.instrument_id}")

    def on_start(self):
        """Actions to be performed on strategy start."""
        self.log.info("Starting DeepSeek AI Strategy...")

        # Load instrument
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        self.log.info(f"Loaded instrument: {self.instrument.id}")

        # Pre-fetch historical bars before subscribing to live data
        # (disabled for backtesting, where history arrives via the engine)
        if self.config.prefetch_bars:
            self._prefetch_historical_bars(limit=200)

        if self.analysis_source == "rest":
            # REST-polling mode: DO NOT subscribe to the (unreliable) kline
            # WebSocket. Drive analysis from a timer that re-fetches fresh
            # bars via REST each cycle. Backtests never use this path
            # (backtest configs set analysis_source implicitly via the engine).
            interval = max(60, self.config.timer_interval_sec)
            self.clock.set_timer(
                name="analysis_timer",
                interval=timedelta(seconds=interval),
                callback=self.on_timer,
            )
            self.log.info(
                f"Analysis mode: REST polling every {interval}s "
                f"(WebSocket bar push bypassed for reliability)"
            )
        else:
            # WebSocket mode (event-driven or timer)
            self.subscribe_bars(self.bar_type)
            self.log.info(f"Subscribed to {self.bar_type}")
            if self.enable_htf_filter and self.htf_bar_type is not None:
                if self.config.prefetch_bars:
                    self._prefetch_htf_bars(limit=200)
                self.subscribe_bars(self.htf_bar_type)
                self.log.info(f"Subscribed to HTF {self.htf_bar_type}")

            if self.analyze_on_bar_close:
                self.log.info("Analysis mode: on bar close (event-driven)")
            else:
                self.clock.set_timer(
                    name="analysis_timer",
                    interval=timedelta(seconds=self.config.timer_interval_sec),
                    callback=self.on_timer,
                )
                self.log.info(
                    f"Analysis mode: timer every {self.config.timer_interval_sec}s"
                )

        self.log.info("Strategy started successfully")

        # Record start time for uptime tracking
        from datetime import datetime
        self.strategy_start_time = datetime.utcnow()

        # Send Telegram startup notification
        if self.telegram_bot and self.enable_telegram:
            try:
                startup_msg = self.telegram_bot.format_startup_message(
                    instrument_id=str(self.instrument_id),
                    config={
                        'enable_auto_sl_tp': self.enable_auto_sl_tp,
                        'enable_oco': self.enable_oco,
                        'enable_trailing_stop': self.enable_trailing_stop,
                        'enable_partial_tp': hasattr(self, 'enable_partial_tp') and getattr(self, 'enable_partial_tp', False),
                    }
                )
                self.telegram_bot.send_message_sync(startup_msg)

                # Send command help message
                help_msg = self.telegram_bot.format_help_response()
                self.telegram_bot.send_message_sync(help_msg)

            except Exception as e:
                self.log.warning(f"Failed to send Telegram startup notification: {e}")

    def on_stop(self):
        """Actions to be performed on strategy stop."""
        self.log.info("Stopping DeepSeek AI Strategy...")

        # Cancel any pending orders
        self.cancel_all_orders(self.instrument_id)

        # Flatten: SL/TP are emulated (bot-local), so a position held through
        # shutdown would sit on the exchange with NO protective orders.
        if self.config.close_positions_on_stop:
            positions = self.cache.positions_open(instrument_id=self.instrument_id)
            if positions:
                self.log.warning(
                    "⚠️ Closing open position on stop (emulated SL/TP cannot "
                    "protect it while the bot is offline)"
                )
                self.close_all_positions(self.instrument_id)

        # Unsubscribe from data (only if we subscribed - REST mode does not)
        if self.analysis_source != "rest":
            self.unsubscribe_bars(self.bar_type)
            if self.enable_htf_filter and self.htf_bar_type is not None:
                self.unsubscribe_bars(self.htf_bar_type)

        self.log.info("Strategy stopped")

    def _build_indicator_manager(self) -> TechnicalIndicatorManager:
        """Create a fresh indicator manager with the configured parameters.

        Used both at init and to rebuild state from a REST refresh (so the
        REST-polling path recomputes indicators cleanly each cycle).
        """
        return TechnicalIndicatorManager(
            sma_periods=self._sma_periods,
            ema_periods=[self.config.macd_fast, self.config.macd_slow],
            rsi_period=self.config.rsi_period,
            macd_fast=self.config.macd_fast,
            macd_slow=self.config.macd_slow,
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            atr_period=self.config.atr_period,
        )

    def _bar_interval(self) -> str:
        """Binance interval string for the primary bar type.

        Longest-token-first: '5-MINUTE' is a substring of '15-MINUTE', so a
        naive check would mis-map 15m to 5m.
        """
        bar_type_str = str(self.bar_type)
        for token, iv in [
            ('15-MINUTE', '15m'), ('30-MINUTE', '30m'), ('5-MINUTE', '5m'),
            ('3-MINUTE', '3m'), ('1-MINUTE', '1m'), ('12-HOUR', '12h'),
            ('4-HOUR', '4h'), ('2-HOUR', '2h'), ('1-HOUR', '1h'), ('1-DAY', '1d'),
        ]:
            if token in bar_type_str:
                return iv
        return '15m'

    def _fetch_klines(self, interval: str, limit: int = 200) -> list:
        """Fetch raw klines from Binance Futures REST. Returns [] on failure."""
        import requests
        symbol = str(self.instrument_id).split('-')[0]
        url = "https://fapi.binance.com/fapi/v1/klines"
        response = requests.get(
            url,
            params={'symbol': symbol, 'interval': interval, 'limit': min(limit, 1500)},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def _kline_to_bar(self, kline: list) -> Bar:
        """Convert a raw Binance kline to a Nautilus Bar (primary bar type)."""
        from nautilus_trader.core.datetime import millis_to_nanos
        return Bar(
            bar_type=self.bar_type,
            open=self.instrument.make_price(float(kline[1])),
            high=self.instrument.make_price(float(kline[2])),
            low=self.instrument.make_price(float(kline[3])),
            close=self.instrument.make_price(float(kline[4])),
            volume=self.instrument.make_qty(float(kline[5])),
            ts_event=millis_to_nanos(kline[0]),
            ts_init=millis_to_nanos(kline[0]),
        )

    def _prefetch_historical_bars(self, limit: int = 200):
        """Warm indicators from REST klines at startup (feeds the manager)."""
        try:
            interval = self._bar_interval()
            self.log.info(
                f"📡 Pre-fetching {limit} historical bars from Binance "
                f"(interval={interval})..."
            )
            klines = self._fetch_klines(interval, limit)
            if not klines:
                self.log.warning("⚠️ No bars received from Binance API")
                return
            self.log.info(f"📊 Received {len(klines)} bars from Binance")
            bars_fed = 0
            for kline in klines:
                try:
                    self.indicator_manager.update(self._kline_to_bar(kline))
                    bars_fed += 1
                except Exception as e:
                    self.log.warning(f"Failed to convert kline to bar: {e}")
            self.log.info(
                f"✅ Pre-fetched {bars_fed} bars! "
                f"Indicators ready: {self.indicator_manager.is_initialized()}"
            )
        except Exception as e:
            self.log.error(f"❌ Failed to pre-fetch bars from Binance: {e}")

    def _rebuild_from_rest(self) -> bool:
        """
        Rebuild ALL indicator state from a fresh REST kline pull.

        Excludes the in-progress (still-open) candle so indicators reflect
        only closed bars. This is the reliable analysis path - it does not
        depend on the flaky kline WebSocket push. Returns True on success.
        """
        try:
            klines = self._fetch_klines(self._bar_interval(), 200)
            if not klines or len(klines) < 2:
                self.log.warning("⚠️ REST returned no/insufficient klines this cycle")
                return False

            # Drop the last (in-progress) candle - only trade on closed bars
            closed = klines[:-1]

            mgr = self._build_indicator_manager()
            for kline in closed:
                mgr.update(self._kline_to_bar(kline))
            self.indicator_manager = mgr

            # Rebuild HTF (1h) trend from REST too
            if self.enable_htf_filter:
                htf_klines = self._fetch_klines('1h', 200)
                if htf_klines and len(htf_klines) >= 2:
                    self.htf_sma_fast = SimpleMovingAverage(self.config.htf_sma_fast)
                    self.htf_sma_slow = SimpleMovingAverage(self.config.htf_sma_slow)
                    for k in htf_klines[:-1]:
                        c = float(k[4])
                        self.htf_sma_fast.update_raw(c)
                        self.htf_sma_slow.update_raw(c)
                        self.htf_last_close = c

            # Advance the bar counter (cooldown/age/streak logic is in units
            # of analysis cycles ≈ closed bars)
            self.bars_received += 1
            return True
        except Exception as e:
            self.log.error(f"❌ REST rebuild failed this cycle: {e}")
            return False

    def on_bar(self, bar: Bar):
        """
        Handle bar updates.

        Parameters
        ----------
        bar : Bar
            The bar received
        """
        # Route higher-timeframe bars to the HTF trend filter
        if self.htf_bar_type is not None and bar.bar_type == self.htf_bar_type:
            self._update_htf(bar)
            return

        self.bars_received += 1

        # Update technical indicators
        self.indicator_manager.update(bar)

        # Log bar data
        if self.bars_received % 10 == 0:
            self.log.info(
                f"Bar #{self.bars_received}: "
                f"O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close} V:{bar.volume}"
            )

        # Event-driven analysis: every closed primary bar is a decision point
        if self.analyze_on_bar_close:
            self._run_analysis()

    def _update_htf(self, bar: Bar):
        """Update higher-timeframe SMAs from 1h bars."""
        close = float(bar.close)
        self.htf_sma_fast.update_raw(close)
        self.htf_sma_slow.update_raw(close)
        self.htf_last_close = close

    def _get_htf_trend(self) -> str:
        """
        Classify the higher-timeframe (1h) trend.

        Returns 'UPTREND', 'DOWNTREND', or 'NEUTRAL' (also when not ready).
        """
        if (
            self.htf_last_close is None
            or not self.htf_sma_fast.initialized
            or not self.htf_sma_slow.initialized
        ):
            return "NEUTRAL"

        fast = self.htf_sma_fast.value
        slow = self.htf_sma_slow.value

        if self.htf_last_close > fast > slow:
            return "UPTREND"
        if self.htf_last_close < fast < slow:
            return "DOWNTREND"
        return "NEUTRAL"

    def _prefetch_htf_bars(self, limit: int = 200):
        """Pre-fetch 1h bars so the HTF filter is warm at startup."""
        try:
            import requests
            symbol = str(self.instrument_id).split('-')[0]
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': '1h', 'limit': min(limit, 1500)}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            for kline in response.json():
                close = float(kline[4])
                self.htf_sma_fast.update_raw(close)
                self.htf_sma_slow.update_raw(close)
                self.htf_last_close = close
            self.log.info(
                f"✅ HTF pre-fetch complete, trend: {self._get_htf_trend()}"
            )
        except Exception as e:
            self.log.warning(f"HTF pre-fetch failed (filter warms up live): {e}")

    def on_timer(self, event):
        """
        Timer callback.

        REST mode: re-fetch fresh bars via REST (reliable) and rebuild
        indicators before analyzing. WebSocket-timer mode: analyze on the
        bars delivered via on_bar.
        """
        if self.analysis_source == "rest":
            if not self._rebuild_from_rest():
                self.log.warning("Skipping analysis cycle - REST refresh failed")
                return
        self._run_analysis()

    def _run_analysis(self):
        """
        Core analysis and trading logic.

        Triggered on each closed primary bar (default) or by timer (fallback).
        """
        self.log.info("=" * 60)
        self.log.info("Running periodic analysis...")

        # Loss-streak decay: a streak that has produced no trades for a long
        # stretch is stale context - reset it so the throttle cannot deadlock
        # accounts whose throttled size falls below the exchange minimum.
        if (
            self.consecutive_losses > 0
            and self.loss_streak_reset_bars > 0
            and self.bars_received - self._last_trade_close_bar >= self.loss_streak_reset_bars
        ):
            self.log.info(
                f"🔄 Loss streak ({self.consecutive_losses}) expired after "
                f"{self.loss_streak_reset_bars} bars without a trade - reset"
            )
            self.consecutive_losses = 0

        # UTC day rollover: reset the daily loss breaker and baseline equity
        today = self.clock.utc_now().date()
        if today != self._current_day:
            self._current_day = today
            self._day_start_equity = self._get_effective_equity()
            self._day_realized_pnl = 0.0
            if self._daily_breaker_active:
                self.log.info("🟢 New UTC day - daily loss breaker reset, trading resumed")
            self._daily_breaker_active = False

        # Check if indicators are ready
        if not self.indicator_manager.is_initialized():
            self.log.warning("Indicators not yet initialized, skipping analysis")
            return

        # Get current market data
        current_bar = self.indicator_manager.recent_bars[-1] if self.indicator_manager.recent_bars else None
        if not current_bar:
            self.log.warning("No bars available for analysis")
            return

        current_price = float(current_bar.close)

        # Get technical data
        try:
            technical_data = self.indicator_manager.get_technical_data(current_price)
            self.log.debug(f"Technical data retrieved: {len(technical_data)} indicators")
        except Exception as e:
            self.log.error(f"Failed to get technical data: {e}")
            return

        # Get K-line data
        kline_data = self.indicator_manager.get_kline_data(count=10)
        self.log.debug(f"Retrieved {len(kline_data)} K-lines for analysis")

        # Get sentiment data
        sentiment_data = None
        if self.sentiment_enabled and self.sentiment_fetcher:
            try:
                sentiment_data = self.sentiment_fetcher.fetch()
                if sentiment_data:
                    self.log.info(self.sentiment_fetcher.format_for_display(sentiment_data))
            except Exception as e:
                self.log.warning(f"Failed to fetch sentiment data: {e}")

        # Higher-timeframe trend context (also used as an entry gate)
        htf_trend = self._get_htf_trend()
        technical_data['htf_trend'] = htf_trend

        # Funding rate context (perp carry cost - free signal for the AI).
        # Skipped in rule-based mode (backtests stay fully offline).
        funding_data = None
        if not self.config.use_rule_based_analyzer:
            try:
                from utils.funding_client import FundingRateFetcher
                if not hasattr(self, '_funding_fetcher'):
                    self._funding_fetcher = FundingRateFetcher(
                        symbol=str(self.instrument_id).split('-')[0]
                    )
                funding_data = self._funding_fetcher.fetch()
                if funding_data:
                    self.log.info(
                        f"Funding rate: {funding_data['funding_rate']*100:+.4f}% "
                        f"(next: {funding_data.get('next_funding_time', 'N/A')})"
                    )
            except Exception as e:
                self.log.debug(f"Funding rate fetch failed: {e}")

        # Build price data for AI
        price_data = {
            'price': current_price,
            'timestamp': self.clock.utc_now().isoformat(),
            'high': float(current_bar.high),
            'low': float(current_bar.low),
            'volume': float(current_bar.volume),
            'price_change': self._calculate_price_change(),
            'kline_data': kline_data,
        }

        # Get current position
        current_position = self._get_current_position_data()

        # Log current state
        self.log.info(f"Current Price: ${current_price:,.2f}")
        self.log.info(f"Overall Trend: {technical_data.get('overall_trend', 'N/A')}")
        self.log.info(f"RSI: {technical_data.get('rsi', 0):.2f}")
        if current_position:
            self.log.info(
                f"Current Position: {current_position['side']} "
                f"{current_position['quantity']} @ ${current_position['avg_px']:.2f}"
            )

        # Analyze with DeepSeek AI
        try:
            import time as _time
            self.log.info("Calling DeepSeek AI for analysis...")
            _t0 = _time.monotonic()
            signal_data = self.deepseek.analyze(
                price_data=price_data,
                technical_data=technical_data,
                sentiment_data=sentiment_data,
                current_position=current_position,
                trade_history=self.trade_history,
                funding_data=funding_data,
            )
            _elapsed = _time.monotonic() - _t0
            self.log.info(
                f"🤖 Signal: {signal_data['signal']} | "
                f"Confidence: {signal_data['confidence']} | "
                f"API time: {_elapsed:.1f}s | "
                f"Reason: {signal_data['reason']}"
            )
            
            # Send Telegram signal notification (only for actionable signals)
            if self.telegram_bot and self.enable_telegram and self.telegram_notify_signals:
                if signal_data['signal'] in ['BUY', 'SELL']:
                    try:
                        signal_notification = self.telegram_bot.format_trade_signal({
                            'signal': signal_data['signal'],
                            'confidence': signal_data['confidence'],
                            'price': price_data['price'],
                            'timestamp': price_data['timestamp'],
                            'rsi': technical_data.get('rsi', 0),
                            'macd': technical_data.get('macd', 0),
                            'support': technical_data.get('support', 0),
                            'resistance': technical_data.get('resistance', 0),
                            'reasoning': signal_data['reason'],
                        })
                        self.telegram_bot.send_message_sync(signal_notification)
                    except Exception as e:
                        self.log.warning(f"Failed to send Telegram signal notification: {e}")
                        
        except Exception as e:
            self.log.error(f"DeepSeek AI analysis failed: {e}", exc_info=True)
            
            # Send error notification
            if self.telegram_bot and self.enable_telegram and self.telegram_notify_errors:
                try:
                    error_msg = self.telegram_bot.format_error_alert({
                        'level': 'ERROR',
                        'message': f"AI Analysis Failed: {str(e)[:100]}",
                        'context': 'on_timer'
                    })
                    self.telegram_bot.send_message_sync(error_msg)
                except:
                    pass
            return

        # Store signal
        self.last_signal = signal_data

        # Execute trade
        self._execute_trade(signal_data, price_data, technical_data, current_position)
        
        # Orphan-order safety net. (Was gated on the deprecated oco_manager,
        # which is always None - meaning this never ran. Primary cleanup is
        # event-driven in on_position_closed; this catches edge cases like
        # fills that arrive during a disconnect.)
        self._cleanup_oco_orphans()
        
        # Trailing stop maintenance: check and update trailing stops
        if self.enable_trailing_stop:
            self._update_trailing_stops(price_data['price'])

        # Stagnant position exit: cut positions that go nowhere
        if self.max_position_age_bars > 0:
            self._check_stale_position()

    def _check_stale_position(self):
        """
        Close positions that have neither hit SL/TP nor reached the trailing
        activation threshold within max_position_age_bars.

        Rationale: a position that does nothing for hours ties up margin,
        bleeds funding, and its original signal has long expired. Winners
        (trailing activated) are exempt and ride the trend.
        """
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        if not positions:
            return

        state = self.trailing_stop_state.get(str(self.instrument_id))
        if not state:
            return

        if state.get("activated"):
            return  # winner - let it ride

        entry_bar = state.get("entry_bar")
        if entry_bar is None:
            return

        age = self.bars_received - entry_bar
        if age < self.max_position_age_bars:
            return

        position = positions[0]
        self.log.info(
            f"⏳ Stagnant position: {age} bars without reaching trailing "
            f"activation - closing {position.side.name} "
            f"{float(position.quantity):.3f} BTC to free capital"
        )
        self._submit_order(
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            quantity=float(position.quantity),
            reduce_only=True,
        )

    def _calculate_price_change(self) -> float:
        """Calculate price change percentage."""
        bars = self.indicator_manager.recent_bars
        if len(bars) < 2:
            return 0.0

        current = float(bars[-1].close)
        previous = float(bars[-2].close)

        return ((current - previous) / previous) * 100

    def _get_current_position_data(self) -> Optional[Dict[str, Any]]:
        """Get current position information."""
        # Get open positions for this instrument
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        
        if not positions:
            return None
        
        # Get the first open position (should only be one for netting OMS)
        position = positions[0]
        
        if position and position.is_open:
            # Get current price for PnL calculation
            # Use last bar close price as it's more reliable than cache.price()
            # cache.price() requires tick data which may not be available
            bars = self.indicator_manager.recent_bars
            if bars:
                current_price = bars[-1].close
            else:
                # Fallback: try cache.price() if bars not available
                try:
                    current_price = self.cache.price(self.instrument_id, PriceType.LAST)
                except (TypeError, AttributeError):
                    current_price = None
            
            return {
                'side': 'long' if position.side == PositionSide.LONG else 'short',
                'quantity': float(position.quantity),
                'avg_px': float(position.avg_px_open),
                'unrealized_pnl': float(position.unrealized_pnl(current_price)) if current_price else 0.0,
            }

        return None

    def _execute_trade(
        self,
        signal_data: Dict[str, Any],
        price_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        current_position: Optional[Dict[str, Any]],
    ):
        """
        Execute trading logic based on signal.

        Parameters
        ----------
        signal_data : Dict
            AI-generated signal
        price_data : Dict
            Current price data
        technical_data : Dict
            Technical indicators
        current_position : Dict or None
            Current position info
        """
        # Check if trading is paused
        if self.is_trading_paused:
            self.log.info("⏸️ Trading is paused - skipping signal execution")
            return
        
        # Store signal and technical data for SL/TP calculation
        self.latest_signal_data = signal_data
        self.latest_technical_data = technical_data
        self.latest_price_data = price_data
        
        signal = signal_data['signal']
        confidence = signal_data['confidence']

        # Check minimum confidence
        confidence_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        min_conf_level = confidence_levels.get(self.min_confidence, 1)
        signal_conf_level = confidence_levels.get(confidence, 1)

        if signal_conf_level < min_conf_level:
            self.log.warning(
                f"⚠️ Signal confidence {confidence} below minimum {self.min_confidence}, skipping trade"
            )
            return

        # Handle HOLD signal
        if signal == 'HOLD':
            self.log.info("📊 Signal: HOLD - No action taken")
            return

        # Daily loss circuit breaker: no new risk after the daily limit.
        # An open position may still be CLOSED on an opposite signal
        # (risk reduction), but never adjusted or reversed.
        if self._daily_breaker_active:
            if current_position is None:
                self.log.warning("🛑 Daily loss breaker active - no new entries today")
                return
            target = 'long' if signal == 'BUY' else 'short'
            if target != current_position['side']:
                self.log.warning(
                    "🛑 Daily breaker active - closing position on opposite "
                    "signal (no re-entry)"
                )
                self._submit_order(
                    side=OrderSide.SELL if current_position['side'] == 'long' else OrderSide.BUY,
                    quantity=current_position['quantity'],
                    reduce_only=True,
                )
            else:
                self.log.info("🛑 Daily breaker active - holding position, no adjustments")
            return

        # Dead-market filter: in very low volatility, fees exceed any edge
        if (
            current_position is None
            and self.min_atr_pct_to_trade > 0
        ):
            atr = technical_data.get('atr', 0.0)
            price = price_data['price']
            if atr > 0 and price > 0 and (atr / price) < self.min_atr_pct_to_trade:
                self.log.info(
                    f"💤 Dead market: ATR {atr/price:.3%} < "
                    f"{self.min_atr_pct_to_trade:.3%} threshold - skipping entry"
                )
                return

        # Chop filter: volatile-but-directionless markets whipsaw trend
        # entries (ATR passes, efficiency doesn't - the live loss pattern)
        if current_position is None and self.min_efficiency_ratio > 0:
            er = technical_data.get('efficiency_ratio', 0.0)
            if er < self.min_efficiency_ratio:
                self.log.info(
                    f"🌀 Choppy market: efficiency ratio {er:.2f} < "
                    f"{self.min_efficiency_ratio:.2f} - skipping entry"
                )
                return

        # Stale-signal guard (live REST mode): the analysis price is from
        # BEFORE the 30-70s DeepSeek call. If price has since run in the
        # signal direction beyond the threshold, entering now chases a worse
        # price with SL/TP geometry computed for the old one - skip.
        if (
            current_position is None
            and self.max_signal_staleness_pct > 0
            and self.analysis_source == "rest"
        ):
            try:
                latest = self._fetch_klines(self._bar_interval(), limit=1)
                if latest:
                    now_price = float(latest[-1][4])
                    analysis_price = price_data['price']
                    drift = (now_price - analysis_price) / analysis_price
                    chasing = (signal == 'BUY' and drift > self.max_signal_staleness_pct) or (
                        signal == 'SELL' and drift < -self.max_signal_staleness_pct
                    )
                    if chasing:
                        self.log.info(
                            f"⏱️ Stale signal: price moved {drift:+.2%} since analysis "
                            f"(${analysis_price:,.2f} → ${now_price:,.2f}) - skipping "
                            f"{signal} entry rather than chasing"
                        )
                        return
                    # Use the fresher price for sizing/SL/TP geometry
                    price_data['price'] = now_price
                    self.latest_price_data = price_data
            except Exception as e:
                self.log.debug(f"Staleness check failed (proceeding): {e}")

        # Loss cooldown: block NEW entries for a few bars after a stop-out
        # (managing an existing position is still allowed)
        if (
            current_position is None
            and self.bars_received < self.cooldown_until_bar
        ):
            self.log.info(
                f"🧊 In loss cooldown ({self.cooldown_until_bar - self.bars_received} "
                f"bars remaining) - skipping new entry"
            )
            return

        # Higher-timeframe filter for NEW positions.
        # strict: entries must be WITH the 1h trend (NEUTRAL blocks too)
        # default: only counter-trend entries are blocked
        if self.enable_htf_filter and current_position is None:
            htf_trend = technical_data.get('htf_trend', 'NEUTRAL')
            if self.htf_strict_alignment:
                required = 'UPTREND' if signal == 'BUY' else 'DOWNTREND'
                if htf_trend != required:
                    self.log.info(
                        f"🚧 HTF strict: {signal} requires 1h {required}, "
                        f"got {htf_trend} - skipping entry"
                    )
                    return
            elif (signal == 'BUY' and htf_trend == 'DOWNTREND') or (
                signal == 'SELL' and htf_trend == 'UPTREND'
            ):
                self.log.info(
                    f"🚧 HTF filter: {signal} signal against 1h {htf_trend}, "
                    f"skipping counter-trend entry"
                )
                return

        # Calculate target position size
        target_quantity = self._calculate_position_size(
            signal_data, price_data, technical_data, current_position
        )

        if target_quantity == 0:
            self.log.warning("⚠️ Calculated position size is 0, skipping trade")
            return

        # Determine order side
        target_side = 'long' if signal == 'BUY' else 'short'

        # Execute position management logic
        if current_position:
            self._manage_existing_position(
                current_position, target_side, target_quantity, confidence
            )
        else:
            self._open_new_position(target_side, target_quantity)

    def _calculate_position_size(
        self,
        signal_data: Dict[str, Any],
        price_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        current_position: Optional[Dict[str, Any]],
    ) -> float:
        """
        Calculate intelligent position size.

        Returns BTC quantity based on confidence, trend, and RSI.
        """
        # Base USDT amount
        base_usdt = self.base_usdt

        # Confidence multiplier
        conf_mult = self.position_config.get(
            f"{signal_data['confidence'].lower()}_confidence_multiplier",
            1.0
        )

        # Trend multiplier
        trend = technical_data.get('overall_trend', '震荡整理')
        trend_mult = (
            self.position_config['trend_strength_multiplier']
            if trend in ['强势上涨', '强势下跌']
            else 1.0
        )

        # RSI multiplier (reduce size in extreme RSI)
        rsi = technical_data.get('rsi', 50)
        rsi_mult = (
            self.rsi_extreme_mult
            if rsi > self.rsi_extreme_upper or rsi < self.rsi_extreme_lower
            else 1.0
        )

        # Loss-streak throttle: shrink size while on a losing streak
        # (anti-martingale - risk less when the strategy is out of sync)
        streak_mult = 1.0
        if (
            self.loss_streak_threshold > 0
            and self.consecutive_losses >= self.loss_streak_threshold
        ):
            streak_mult = self.loss_streak_multiplier

        equity = self._get_effective_equity()
        current_price = price_data['price']
        combined_mult = conf_mult * trend_mult * rsi_mult * streak_mult

        # Target notional by sizing mode.
        if self.sizing_mode == "risk":
            # Size so a stop-out loses (risk_per_trade_pct × combined_mult) of
            # equity. Notional = risk_usd / stop_distance_fraction, using the
            # SAME stop the bracket will place -> risk is consistent whether
            # the ATR stop is tight or wide (this is the whole point).
            is_buy = signal_data.get('signal') == 'BUY'
            atr = technical_data.get('atr', 0.0)
            sl_distance, _ = self._compute_sl_distance(
                is_buy, current_price, atr,
                technical_data.get('support', 0.0),
                technical_data.get('resistance', 0.0),
            )
            stop_frac = sl_distance / current_price if current_price > 0 else 0.0
            if stop_frac <= 0:
                self.log.warning("⚠️ Stop distance is 0, cannot risk-size; skipping")
                return 0.0
            risk_usd = equity * self.risk_per_trade_pct * combined_mult
            suggested_usdt = risk_usd / stop_frac
            sizing_desc = (
                f"Risk {self.risk_per_trade_pct:.1%}×{combined_mult:.2f}="
                f"${risk_usd:.2f} / stop {stop_frac:.2%}"
            )
        else:
            # Legacy fixed-notional: base × multipliers
            suggested_usdt = base_usdt * combined_mult
            sizing_desc = (
                f"Base:{base_usdt} × Conf:{conf_mult} × Trend:{trend_mult} "
                f"× RSI:{rsi_mult}"
                f"{f' × Streak:{streak_mult}' if streak_mult != 1.0 else ''}"
            )

        # Risk cap: max_position_ratio applies to MARGIN (equity at risk).
        # With leverage, the equivalent notional cap is margin_cap × leverage.
        # (The old code applied the ratio to notional directly, which made the
        # cap smaller than the exchange minimum and forced every trade to a
        # constant $100 — killing confidence/trend/RSI-based sizing entirely.)
        max_margin = equity * self.position_config['max_position_ratio']
        max_notional = max_margin * self.leverage
        final_usdt = min(suggested_usdt, max_notional)

        # Exchange minimum notional: NEVER round up past the risk cap.
        # If we can't meet the exchange minimum within our risk limits,
        # the correct action is to skip the trade, not to take more risk.
        MIN_NOTIONAL_USDT = 100.0
        if final_usdt < MIN_NOTIONAL_USDT:
            self.log.warning(
                f"⚠️ Sized notional ${final_usdt:.2f} below exchange minimum "
                f"${MIN_NOTIONAL_USDT:.0f} (risk cap: ${max_notional:.2f}). "
                f"Skipping trade instead of exceeding risk limits."
            )
            return 0.0

        # Convert to BTC quantity. Round to NEAREST precision step: at high
        # BTC prices one 0.001 step is ~$100 of notional, so flooring would
        # collapse confidence-differentiated sizes back to the same quantity.
        # The risk cap is still enforced after rounding (floor if exceeded).
        import math
        btc_quantity = round(final_usdt / current_price, 3)
        if btc_quantity * current_price > max_notional:
            btc_quantity = math.floor((max_notional / current_price) * 1000) / 1000

        # Rounding down may drop us below the exchange minimum; only round
        # back up if doing so still respects the risk cap.
        if btc_quantity * current_price < MIN_NOTIONAL_USDT:
            bumped = math.ceil((MIN_NOTIONAL_USDT / current_price) * 1000) / 1000
            if bumped * current_price <= max_notional:
                btc_quantity = bumped
            else:
                self.log.warning(
                    "⚠️ Cannot meet exchange minimum notional within risk cap "
                    "after rounding, skipping trade"
                )
                return 0.0

        # Apply minimum trade amount check (skip, never bump)
        if btc_quantity < self.position_config['min_trade_amount']:
            self.log.warning(
                f"⚠️ Quantity {btc_quantity:.3f} below minimum trade amount, skipping"
            )
            return 0.0

        actual_notional = btc_quantity * current_price
        self.log.info(
            f"📊 Position Sizing [{self.sizing_mode}]: {sizing_desc} "
            f"= ${final_usdt:.2f} (cap ${max_notional:.2f}, equity ${equity:.2f}) "
            f"= {btc_quantity:.3f} BTC "
            f"(notional: ${actual_notional:.2f}, "
            f"margin: ${actual_notional / self.leverage:.2f})"
        )

        return btc_quantity

    def _get_effective_equity(self) -> float:
        """
        Get equity for sizing: live account balance when available,
        falling back to the static configured equity.

        Using the live balance means position sizes shrink in drawdowns
        and grow with profits, instead of sizing off a stale .env number.
        """
        if not self.use_account_balance:
            return self.equity

        try:
            account = self.portfolio.account(self.instrument_id.venue)
            if account is not None:
                from nautilus_trader.model.currencies import USDT
                balance = account.balance_total(USDT)
                if balance is not None:
                    live_equity = float(balance.as_double())
                    if live_equity > 0:
                        return live_equity
        except Exception as e:
            self.log.debug(f"Could not read account balance, using static equity: {e}")

        return self.equity

    def _manage_existing_position(
        self,
        current_position: Dict[str, Any],
        target_side: str,
        target_quantity: float,
        confidence: str,
    ):
        """Manage existing position (add, reduce, or reverse)."""
        current_side = current_position['side']
        current_qty = current_position['quantity']

        # Same direction - adjust position
        if target_side == current_side:
            # A same-direction signal breaks any opposite-signal streak
            self._opposite_signal_streak = 0

            size_diff = target_quantity - current_qty
            threshold = self.position_config['adjustment_threshold']

            if abs(size_diff) < threshold:
                self.log.info(
                    f"✅ Position size appropriate ({current_qty:.3f} BTC), no adjustment needed"
                )
                return

            if size_diff > 0:
                # Add to position - protection orders re-synced to the new
                # quantity in on_position_changed (via refresh flag)
                self._pending_protection_refresh = True
                self._submit_order(
                    side=OrderSide.BUY if target_side == 'long' else OrderSide.SELL,
                    quantity=abs(size_diff),
                    reduce_only=False,
                )
                self.log.info(
                    f"📈 Adding to {target_side} position: {abs(size_diff):.3f} BTC "
                    f"({current_qty:.3f} → {target_quantity:.3f})"
                )
            else:
                # Reduce position - protection orders re-synced likewise
                self._pending_protection_refresh = True
                self._submit_order(
                    side=OrderSide.SELL if target_side == 'long' else OrderSide.BUY,
                    quantity=abs(size_diff),
                    reduce_only=True,
                )
                self.log.info(
                    f"📉 Reducing {target_side} position: {abs(size_diff):.3f} BTC "
                    f"({current_qty:.3f} → {target_quantity:.3f})"
                )

        # Opposite direction - reverse position
        elif self.allow_reversals:
            # Check if high confidence required for reversal
            if self.require_high_conf_reversal and confidence != 'HIGH':
                self.log.warning(
                    f"🔒 Reversal requires HIGH confidence, got {confidence}. "
                    f"Keeping {current_side} position."
                )
                return

            # Reversal confirmation: require N consecutive opposite signals
            # before flipping - a single counter-signal on one bar is often
            # noise, and each reversal costs double fees plus slippage.
            if self.reversal_confirmation_signals > 1:
                self._opposite_signal_streak += 1
                if self._opposite_signal_streak < self.reversal_confirmation_signals:
                    self.log.info(
                        f"⏳ Reversal signal {self._opposite_signal_streak}/"
                        f"{self.reversal_confirmation_signals} - awaiting confirmation "
                        f"before flipping {current_side} → {target_side}"
                    )
                    return
            self._opposite_signal_streak = 0

            self.log.info(
                f"🔄 Reversal signal: closing {current_side} position "
                f"(re-enter {target_side} next cycle if the signal holds)"
            )

            # Cancel the old position's SL/TP orders FIRST.
            self.cancel_all_orders(self.instrument_id)
            self.trailing_stop_state.pop(str(self.instrument_id), None)
            self.position_protection = {}

            # CLOSE-AND-WAIT: only close here; do NOT open the opposite in the
            # same step. Submitting a reduce-only close AND a full bracket
            # entry simultaneously races in a netting account and triggers
            # Binance -2022 "ReduceOnly Order is rejected", leaving the
            # reversal half-done (position closed at a loss, new side never
            # cleanly established - observed live 2026-07-10). Re-entering on
            # the NEXT cycle (once flat, if the opposite signal persists) is
            # race-free and also cuts whipsaw: a one-bar counter-signal that
            # reverses immediately tends to buy the top / sell the bottom.
            self._submit_order(
                side=OrderSide.SELL if current_side == 'long' else OrderSide.BUY,
                quantity=current_qty,
                reduce_only=True,
            )
            # Next cycle sees current_position=None and opens fresh via the
            # normal (bracket-protected) entry path if the signal still holds.

        else:
            self.log.warning(
                f"⚠️ Signal suggests {target_side} but have {current_side} position. "
                f"Reversals disabled."
            )

    def _open_new_position(self, side: str, quantity: float):
        """
        Open new position using bracket order (entry + SL + TP).

        Also snapshots the entry signal so the trade outcome is labeled
        with the decision that opened it.

        This method submits a bracket order which automatically includes:
        - Entry order (MARKET)
        - Stop Loss order (STOP_MARKET)
        - Take Profit order(s) (LIMIT)

        The SL and TP orders are linked with OCO, so when one fills, the others cancel.
        """
        order_side = OrderSide.BUY if side == 'long' else OrderSide.SELL

        # Snapshot the signal that is opening this position
        if self.latest_signal_data:
            self._entry_signal_snapshot = {
                'signal': self.latest_signal_data.get('signal'),
                'confidence': self.latest_signal_data.get('confidence'),
            }

        # Submit bracket order with SL/TP
        self._submit_bracket_order(
            side=order_side,
            quantity=quantity,
        )

        self.log.info(f"🚀 Opening {side} position: {quantity:.3f} BTC (with bracket SL/TP)")

    def _submit_order(
        self,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False,
    ):
        """Submit market order to exchange."""
        if quantity < self.position_config['min_trade_amount']:
            self.log.warning(
                f"⚠️ Order quantity {quantity:.3f} below minimum "
                f"{self.position_config['min_trade_amount']:.3f}, skipping"
            )
            return

        # Create market order
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
            reduce_only=reduce_only,
        )

        # Submit order
        self.submit_order(order)

        self.log.info(
            f"📤 Submitted {side.name} market order: {quantity:.3f} BTC "
            f"(reduce_only={reduce_only})"
        )
    
    def _compute_sl_distance(
        self,
        is_buy: bool,
        entry_price: float,
        atr: float,
        support: float = 0.0,
        resistance: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Stop-loss distance (in price units) from ATR, clamped and optionally
        tightened by structure. Shared by the risk-based sizer and the
        bracket builder so both use the IDENTICAL stop - the sizer's risk
        math is only correct if the actual stop matches what it assumed.
        """
        if atr > 0:
            sl_distance = self.atr_sl_multiplier * atr
            sl_basis = f"{self.atr_sl_multiplier}×ATR(${atr:,.2f})"
        else:
            sl_distance = entry_price * 0.01  # Fallback: 1%
            sl_basis = "fallback 1%"

        # Clamp to [min_sl_pct, max_sl_pct] of entry price
        min_dist = entry_price * self.min_sl_pct
        max_dist = entry_price * self.max_sl_pct
        sl_distance = max(min_dist, min(sl_distance, max_dist))

        # Support/resistance refinement: adopt the S/R level (with buffer)
        # only when it is tighter than the ATR distance but not below the
        # minimum clamp - a stop just past structure beats one in dead space.
        if self.sl_use_support_resistance:
            sr_distance = None
            if is_buy and 0 < support < entry_price:
                sr_distance = entry_price - support * (1 - self.sl_buffer_pct)
            elif not is_buy and resistance > entry_price:
                sr_distance = resistance * (1 + self.sl_buffer_pct) - entry_price

            if sr_distance is not None and min_dist <= sr_distance < sl_distance:
                sl_distance = sr_distance
                sl_basis = "support/resistance (tighter than ATR)"

        return sl_distance, sl_basis

    def _compute_sl_tp(
        self,
        is_buy: bool,
        entry_price: float,
        confidence: str,
        atr: float,
        support: float = 0.0,
        resistance: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute stop-loss and take-profit prices from volatility.

        SL distance: ATR-based, clamped to [min_sl_pct, max_sl_pct] of entry.
        When sl_use_support_resistance is enabled and a support/resistance
        level (plus buffer) sits TIGHTER than the ATR distance, the S/R level
        is used instead - S/R can only tighten the stop, never widen it.

        TP distance: confidence-based %, floored at min_risk_reward × SL.

        (The old approach used the raw 20-bar low/high for SL, which could
        sit 3-5% away: at 10x leverage a 5% stop-out is 50% of the margin
        behind the trade, against a fixed 2% TP - inverted risk/reward.)
        """
        sl_distance, sl_basis = self._compute_sl_distance(
            is_buy, entry_price, atr, support, resistance
        )

        # TP distance by mode:
        # r_multiple - a fixed multiple of the SL distance. Closer targets
        #   get hit far more often (higher win rate profile).
        # confidence_pct - percentage target by AI confidence, R:R floored.
        if self.tp_mode == "r_multiple":
            tp_distance = self.tp_r_multiple * sl_distance
        else:
            tp_pct = self.tp_pct_config.get(confidence, 0.02)
            tp_distance = max(entry_price * tp_pct, self.min_risk_reward * sl_distance)

        if is_buy:
            stop_loss_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            stop_loss_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        rr = tp_distance / sl_distance if sl_distance > 0 else 0.0
        self.log.info(
            f"📍 SL/TP geometry: SL {sl_basis} = ${sl_distance:,.2f} "
            f"({sl_distance/entry_price*100:.2f}%), "
            f"TP ${tp_distance:,.2f} ({tp_distance/entry_price*100:.2f}%), "
            f"R:R = 1:{rr:.2f}"
        )

        return stop_loss_price, tp_price

    def _submit_bracket_order(
        self,
        side: OrderSide,
        quantity: float,
    ):
        """
        Submit a bracket order with entry, stop loss, and take profit using NautilusTrader's built-in bracket orders.

        This uses the OrderFactory.bracket() method which automatically creates:
        - Entry order (MARKET)
        - Stop Loss order (STOP_MARKET) linked with OTO (One-Triggers-Other)
        - Take Profit order (LIMIT) linked with OTO and OCO with SL

        The OCO linkage is handled automatically by NautilusTrader.

        Parameters
        ----------
        side : OrderSide
            Side of the entry order (BUY or SELL)
        quantity : float
            Quantity to trade
        """
        if quantity < self.position_config['min_trade_amount']:
            self.log.warning(
                f"⚠️ Order quantity {quantity:.3f} below minimum "
                f"{self.position_config['min_trade_amount']:.3f}, skipping"
            )
            return

        if not self.enable_auto_sl_tp:
            self.log.warning("⚠️ Auto SL/TP is disabled - submitting simple market order instead")
            self._submit_order(side=side, quantity=quantity, reduce_only=False)
            return

        if not self.latest_signal_data or not self.latest_technical_data:
            self.log.warning("⚠️ No signal/technical data available for SL/TP - submitting simple market order")
            self._submit_order(side=side, quantity=quantity, reduce_only=False)
            return

        # Determine latest price for entry estimation
        entry_price: Optional[float] = None

        if self.latest_price_data and self.latest_price_data.get('price'):
            entry_price = float(self.latest_price_data['price'])

        if entry_price is None and hasattr(self.indicator_manager, "recent_bars"):
            recent_bars = self.indicator_manager.recent_bars
            if recent_bars:
                entry_price = float(recent_bars[-1].close)

        if entry_price is None:
            cache_bars = self.cache.bars(self.bar_type)
            if cache_bars:
                entry_price = float(cache_bars[-1].close)

        if entry_price is None or entry_price <= 0:
            self.log.error("❌ Unable to determine entry price for bracket order, submitting market order instead")
            self._submit_order(side=side, quantity=quantity, reduce_only=False)
            return

        # Get confidence and technical data
        confidence = self.latest_signal_data.get('confidence', 'MEDIUM')
        atr = self.latest_technical_data.get('atr', 0.0)

        stop_loss_price, tp_price = self._compute_sl_tp(
            is_buy=(side == OrderSide.BUY),
            entry_price=entry_price,
            confidence=confidence,
            atr=atr,
            support=self.latest_technical_data.get('support', 0.0),
            resistance=self.latest_technical_data.get('resistance', 0.0),
        )

        # Log SL/TP summary
        self.log.info(
            f"🎯 Creating bracket order for {side.name}:\n"
            f"   Entry: ~${entry_price:,.2f} (MARKET)\n"
            f"   Stop Loss: ${stop_loss_price:,.2f} ({((stop_loss_price/entry_price - 1) * 100):.2f}%)\n"
            f"   Take Profit: ${tp_price:,.2f} ({((tp_price/entry_price - 1) * 100):.2f}%)\n"
            f"   Quantity: {quantity:.3f}\n"
            f"   Confidence: {confidence}"
        )

        try:
            # Create bracket order using OrderFactory
            # This automatically creates entry + SL + TP with OTO/OCO linkage
            # IMPORTANT: Use emulation_trigger to enable order emulation for Binance compatibility
            # Binance doesn't support native OCO+OTO orders, so NautilusTrader will emulate them.
            # In backtests emulation is disabled (bars-only data never feeds the emulator).
            emulation = (
                TriggerType.DEFAULT if self.config.use_order_emulation
                else TriggerType.NO_TRIGGER
            )
            bracket_order_list = self.order_factory.bracket(
                instrument_id=self.instrument_id,
                order_side=side,
                quantity=self.instrument.make_qty(quantity),
                sl_trigger_price=self.instrument.make_price(stop_loss_price),
                tp_price=self.instrument.make_price(tp_price),
                time_in_force=TimeInForce.GTC,
                emulation_trigger=emulation,
            )

            # Submit the bracket order list
            self.submit_order_list(bracket_order_list)

            self.log.info(
                f"✅ Submitted bracket order: {side.name} {quantity:.3f} BTC with SL/TP\n"
                f"   OrderList ID: {bracket_order_list.id}"
            )

            # Track SL/TP prices so protection can be re-synced after
            # adds/reduces/partial fills
            self.position_protection = {
                'sl_price': stop_loss_price,
                'tp_price': tp_price,
            }

            # Save bracket order info for trailing stop
            if self.enable_trailing_stop:
                instrument_key = str(self.instrument_id)

                # Extract SL order from bracket (it's typically the second order in the list)
                sl_order = None
                for order in bracket_order_list.orders:
                    if order.order_type == OrderType.STOP_MARKET:
                        sl_order = order
                        break

                if sl_order:
                    self.trailing_stop_state[instrument_key] = {
                        "entry_price": entry_price,
                        "highest_price": entry_price if side == OrderSide.BUY else None,
                        "lowest_price": entry_price if side == OrderSide.SELL else None,
                        "current_sl_price": stop_loss_price,
                        "sl_order_id": str(sl_order.client_order_id),
                        "activated": False,
                        "side": "LONG" if side == OrderSide.BUY else "SHORT",
                        "quantity": quantity,
                        "initial_risk": abs(entry_price - stop_loss_price),
                        "breakeven_done": False,
                        "entry_bar": self.bars_received,
                    }
                    self.log.debug(
                        f"📌 Saved SL order ID for trailing stop: {str(sl_order.client_order_id)[:8]}..."
                    )

        except Exception as e:
            self.log.error(f"❌ Failed to submit bracket order: {e}")
            self.log.warning("⚠️ Falling back to simple market order without SL/TP")
            self._submit_order(side=side, quantity=quantity, reduce_only=False)

    def on_order_filled(self, event):
        """
        Handle order filled events.

        Note: OCO logic is now handled automatically by NautilusTrader's bracket orders.
        We no longer need to manually cancel peer orders.
        """
        filled_order_id = str(event.client_order_id)

        self.log.info(
            f"✅ Order filled: {event.order_side.name} "
            f"{event.last_qty} @ {event.last_px} "
            f"(ID: {filled_order_id[:8]}...)"
        )

        # Send Telegram order fill notification
        if self.telegram_bot and self.enable_telegram and self.telegram_notify_fills:
            try:
                fill_msg = self.telegram_bot.format_order_fill({
                    'side': event.order_side.name,
                    'quantity': float(event.last_qty),
                    'price': float(event.last_px),
                    'order_type': 'MARKET',  # Could extract from order if needed
                })
                self.telegram_bot.send_message_sync(fill_msg)
            except Exception as e:
                self.log.warning(f"Failed to send Telegram fill notification: {e}")
    

    def on_order_rejected(self, event):
        """Handle order rejected events."""
        self.log.error(f"❌ Order rejected: {event.reason}")

    def on_position_opened(self, event):
        """
        Handle position opened events.

        Note: With bracket orders, SL/TP orders are automatically submitted as part of the bracket.
        We no longer need to manually submit them here.
        """
        # PositionOpened event contains position data directly
        self.log.info(
            f"🟢 Position opened: {event.side.name} "
            f"{event.quantity} @ {event.avg_px_open}"
        )

        # Update trailing stop state with actual entry price if it exists
        # (bracket order already initialized it with estimated price)
        if self.enable_trailing_stop:
            instrument_key = str(self.instrument_id)
            entry_price = float(event.avg_px_open)

            if instrument_key in self.trailing_stop_state:
                # Update with actual entry price
                state = self.trailing_stop_state[instrument_key]
                state["entry_price"] = entry_price
                state["entry_bar"] = self.bars_received
                if event.side == PositionSide.LONG:
                    state["highest_price"] = entry_price
                else:
                    state["lowest_price"] = entry_price

                self.log.debug(
                    f"📊 Updated trailing stop state with actual entry price: ${entry_price:,.2f}"
                )
            else:
                # Fallback: initialize if not already set (shouldn't happen with bracket orders)
                self.trailing_stop_state[instrument_key] = {
                    "entry_price": entry_price,
                    "highest_price": entry_price if event.side == PositionSide.LONG else None,
                    "lowest_price": entry_price if event.side == PositionSide.SHORT else None,
                    "current_sl_price": None,
                    "sl_order_id": None,
                    "activated": False,
                    "side": event.side.name,
                    "quantity": float(event.quantity),
                    "initial_risk": 0.0,
                    "breakeven_done": False,
                    "entry_bar": self.bars_received,
                }
                self.log.info(
                    f"📊 Trailing stop initialized for {event.side.name} position @ ${entry_price:,.2f}"
                )

        # Split the single bracket TP into partial take-profit levels
        # (only when each slice satisfies exchange minimums)
        if self.enable_partial_tp:
            try:
                self._setup_partial_tps(event)
            except Exception as e:
                self.log.error(f"❌ Partial TP setup failed (keeping single TP): {e}")

        # Send Telegram position opened notification
        if self.telegram_bot and self.enable_telegram and self.telegram_notify_positions:
            try:
                position_msg = self.telegram_bot.format_position_update({
                    'action': 'OPENED',
                    'side': event.side.name,
                    'quantity': float(event.quantity),
                    'entry_price': float(event.avg_px_open),
                    'current_price': float(event.avg_px_open),
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                })
                self.telegram_bot.send_message_sync(position_msg)
            except Exception as e:
                self.log.warning(f"Failed to send Telegram position opened notification: {e}")

    def on_position_changed(self, event):
        """
        Handle position changed events (adds, reduces, partial fills).

        Two cases:
        1. Signal-driven add/reduce (pending flag set): full protection
           refresh - cancel SL/TP and resubmit sized to the new quantity.
        2. Any other change (e.g. a partial-TP level filled): re-size ONLY
           the stop-loss to the remaining quantity via modify_order,
           leaving the rest of the TP ladder untouched.
        """
        if self._pending_protection_refresh:
            self._pending_protection_refresh = False
            try:
                self._refresh_protection_orders()
            except Exception as e:
                self.log.error(f"❌ Failed to refresh protection orders: {e}")
            return

        try:
            self._resync_sl_quantity()
        except Exception as e:
            self.log.error(f"❌ Failed to resync SL quantity: {e}")

    def _working_orders(self) -> List:
        """
        All working orders for this instrument: open AND emulated.

        Emulated orders (status EMULATED) live in the OrderEmulator and are
        NOT returned by cache.orders_open() - any logic that only checks
        open orders silently misses every emulated SL/TP.
        """
        orders = list(self.cache.orders_open(instrument_id=self.instrument_id))
        orders += [
            o for o in self.cache.orders_emulated(instrument_id=self.instrument_id)
            if o not in orders
        ]
        return orders

    def _resync_sl_quantity(self):
        """
        Keep the stop-loss quantity equal to the open position quantity.

        After a partial TP fills, the SL would otherwise remain sized to the
        original full position - oversized reduce-only stops can be rejected
        or behave unpredictably when triggered.
        """
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        if not positions:
            return

        qty = float(positions[0].quantity)
        if qty <= 0:
            return

        for order in self._working_orders():
            if (
                order.is_reduce_only
                and order.order_type == OrderType.STOP_MARKET
                and abs(float(order.quantity) - qty) > 1e-9
            ):
                self.modify_order(order, quantity=self.instrument.make_qty(qty))
                self.log.info(
                    f"🔧 SL quantity re-synced to remaining position: "
                    f"{float(order.quantity):.3f} → {qty:.3f} BTC"
                )

    def _refresh_protection_orders(self):
        """
        Re-sync SL/TP orders to the current position quantity.

        Cancels existing reduce-only protection orders and resubmits them
        sized to the full current position, at the tracked SL/TP prices.
        """
        positions = self.cache.positions_open(instrument_id=self.instrument_id)
        if not positions:
            return

        position = positions[0]
        qty = float(position.quantity)
        sl_price = self.position_protection.get('sl_price')
        tp_price = self.position_protection.get('tp_price')

        if not sl_price or not tp_price:
            self.log.warning(
                "⚠️ No tracked SL/TP prices to refresh protection orders with"
            )
            return

        # Cancel existing reduce-only protection orders (open AND emulated)
        for order in self._working_orders():
            if order.is_reduce_only:
                try:
                    self.cancel_order(order)
                except Exception as e:
                    self.log.warning(f"Failed to cancel protection order: {e}")

        exit_side = (
            OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        )

        # New SL sized to full position
        sl_order = self.order_factory.stop_market(
            instrument_id=self.instrument_id,
            order_side=exit_side,
            quantity=self.instrument.make_qty(qty),
            trigger_price=self.instrument.make_price(sl_price),
            trigger_type=TriggerType.LAST_PRICE,
            emulation_trigger=(
                TriggerType.LAST_PRICE if self.config.use_order_emulation
                else TriggerType.NO_TRIGGER
            ),
            reduce_only=True,
        )
        self.submit_order(sl_order)

        # New TP sized to full position
        tp_order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=exit_side,
            quantity=self.instrument.make_qty(qty),
            price=self.instrument.make_price(tp_price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        self.submit_order(tp_order)

        # Keep trailing stop state pointing at the fresh SL order
        instrument_key = str(self.instrument_id)
        if instrument_key in self.trailing_stop_state:
            self.trailing_stop_state[instrument_key]["sl_order_id"] = str(
                sl_order.client_order_id
            )
            self.trailing_stop_state[instrument_key]["quantity"] = qty

        self.log.info(
            f"🛡️ Protection orders re-synced: {qty:.3f} BTC, "
            f"SL ${sl_price:,.2f} / TP ${tp_price:,.2f}"
        )

    def _setup_partial_tps(self, event):
        """
        Replace the bracket's single TP with laddered partial TPs.

        Only activates when every slice satisfies BOTH the minimum trade
        amount AND the exchange minimum notional (~$100 on Binance Futures);
        otherwise the single bracket TP is kept and the reason is logged.
        (Previously enable_partial_tp was config-only dead code.)
        """
        if not self.partial_tp_levels:
            return

        entry_price = float(event.avg_px_open)
        total_qty = float(event.quantity)
        is_long = event.side == PositionSide.LONG
        MIN_NOTIONAL_USDT = 100.0
        min_qty = self.position_config['min_trade_amount']

        # Compute slices and validate all of them BEFORE touching any orders
        slices = []
        remaining = total_qty
        for i, level in enumerate(self.partial_tp_levels):
            profit_pct = float(level['profit_pct'])
            if i == len(self.partial_tp_levels) - 1:
                qty = remaining  # last level takes the remainder (no dust)
            else:
                import math
                qty = math.floor(total_qty * float(level['position_pct']) * 1000) / 1000
                remaining = round(remaining - qty, 3)

            tp_price = (
                entry_price * (1 + profit_pct) if is_long
                else entry_price * (1 - profit_pct)
            )

            if qty < min_qty or qty * tp_price < MIN_NOTIONAL_USDT:
                self.log.info(
                    f"ℹ️ Partial TP slice {i+1} too small "
                    f"({qty:.3f} BTC / ${qty * tp_price:.2f}) - "
                    f"keeping single bracket TP"
                )
                return

            slices.append((qty, tp_price))

        # Cancel the bracket's single TP (reduce-only LIMIT - it may be
        # EMULATED rather than open, so check working orders, not just open)
        for order in self._working_orders():
            if order.is_reduce_only and order.order_type == OrderType.LIMIT:
                self.cancel_order(order)

        # Submit laddered TP orders
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY
        for qty, tp_price in slices:
            tp_order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=exit_side,
                quantity=self.instrument.make_qty(qty),
                price=self.instrument.make_price(tp_price),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self.submit_order(tp_order)

        levels_str = ", ".join(
            f"{q:.3f}@${p:,.2f}" for q, p in slices
        )
        self.log.info(f"🪜 Partial TPs active: {levels_str}")

    def on_position_closed(self, event):
        """Handle position closed events."""
        realized_pnl = float(event.realized_pnl)
        self.log.info(
            f"🔴 Position closed: {event.side.name} "
            f"P&L: {realized_pnl:.2f} USDT"
        )

        # Record trade outcome for the AI feedback loop
        self._record_trade_outcome(event, realized_pnl)

        # A closed position invalidates any pending reversal confirmation
        self._opposite_signal_streak = 0

        # Cooldown after a losing trade - avoid immediate revenge re-entry
        if realized_pnl < 0 and self.loss_cooldown_bars > 0:
            self.cooldown_until_bar = self.bars_received + self.loss_cooldown_bars
            self.log.info(
                f"🧊 Loss cooldown active for the next "
                f"{self.loss_cooldown_bars} bars"
            )

        # Reversal: this close event belongs to the OLD position, but the
        # NEW opposite position's bracket may already be submitted. Blanket
        # cancellation here would destroy the new position's SL/TP (and its
        # trailing state) - the reversal path already cancelled the old
        # orders before closing.
        if self._reversal_in_progress:
            self._reversal_in_progress = False
            self.log.debug("Reversal close - keeping new position's orders/state")
            return

        # Cancel any remaining protection orders immediately (replaces the
        # old behavior of waiting up to a full timer cycle for orphan cleanup)
        self.cancel_all_orders(self.instrument_id)
        self.position_protection = {}

        # Clear trailing stop state
        instrument_key = str(self.instrument_id)
        if instrument_key in self.trailing_stop_state:
            del self.trailing_stop_state[instrument_key]
            self.log.debug(f"🗑️ Cleared trailing stop state for {instrument_key}")
        
        # Send Telegram position closed notification
        if self.telegram_bot and self.enable_telegram and self.telegram_notify_positions:
            try:
                # Calculate P&L percentage (approximate)
                pnl = float(event.realized_pnl)
                # Get rough position size estimate for percentage
                # Note: This is approximate, actual calculation would require more data
                pnl_pct = (pnl / 100.0) * 100 if pnl != 0 else 0.0  # Rough estimate
                
                position_msg = self.telegram_bot.format_position_update({
                    'action': 'CLOSED',
                    'side': event.side.name,
                    'quantity': float(event.quantity) if hasattr(event, 'quantity') else 0.0,
                    'entry_price': float(event.avg_px_open) if hasattr(event, 'avg_px_open') else 0.0,
                    'current_price': float(event.avg_px_close) if hasattr(event, 'avg_px_close') else 0.0,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                })
                self.telegram_bot.send_message_sync(position_msg)
            except Exception as e:
                self.log.warning(f"Failed to send Telegram position closed notification: {e}")
    
    def _cleanup_oco_orphans(self):
        """
        Clean up orphan orders.

        This is a safety mechanism that runs periodically to:
        1. Cancel orphan reduce-only orders when no position exists

        Note: OCO group management is no longer needed as NautilusTrader handles it automatically.
        """
        try:
            # Get current positions
            positions = self.cache.positions_open(instrument_id=self.instrument_id)
            has_position = len(positions) > 0

            if not has_position:
                # No position but check for orphan orders (open AND emulated)
                open_orders = self._working_orders()

                if open_orders:
                    orphan_count = 0
                    for order in open_orders:
                        if order.is_reduce_only:
                            # This is a reduce-only order without a position - orphan!
                            try:
                                self.cancel_order(order)
                                orphan_count += 1
                                self.log.info(
                                    f"🗑️ Cancelled orphan reduce-only order: "
                                    f"{str(order.client_order_id)[:8]}..."
                                )
                            except Exception as e:
                                self.log.error(
                                    f"Failed to cancel orphan order: {e}"
                                )

                    if orphan_count > 0:
                        self.log.warning(
                            f"⚠️ Cleaned up {orphan_count} orphan orders"
                        )

        except Exception as e:
            self.log.error(f"❌ Orphan order cleanup failed: {e}")
    
    def _update_trailing_stops(self, current_price: float):
        """
        Update trailing stop loss orders based on current price.
        
        Logic:
        1. Check if position is profitable enough to activate trailing stop
        2. Track highest price (LONG) or lowest price (SHORT)
        3. Update stop loss when price moves favorably beyond threshold
        4. Stop loss only moves in favorable direction, never backwards
        
        Parameters
        ----------
        current_price : float
            Current market price
        """
        try:
            instrument_key = str(self.instrument_id)
            
            # Check if we have trailing stop state for this instrument
            if instrument_key not in self.trailing_stop_state:
                return
            
            state = self.trailing_stop_state[instrument_key]
            entry_price = state["entry_price"]
            side = state["side"]
            activated = state["activated"]

            # Breakeven stop: once profit reaches trigger_r × initial risk,
            # move the SL to entry ± fee buffer - the trade can't lose anymore.
            self._check_breakeven(instrument_key, state, current_price)

            # Calculate profit percentage
            if side == "LONG":
                profit_pct = (current_price - entry_price) / entry_price
                
                # Update highest price
                if state["highest_price"] is None or current_price > state["highest_price"]:
                    state["highest_price"] = current_price
                
                highest_price = state["highest_price"]
                
                # Check if we should activate trailing stop
                if not activated and profit_pct >= self.trailing_activation_pct:
                    state["activated"] = True
                    self.log.info(
                        f"🎯 Trailing stop ACTIVATED for LONG @ ${current_price:,.2f} "
                        f"(Profit: {profit_pct*100:.2f}%)"
                    )
                    activated = True
                
                # If activated, check if we should update stop loss
                if activated:
                    # Calculate new stop loss based on highest price
                    new_sl_price = highest_price * (1 - self.trailing_distance_pct)
                    current_sl_price = state["current_sl_price"]
                    
                    # Only update if new SL is significantly higher than current
                    if current_sl_price is None:
                        should_update = True
                    else:
                        price_move_pct = (new_sl_price - current_sl_price) / current_sl_price
                        should_update = price_move_pct >= self.trailing_update_threshold_pct
                    
                    if should_update and (current_sl_price is None or new_sl_price > current_sl_price):
                        self._execute_trailing_stop_update(
                            instrument_key=instrument_key,
                            new_sl_price=new_sl_price,
                            current_price=current_price,
                            side="LONG"
                        )
            
            elif side == "SHORT":
                profit_pct = (entry_price - current_price) / entry_price
                
                # Update lowest price
                if state["lowest_price"] is None or current_price < state["lowest_price"]:
                    state["lowest_price"] = current_price
                
                lowest_price = state["lowest_price"]
                
                # Check if we should activate trailing stop
                if not activated and profit_pct >= self.trailing_activation_pct:
                    state["activated"] = True
                    self.log.info(
                        f"🎯 Trailing stop ACTIVATED for SHORT @ ${current_price:,.2f} "
                        f"(Profit: {profit_pct*100:.2f}%)"
                    )
                    activated = True
                
                # If activated, check if we should update stop loss
                if activated:
                    # Calculate new stop loss based on lowest price
                    new_sl_price = lowest_price * (1 + self.trailing_distance_pct)
                    current_sl_price = state["current_sl_price"]
                    
                    # Only update if new SL is significantly lower than current
                    if current_sl_price is None:
                        should_update = True
                    else:
                        price_move_pct = (current_sl_price - new_sl_price) / current_sl_price
                        should_update = price_move_pct >= self.trailing_update_threshold_pct
                    
                    if should_update and (current_sl_price is None or new_sl_price < current_sl_price):
                        self._execute_trailing_stop_update(
                            instrument_key=instrument_key,
                            new_sl_price=new_sl_price,
                            current_price=current_price,
                            side="SHORT"
                        )
                        
        except Exception as e:
            self.log.error(f"❌ Trailing stop update failed: {e}")
    
    def _check_breakeven(self, instrument_key: str, state: Dict[str, Any], current_price: float):
        """
        Move SL to breakeven once profit >= breakeven_trigger_r × initial risk.

        Runs before trailing-stop logic each bar. Once done, the flag is set
        so it never fires twice; the trailing stop takes over from there.
        """
        if not self.enable_breakeven_stop or state.get("breakeven_done"):
            return

        initial_risk = state.get("initial_risk") or 0.0
        if initial_risk <= 0:
            return  # unknown initial SL distance (e.g. fallback-initialized state)

        entry_price = state["entry_price"]
        side = state["side"]
        profit_abs = (
            current_price - entry_price if side == "LONG"
            else entry_price - current_price
        )

        if profit_abs < self.breakeven_trigger_r * initial_risk:
            return

        # Breakeven price with a small buffer to cover round-trip fees
        if side == "LONG":
            be_price = entry_price * (1 + self.breakeven_buffer_pct)
            improves = (
                state["current_sl_price"] is None
                or be_price > state["current_sl_price"]
            )
        else:
            be_price = entry_price * (1 - self.breakeven_buffer_pct)
            improves = (
                state["current_sl_price"] is None
                or be_price < state["current_sl_price"]
            )

        state["breakeven_done"] = True
        if not improves:
            return  # trailing already moved the SL past breakeven

        self.log.info(
            f"🛡️ BREAKEVEN: profit reached {self.breakeven_trigger_r:.1f}R - "
            f"moving SL to ${be_price:,.2f} (entry ${entry_price:,.2f})"
        )
        self._execute_trailing_stop_update(
            instrument_key=instrument_key,
            new_sl_price=be_price,
            current_price=current_price,
            side=side,
        )

    def _execute_trailing_stop_update(
        self,
        instrument_key: str,
        new_sl_price: float,
        current_price: float,
        side: str
    ):
        """
        Execute the actual update of trailing stop loss order.
        
        Parameters
        ----------
        instrument_key : str
            Instrument identifier
        new_sl_price : float
            New stop loss price
        current_price : float
            Current market price
        side : str
            Position side (LONG/SHORT)
        """
        try:
            state = self.trailing_stop_state[instrument_key]
            old_sl_price = state["current_sl_price"]
            old_sl_order_id = state["sl_order_id"]
            quantity = state["quantity"]
            
            # Log the update
            if old_sl_price:
                move_pct = ((new_sl_price - old_sl_price) / old_sl_price) * 100
                self.log.info(
                    f"⬆️ Trailing Stop Update ({side}):\n"
                    f"   Current Price: ${current_price:,.2f}\n"
                    f"   Old SL: ${old_sl_price:,.2f}\n"
                    f"   New SL: ${new_sl_price:,.2f} ({move_pct:+.2f}%)\n"
                    f"   Distance: {abs((new_sl_price - current_price) / current_price * 100):.2f}%"
                )
            else:
                self.log.info(
                    f"📍 Initial Trailing Stop ({side}):\n"
                    f"   Current Price: ${current_price:,.2f}\n"
                    f"   SL Price: ${new_sl_price:,.2f}\n"
                    f"   Distance: {abs((new_sl_price - current_price) / current_price * 100):.2f}%"
                )
            
            # Prefer modifying the existing SL order in place: this preserves
            # the bracket's OCO linkage with the TP order. (The old approach
            # cancelled the SL and submitted a standalone replacement, which
            # broke the contingency link - the TP could be orphaned or
            # cancelled along with the old SL leg.)
            modified = False
            old_order = None
            if old_sl_order_id:
                try:
                    from nautilus_trader.model.identifiers import ClientOrderId
                    old_order = self.cache.order(ClientOrderId(old_sl_order_id))

                    # Gate on "not closed" rather than "is_open": emulated
                    # orders (status EMULATED) are not open, but modify_order
                    # routes to the OrderEmulator and works fine for them.
                    if old_order is not None and not old_order.is_closed:
                        self.modify_order(
                            old_order,
                            trigger_price=self.instrument.make_price(new_sl_price),
                        )
                        modified = True
                        state["current_sl_price"] = new_sl_price
                        self.log.info(
                            f"✅ Trailing SL modified in place → ${new_sl_price:,.2f} "
                            f"(OCO link preserved)"
                        )
                except Exception as e:
                    self.log.warning(f"⚠️ modify_order failed, falling back: {e}")

            if not modified:
                # Fallback: cancel the old SL (if it still exists) BEFORE
                # submitting a replacement - otherwise stops accumulate.
                if old_order is not None and not old_order.is_closed:
                    try:
                        self.cancel_order(old_order)
                    except Exception as e:
                        self.log.warning(f"⚠️ Failed to cancel old SL: {e}")

                exit_side = OrderSide.SELL if side == "LONG" else OrderSide.BUY

                new_sl_order = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=exit_side,
                    quantity=self.instrument.make_qty(quantity),
                    trigger_price=self.instrument.make_price(new_sl_price),
                    trigger_type=TriggerType.LAST_PRICE,
                    emulation_trigger=(
                        TriggerType.LAST_PRICE if self.config.use_order_emulation
                        else TriggerType.NO_TRIGGER
                    ),
                    reduce_only=True,
                )
                self.submit_order(new_sl_order)

                state["current_sl_price"] = new_sl_price
                state["sl_order_id"] = str(new_sl_order.client_order_id)

                self.log.info(f"✅ New trailing SL order submitted @ ${new_sl_price:,.2f}")

            # Keep protection tracking in sync for later re-syncs
            if self.position_protection:
                self.position_protection['sl_price'] = new_sl_price

        except Exception as e:
            self.log.error(f"❌ Failed to execute trailing stop update: {e}")
    
    # ===== Trade Outcome Feedback (for AI context) =====

    def _record_trade_outcome(self, event, realized_pnl: float):
        """Record a closed trade so the AI sees its own track record."""
        # --- Risk-state updates FIRST (must never be skipped) ---

        # Loss-streak throttle counter
        self._last_trade_close_bar = self.bars_received
        if realized_pnl < 0:
            self.consecutive_losses += 1
            if (
                self.loss_streak_threshold > 0
                and self.consecutive_losses >= self.loss_streak_threshold
            ):
                self.log.warning(
                    f"📉 {self.consecutive_losses} consecutive losses - position "
                    f"size throttled to {self.loss_streak_multiplier:.0%} until next win"
                )
        elif realized_pnl > 0:
            self.consecutive_losses = 0

        # Daily loss circuit breaker
        self._day_realized_pnl += realized_pnl
        if (
            self.daily_loss_limit_pct > 0
            and not self._daily_breaker_active
            and self._day_start_equity
            and self._day_realized_pnl <= -self.daily_loss_limit_pct * self._day_start_equity
        ):
            self._daily_breaker_active = True
            self.log.error(
                f"🛑 DAILY LOSS LIMIT HIT: {self._day_realized_pnl:+.2f} USDT "
                f"(limit: -{self.daily_loss_limit_pct:.0%} of ${self._day_start_equity:,.2f}). "
                f"No new entries until the next UTC day."
            )
            if self.telegram_bot and self.enable_telegram:
                try:
                    self.telegram_bot.send_message_sync(
                        f"🛑 Daily loss limit hit ({self._day_realized_pnl:+.2f} USDT). "
                        f"Trading halted until next UTC day."
                    )
                except Exception:
                    pass

        # --- Trade history for the AI feedback loop ---
        try:
            # Use the ENTRY-time snapshot, falling back to the last signal
            # only if no snapshot exists (e.g. position from reconciliation)
            entry_signal = None
            entry_confidence = None
            if self._entry_signal_snapshot:
                entry_signal = self._entry_signal_snapshot.get('signal')
                entry_confidence = self._entry_signal_snapshot.get('confidence')
            elif self.last_signal:
                entry_signal = self.last_signal.get('signal')
                entry_confidence = self.last_signal.get('confidence')

            trade = {
                'side': event.side.name,
                'entry_price': float(event.avg_px_open) if hasattr(event, 'avg_px_open') else None,
                'realized_pnl': realized_pnl,
                'outcome': 'WIN' if realized_pnl > 0 else ('LOSS' if realized_pnl < 0 else 'FLAT'),
                'signal': entry_signal,
                'confidence': entry_confidence,
                'closed_at': self.clock.utc_now().isoformat(),
            }

            self.trade_history.append(trade)
            if len(self.trade_history) > self.max_trade_history:
                self.trade_history.pop(0)

            self._save_trade_history()
        except Exception as e:
            self.log.warning(f"Failed to record trade outcome: {e}")

    def _load_trade_history(self):
        """Load persisted trade history (survives restarts)."""
        try:
            import json
            if os.path.exists(self._trade_history_path):
                with open(self._trade_history_path, 'r') as f:
                    self.trade_history = json.load(f)[-self.max_trade_history:]
        except Exception:
            self.trade_history = []

    def _save_trade_history(self):
        """Persist trade history to logs/trade_history.json."""
        try:
            import json
            os.makedirs(os.path.dirname(self._trade_history_path), exist_ok=True)
            with open(self._trade_history_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.log.warning(f"Failed to persist trade history: {e}")

    # ===== Remote Control Methods (for Telegram commands) =====
    
    def handle_telegram_command(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Telegram commands.
        
        Parameters
        ----------
        command : str
            Command name (status, position, pause, resume)
        args : dict
            Command arguments
        
        Returns
        -------
        dict
            Response with 'success', 'message', and optional 'error'
        """
        try:
            if command == 'status':
                return self._cmd_status()
            elif command == 'position':
                return self._cmd_position()
            elif command == 'pause':
                return self._cmd_pause()
            elif command == 'resume':
                return self._cmd_resume()
            else:
                return {
                    'success': False,
                    'error': f"Unknown command: {command}"
                }
        except Exception as e:
            self.log.error(f"Error handling command '{command}': {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cmd_status(self) -> Dict[str, Any]:
        """Handle /status command."""
        try:
            from datetime import datetime
            
            # Get current price
            current_price = 0
            bars = self.indicator_manager.recent_bars if hasattr(self, 'indicator_manager') else []
            if bars:
                current_price = float(bars[-1].close)
            
            # Get unrealized PnL
            unrealized_pnl = 0
            positions = self.cache.positions_open(instrument_id=self.instrument_id)
            if positions:
                position = positions[0]
                if current_price > 0:
                    unrealized_pnl = float(position.unrealized_pnl(current_price))
            
            # Calculate uptime
            uptime_str = "N/A"
            if self.strategy_start_time:
                uptime_delta = datetime.utcnow() - self.strategy_start_time
                hours = uptime_delta.total_seconds() // 3600
                minutes = (uptime_delta.total_seconds() % 3600) // 60
                uptime_str = f"{int(hours)}h {int(minutes)}m"
            
            # Get last signal
            last_signal = "N/A"
            last_signal_time = "N/A"
            if hasattr(self, 'last_signal') and self.last_signal:
                last_signal = f"{self.last_signal.get('signal', 'N/A')} ({self.last_signal.get('confidence', 'N/A')})"
                # You could store timestamp if needed
            
            status_info = {
                'is_running': True,  # If this method is called, strategy is running
                'is_paused': self.is_trading_paused,
                'instrument_id': str(self.instrument_id),
                'current_price': current_price,
                'equity': self.equity,
                'unrealized_pnl': unrealized_pnl,
                'last_signal': last_signal,
                'last_signal_time': last_signal_time,
                'uptime': uptime_str,
            }
            
            message = self.telegram_bot.format_status_response(status_info) if self.telegram_bot else "Status unavailable"
            
            return {
                'success': True,
                'message': message
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cmd_position(self) -> Dict[str, Any]:
        """Handle /position command."""
        try:
            # Get current position
            current_position = self._get_current_position_data()
            
            position_info = {
                'has_position': current_position is not None,
            }
            
            if current_position:
                bars = self.indicator_manager.recent_bars if hasattr(self, 'indicator_manager') else []
                current_price = float(bars[-1].close) if bars else current_position['avg_px']
                
                entry_price = current_position['avg_px']
                pnl = current_position['unrealized_pnl']
                pnl_pct = (pnl / (entry_price * current_position['quantity'])) * 100 if entry_price > 0 else 0
                
                position_info.update({
                    'side': current_position['side'].upper(),
                    'quantity': current_position['quantity'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': pnl,
                    'pnl_pct': pnl_pct,
                    # SL/TP prices would need to be tracked separately if needed
                })
            
            message = self.telegram_bot.format_position_response(position_info) if self.telegram_bot else "Position unavailable"
            
            return {
                'success': True,
                'message': message
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cmd_pause(self) -> Dict[str, Any]:
        """Handle /pause command."""
        try:
            if self.is_trading_paused:
                message = self.telegram_bot.format_pause_response(False, "Trading is already paused") if self.telegram_bot else "Already paused"
            else:
                self.is_trading_paused = True
                self.log.info("⏸️ Trading paused by Telegram command")
                message = self.telegram_bot.format_pause_response(True) if self.telegram_bot else "Trading paused"
            
            return {
                'success': True,
                'message': message
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cmd_resume(self) -> Dict[str, Any]:
        """Handle /resume command."""
        try:
            if not self.is_trading_paused:
                message = self.telegram_bot.format_resume_response(False, "Trading is not paused") if self.telegram_bot else "Not paused"
            else:
                self.is_trading_paused = False
                self.log.info("▶️ Trading resumed by Telegram command")
                message = self.telegram_bot.format_resume_response(True) if self.telegram_bot else "Trading resumed"
            
            return {
                'success': True,
                'message': message
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
