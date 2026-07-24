"""
Live Trading Entrypoint for DeepSeek AI Strategy

Runs the DeepSeek AI strategy on Binance Futures (BTCUSDT-PERP) with live market data.
"""

import os
import asyncio
import yaml
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig, BinanceExecClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory, BinanceLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig, LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId, InstrumentId
from nautilus_trader.trading.config import ImportableStrategyConfig

from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig


# Load environment variables
load_dotenv()


def load_yaml_config() -> dict:
    """
    Load strategy configuration from YAML file.
    
    Returns
    -------
    dict
        Configuration dictionary from YAML
    """
    config_path = Path(__file__).parent / "configs" / "strategy_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_env_float(key: str, default: str) -> float:
    """
    Safely get float environment variable, removing any inline comments.
    """
    value = os.getenv(key, default)
    # Remove inline comments (anything after #)
    if '#' in value:
        value = value.split('#')[0]
    # Strip whitespace
    value = value.strip()
    return float(value)


def get_env_str(key: str, default: str) -> str:
    """
    Safely get string environment variable, removing any inline comments.
    """
    value = os.getenv(key, default)
    # Remove inline comments (anything after #)
    if '#' in value:
        value = value.split('#')[0]
    # Strip whitespace
    return value.strip()


def get_env_int(key: str, default: str) -> int:
    """
    Safely get integer environment variable, removing any inline comments.
    """
    value = os.getenv(key, default)
    # Remove inline comments (anything after #)
    if '#' in value:
        value = value.split('#')[0]
    # Strip whitespace
    value = value.strip()
    return int(value)


def get_strategy_config() -> DeepSeekAIStrategyConfig:
    """
    Build strategy configuration from environment variables.

    Returns
    -------
    DeepSeekAIStrategyConfig
        Strategy configuration
    """
    # Get API keys
    deepseek_api_key = get_env_str('DEEPSEEK_API_KEY', '')
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

    # Load YAML config
    yaml_config = load_yaml_config()
    strategy_yaml = yaml_config.get('strategy', {})
    
    # Get strategy parameters from env or use defaults
    equity = get_env_float('EQUITY', str(strategy_yaml.get('equity', '400')))
    leverage = get_env_float('LEVERAGE', str(strategy_yaml.get('leverage', '10')))
    base_position = get_env_float('BASE_POSITION_USDT', str(strategy_yaml.get('position_management', {}).get('base_usdt_amount', '30')))
    timeframe = get_env_str('TIMEFRAME', '15m')  # Production: 15-minute timeframe
    
    # Debug output
    print(f"[CONFIG] Equity: {equity}")
    print(f"[CONFIG] Base Position: {base_position}")
    print(f"[CONFIG] Timeframe: {timeframe}")

    # Map timeframe to bar aggregation
    timeframe_map = {
        '1m': 'MINUTE',
        '5m': 'MINUTE',
        '15m': 'MINUTE',
        '1h': 'HOUR',
        '4h': 'HOUR',
        '1d': 'DAY',
    }

    # Parse timeframe
    print(f"DEBUG: timeframe = '{timeframe}'")  # Debug
    if timeframe == '1m':
        bar_spec = '1-MINUTE-LAST'
    elif timeframe == '5m':
        bar_spec = '5-MINUTE-LAST'
    elif timeframe == '15m':
        bar_spec = '15-MINUTE-LAST'
    elif timeframe == '1h':
        bar_spec = '1-HOUR-LAST'
    else:
        bar_spec = '15-MINUTE-LAST'  # Default
    
    print(f"DEBUG: bar_spec = '{bar_spec}'")  # Debug
    final_bar_type = f"BTCUSDT-PERP.BINANCE-{bar_spec}-EXTERNAL"
    print(f"DEBUG: final_bar_type = '{final_bar_type}'")  # Debug

    return DeepSeekAIStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type=final_bar_type,

        # Capital
        equity=equity,
        leverage=leverage,

        # Position sizing
        sizing_mode=get_env_str('SIZING_MODE', 'risk'),
        risk_per_trade_pct=get_env_float('RISK_PER_TRADE_PCT', '0.01'),
        base_usdt_amount=base_position,
        high_confidence_multiplier=get_env_float('HIGH_CONFIDENCE_MULTIPLIER', '1.5'),
        medium_confidence_multiplier=get_env_float('MEDIUM_CONFIDENCE_MULTIPLIER', '1.0'),
        low_confidence_multiplier=get_env_float('LOW_CONFIDENCE_MULTIPLIER', '0.5'),
        max_position_ratio=get_env_float('MAX_POSITION_RATIO', '0.20'),
        trend_strength_multiplier=get_env_float('TREND_STRENGTH_MULTIPLIER', '1.2'),
        min_trade_amount=0.001,  # Binance minimum

        # Technical indicators - Production mode (standard periods)
        # Use reduced periods only for 1m bars (for testing)
        sma_periods=[3, 7, 15] if timeframe == '1m' else [5, 20, 50],
        rsi_period=7 if timeframe == '1m' else 14,
        macd_fast=5 if timeframe == '1m' else 12,
        macd_slow=10 if timeframe == '1m' else 26,
        bb_period=10 if timeframe == '1m' else 20,
        bb_std=2.0,

        # AI
        deepseek_api_key=deepseek_api_key,
        deepseek_model="deepseek-v4-pro",
        deepseek_temperature=0.1,
        deepseek_max_retries=2,

        # Sentiment (off by default - CryptoOracle endpoint has been
        # unreliable/timing out; the AI runs on technicals when absent)
        sentiment_enabled=get_env_str('SENTIMENT_ENABLED', 'false').lower() == 'true',
        sentiment_lookback_hours=4,
        # Set sentiment timeframe based on bar timeframe (default to 15m)
        sentiment_timeframe="1m" if timeframe == "1m" else ("5m" if timeframe == "5m" else "15m"),

        # Risk
        min_confidence_to_trade=get_env_str('MIN_CONFIDENCE_TO_TRADE', 'MEDIUM'),
        allow_reversals=True,
        # Only flip an open position on a HIGH-confidence opposite signal.
        # MEDIUM signals flip-flop in chop and reverse-close at local extremes
        # (e.g. 2026-07-16: covered a short at the bounce high, re-shorted
        # lower for a churn loss). Backtest-neutral on the rule analyzer.
        require_high_confidence_for_reversal=get_env_str(
            'REQUIRE_HIGH_CONFIDENCE_FOR_REVERSAL', 'true').lower() == 'true',
        rsi_extreme_threshold_upper=75.0,
        rsi_extreme_threshold_lower=25.0,
        rsi_extreme_multiplier=0.7,

        # ATR-based SL/TP geometry
        atr_sl_multiplier=get_env_float('ATR_SL_MULTIPLIER', '1.5'),
        min_sl_pct=get_env_float('MIN_SL_PCT', '0.003'),
        max_sl_pct=get_env_float('MAX_SL_PCT', '0.015'),
        min_risk_reward=get_env_float('MIN_RISK_REWARD', '1.5'),

        # Higher-timeframe trend filter
        enable_htf_filter=get_env_str('ENABLE_HTF_FILTER', 'true').lower() == 'true',

        # Loss cooldown (bars) and sizing source
        loss_cooldown_bars=get_env_int('LOSS_COOLDOWN_BARS', '2'),
        use_account_balance=get_env_str('USE_ACCOUNT_BALANCE', 'true').lower() == 'true',

        # Profit protection & circuit breakers
        enable_breakeven_stop=get_env_str('ENABLE_BREAKEVEN_STOP', 'true').lower() == 'true',
        daily_loss_limit_pct=get_env_float('DAILY_LOSS_LIMIT_PCT', '0.05'),
        loss_streak_threshold=get_env_int('LOSS_STREAK_THRESHOLD', '2'),
        max_position_age_bars=get_env_int('MAX_POSITION_AGE_BARS', '48'),
        reversal_confirmation_signals=get_env_int('REVERSAL_CONFIRMATION_SIGNALS', '2'),
        # Anti-churn + exhaustion gates (post-mortem 2026-07-22/23: all 4
        # trades lost - naked-SL race, size churn, exhaustion-chase entries)
        allow_position_adds=get_env_str('ALLOW_POSITION_ADDS', 'true').lower() == 'true',
        rsi_entry_gate_upper=get_env_float('RSI_ENTRY_GATE_UPPER', '70'),
        rsi_entry_gate_lower=get_env_float('RSI_ENTRY_GATE_LOWER', '30'),
        min_atr_pct_to_trade=get_env_float('MIN_ATR_PCT_TO_TRADE', '0.001'),
        min_efficiency_ratio=get_env_float('MIN_EFFICIENCY_RATIO', '0.25'),
        # State-desync self-heal: on the reduce-only tripwire, re-query Binance
        # for the true position and resume, or auto-restart on a confirmed
        # phantom (needs a supervisor like systemd Restart=always).
        desync_rest_recovery=get_env_str('DESYNC_REST_RECOVERY', 'true').lower() == 'true',
        desync_recovery_interval_secs=get_env_float('DESYNC_RECOVERY_INTERVAL_SECS', '60'),
        auto_restart_on_desync=get_env_str('AUTO_RESTART_ON_DESYNC', 'true').lower() == 'true',
        enable_partial_tp=get_env_str('ENABLE_PARTIAL_TP', 'false').lower() == 'true',
        max_signal_staleness_pct=get_env_float('MAX_SIGNAL_STALENESS_PCT', '0.0015'),

        # TP mode and HTF strictness (re-validated July 2026 under REAL fees)
        tp_mode=get_env_str('TP_MODE', 'r_multiple'),
        tp_r_multiple=get_env_float('TP_R_MULTIPLE', '2.0'),
        htf_strict_alignment=get_env_str('HTF_STRICT_ALIGNMENT', 'true').lower() == 'true',

        # Live-safety: flatten on stop (emulated SL/TP die with the process);
        # set USE_ORDER_EMULATION=false to place native venue stops instead
        # (survive crashes, but validate order acceptance on small size first)
        close_positions_on_stop=get_env_str('CLOSE_POSITIONS_ON_STOP', 'true').lower() == 'true',
        use_order_emulation=get_env_str('USE_ORDER_EMULATION', 'true').lower() == 'true',

        # Analysis trigger: bar close (fresh data) vs legacy timer
        analyze_on_bar_close=get_env_str('ANALYZE_ON_BAR_CLOSE', 'true').lower() == 'true',

        # Data source: 'rest' polls Binance REST klines (reliable); 'websocket'
        # uses the kline push stream (observed to silently stop delivering).
        analysis_source=get_env_str('ANALYSIS_SOURCE', 'rest'),

        # Execution
        position_adjustment_threshold=0.001,

        # Timing - only used when ANALYZE_ON_BAR_CLOSE=false
        timer_interval_sec=get_env_int('TIMER_INTERVAL_SEC', str(strategy_yaml.get('timer_interval_sec', 900))),
        
        # Telegram Notifications - env var takes precedence over YAML
        enable_telegram=get_env_str(
            'ENABLE_TELEGRAM',
            str(strategy_yaml.get('telegram', {}).get('enabled', False)),
        ).lower() == 'true',
        telegram_bot_token=get_env_str('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=get_env_str('TELEGRAM_CHAT_ID', ''),
        telegram_notify_signals=strategy_yaml.get('telegram', {}).get('notify_signals', True),
        telegram_notify_fills=strategy_yaml.get('telegram', {}).get('notify_fills', True),
        telegram_notify_positions=strategy_yaml.get('telegram', {}).get('notify_positions', True),
        telegram_notify_errors=strategy_yaml.get('telegram', {}).get('notify_errors', True),
    )


def get_binance_config() -> tuple:
    """
    Build Binance data and execution client configs.

    Returns
    -------
    tuple
        (data_config, exec_config)
    """
    # Get API credentials
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET required in .env")

    # Load ONLY the instrument we trade.
    # load_all=True pulls every Binance futures instrument AND re-loads them
    # hourly. Binance recently added "TRADIFI_PERPETUAL" contracts (tokenized
    # stocks: GMEUSDT, RIVNUSDT, ADBEUSDT, ...) that this nautilus_trader
    # version cannot parse, spamming ~122 warnings/hour and churning the
    # instrument set on every reload. Scoping to BTCUSDT-PERP removes that
    # entire failure surface - we only ever trade this one instrument.
    instrument_ids = frozenset(["BTCUSDT-PERP.BINANCE"])

    # Data client config
    data_config = BinanceDataClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=BinanceAccountType.USDT_FUTURES,  # Binance Futures
        instrument_provider=InstrumentProviderConfig(load_ids=instrument_ids),
    )

    # Execution client config
    exec_config = BinanceExecClientConfig(
        api_key=api_key,
        api_secret=api_secret,
        account_type=BinanceAccountType.USDT_FUTURES,
        instrument_provider=InstrumentProviderConfig(load_ids=instrument_ids),
    )

    return data_config, exec_config


def setup_trading_node() -> TradingNodeConfig:
    """
    Configure the NautilusTrader trading node.

    Returns
    -------
    TradingNodeConfig
        Trading node configuration
    """
    # Get configurations
    strategy_config = get_strategy_config()
    data_config, exec_config = get_binance_config()

    # Wrap strategy config in ImportableStrategyConfig
    importable_config = ImportableStrategyConfig(
        strategy_path="strategy.deepseek_strategy:DeepSeekAIStrategy",
        config_path="strategy.deepseek_strategy:DeepSeekAIStrategyConfig",
        config=strategy_config.dict(),
    )

    # Logging configuration
    log_level = get_env_str('LOG_LEVEL', 'INFO')

    logging_config = LoggingConfig(
        log_level=log_level,
        log_level_file=log_level,
        log_directory="logs",
        log_file_name="deepseek_trader",
        log_file_format="json",
        log_colors=True,
        bypass_logging=False,
        log_file_max_size=10_485_760,  # 10MB in bytes (10 * 1024 * 1024)
        log_file_max_backup_count=3,   # Keep 3 backup files (total: 40MB max)
    )

    # Exec-engine config. CONTINUOUS reconciliation (poll Binance REST for
    # position/order truth every 5 min) repairs mid-run desync - observed
    # live 2026-07-13 when the user-data stream went quiet and the bot
    # managed a fantasy position for 15h. Those knobs only exist on newer
    # nautilus_trader; build tolerantly so the bot still runs on older
    # versions (the strategy-side desync tripwire protects either way).
    exec_engine_kwargs = dict(
        reconciliation=True,               # startup reconciliation
        inflight_check_interval_ms=5000,   # check in-flight orders every 5s
    )
    for _k, _v in (("position_check_interval_secs", 300.0),
                   ("open_check_interval_secs", 300.0)):
        exec_engine_kwargs[_k] = _v
    try:
        exec_engine_config = LiveExecEngineConfig(**exec_engine_kwargs)
    except TypeError:
        for _k in ("position_check_interval_secs", "open_check_interval_secs"):
            exec_engine_kwargs.pop(_k, None)
        exec_engine_config = LiveExecEngineConfig(**exec_engine_kwargs)
        print("⚠️  nautilus_trader lacks continuous-reconciliation knobs - "
              "using startup reconciliation only. Upgrade nautilus_trader "
              "for auto-heal; the desync tripwire still protects trading.")

    # Trading node config
    config = TradingNodeConfig(
        trader_id=TraderId("DeepSeekTrader-001"),
        logging=logging_config,
        exec_engine=exec_engine_config,
        # Data clients
        data_clients={
            "BINANCE": data_config,
        },
        # Execution clients
        exec_clients={
            "BINANCE": exec_config,
        },
        # Strategy configs
        strategies=[importable_config],
    )

    return config


def main():
    """
    Main entry point for live trading.
    """
    print("=" * 70)
    print("DeepSeek AI Trading Strategy - Live Trading Mode")
    print("=" * 70)
    print(f"Exchange: Binance Futures (USDT-M)")
    print(f"Instrument: BTCUSDT-PERP")
    print(f"Strategy: AI-powered with DeepSeek")
    print("=" * 70)

    # Safety check
    test_mode = os.getenv('TEST_MODE', 'false').strip().lower() == 'true'
    auto_confirm = os.getenv('AUTO_CONFIRM', 'false').strip().lower() == 'true'
    
    if test_mode:
        print("⚠️  TEST_MODE=true - This is a simulation, no real orders will be placed")
    else:
        print("🚨 LIVE TRADING MODE - Real orders will be placed!")
        if auto_confirm:
            print("⚠️  AUTO_CONFIRM=true - Skipping user confirmation")
        else:
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Exiting...")
                return

    # Build configuration
    print("\n📋 Building configuration...")
    config = setup_trading_node()

    print(f"✅ Trader ID: {config.trader_id}")
    print(f"✅ Strategy configured with DeepSeek AI")
    print(f"✅ Binance Futures adapter configured")

    # Create and start trading node
    print("\n🚀 Starting trading node...")
    node = TradingNode(config=config)
    
    # Register Binance factories
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
    print("✅ Binance factories registered")

    try:
        # Build the node (connects to exchange, loads instruments)
        node.build()
        print("✅ Trading node built successfully")

        # Run the node (this starts strategies and begins event processing)
        print("✅ Starting trading node...")
        print("\n🟢 Strategy is now running. Press Ctrl+C to stop.\n")
        
        # Run the node - this is a blocking call that processes all events
        node.run()

    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt received...")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Dispose the node to clean up resources
        print("\n🛑 Cleaning up resources...")
        node.dispose()
        print("✅ Resources cleaned up")

        print("\n" + "=" * 70)
        print("Trading session ended")
        print("=" * 70)


if __name__ == "__main__":
    # Check Python path
    import sys
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Run the trading bot
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Program terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
