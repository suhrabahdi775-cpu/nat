"""
Telegram Bot for Trading Notifications

Provides real-time notifications for trading signals, order fills,
position updates, and system status via Telegram.
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None
    TelegramError = Exception


class TelegramBot:
    """
    Telegram Bot for sending trading notifications.
    
    Features:
    - Send formatted trading signals
    - Send order fill notifications
    - Send position updates
    - Send error/warning alerts
    - Async message sending
    - Rate limiting support
    """
    
    def __init__(
        self,
        token: str,
        chat_id: str,
        logger: Optional[logging.Logger] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram Bot.
        
        Parameters
        ----------
        token : str
            Telegram Bot token from @BotFather
        chat_id : str
            Telegram chat ID to send messages to
        logger : Optional[logging.Logger]
            Logger instance for logging
        enabled : bool
            Whether the bot is enabled (default: True)
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is not installed. "
                "Install it with: pip install python-telegram-bot"
            )
        
        self.token = token
        self.chat_id = chat_id
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enabled

        # Dedicated event loop and thread for all Telegram I/O
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="telegram-io",
            daemon=True
        )
        self._thread.start()

        # Initialize bot
        try:
            self.bot = Bot(token=token)
            self.logger.info("✅ Telegram Bot initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Telegram Bot: {e}")
            self.enabled = False
            raise
    
    async def send_message(
        self,
        message: str,
        parse_mode: str = 'Markdown',
        disable_notification: bool = False
    ) -> bool:
        """
        Send a text message to Telegram.

        Parameters
        ----------
        message : str
            Message text to send
        parse_mode : str
            Parse mode for formatting (Markdown, HTML, or None)
        disable_notification : bool
            Send silently without notification

        Returns
        -------
        bool
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            self.logger.debug("Telegram bot is disabled, skipping message")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            self.logger.info(f"📱 Telegram message sent: {message[:50]}...")
            return True

        except TelegramError as e:
            # If parse error, retry without formatting
            if "can't parse" in str(e).lower() or "parse entities" in str(e).lower():
                self.logger.warning(f"⚠️ Markdown parse error, retrying without formatting: {e}")
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=None,  # Send as plain text
                        disable_notification=disable_notification
                    )
                    self.logger.info(f"📱 Telegram message sent (plain text): {message[:50]}...")
                    return True
                except Exception as retry_e:
                    self.logger.error(f"❌ Failed to send even without formatting: {retry_e}")
                    return False
            else:
                self.logger.error(f"❌ Telegram error: {e}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def send_message_sync(self, message: str, **kwargs) -> bool:
        """
        Synchronous wrapper for send_message.

        Submits the coroutine to the dedicated background event loop so the
        same loop (and its underlying connection pool) is reused for every
        message, avoiding pool-exhaustion errors.
        """
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.send_message(message, **kwargs),
                self._loop
            )
            return future.result(timeout=15)
        except Exception as e:
            self.logger.error(f"❌ Error in send_message_sync: {e}")
            return False
    
    # Message Formatters
    
    def format_startup_message(self, instrument_id: str, config: Dict[str, Any]) -> str:
        """Format strategy startup notification."""
        return f"""
🚀 *Strategy Started*

📊 *Instrument*: {instrument_id}
⏰ *Timeframe*: 15 minutes
🕐 *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

✅ *Features Enabled*:
• Automatic SL/TP
• OCO Management
• Trailing Stop Loss
• Partial Take Profit
• Redis Persistence

🎯 Strategy is now monitoring the market...
"""
    
    def format_trade_signal(self, signal_data: Dict[str, Any]) -> str:
        """Format trading signal notification."""
        signal = signal_data.get('signal', 'UNKNOWN')
        confidence = signal_data.get('confidence', 'UNKNOWN')
        price = signal_data.get('price', 0.0)
        timestamp = signal_data.get('timestamp', datetime.now())
        
        # Technical indicators
        rsi = signal_data.get('rsi', 0.0)
        macd = signal_data.get('macd', 0.0)
        support = signal_data.get('support', 0.0)
        resistance = signal_data.get('resistance', 0.0)
        
        # AI reasoning
        reasoning = signal_data.get('reasoning', 'No reasoning provided')
        
        # Signal emoji
        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
        
        return f"""
{signal_emoji} *Trading Signal Detected*

*Signal*: {signal}
*Confidence*: {confidence}
*Price*: ${price:,.2f}
*Time*: {timestamp}

📈 *Technical Indicators*:
• RSI: {rsi:.2f}
• MACD: {macd:.4f}
• Support: ${support:,.2f}
• Resistance: ${resistance:,.2f}

🤖 *AI Analysis*:
{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}
"""
    
    def format_order_fill(self, order_data: Dict[str, Any]) -> str:
        """Format order fill notification."""
        side = order_data.get('side', 'UNKNOWN')
        quantity = order_data.get('quantity', 0.0)
        price = order_data.get('price', 0.0)
        order_type = order_data.get('order_type', 'MARKET')
        
        side_emoji = "🟢" if side == "BUY" else "🔴" if side == "SELL" else "⚪"
        
        return f"""
{side_emoji} *Order Filled*

*Side*: {side}
*Type*: {order_type}
*Quantity*: {quantity} BTC
*Price*: ${price:,.2f}
*Total*: ${quantity * price:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def format_position_update(self, position_data: Dict[str, Any]) -> str:
        """Format position update notification."""
        action = position_data.get('action', 'UPDATE')  # OPENED, CLOSED, UPDATE
        side = position_data.get('side', 'UNKNOWN')
        quantity = position_data.get('quantity', 0.0)
        entry_price = position_data.get('entry_price', 0.0)
        current_price = position_data.get('current_price', 0.0)
        pnl = position_data.get('pnl', 0.0)
        pnl_pct = position_data.get('pnl_pct', 0.0)
        
        if action == "OPENED":
            emoji = "📈" if side == "LONG" else "📉"
            title = "Position Opened"
        elif action == "CLOSED":
            emoji = "✅" if pnl >= 0 else "❌"
            title = "Position Closed"
        else:
            emoji = "📊"
            title = "Position Update"
        
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        message = f"""
{emoji} *{title}*

*Side*: {side}
*Quantity*: {quantity} BTC
*Entry Price*: ${entry_price:,.2f}
*Current Price*: ${current_price:,.2f}
"""
        
        if action == "CLOSED" or action == "UPDATE":
            message += f"""
{pnl_emoji} *P&L*: ${pnl:,.2f} ({pnl_pct:+.2f}%)
"""
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return message
    
    def format_error_alert(self, error_data: Dict[str, Any]) -> str:
        """Format error/warning notification."""
        level = error_data.get('level', 'ERROR')  # ERROR, WARNING, CRITICAL
        message = error_data.get('message', 'Unknown error')
        context = error_data.get('context', '')
        
        if level == "CRITICAL":
            emoji = "🚨"
        elif level == "WARNING":
            emoji = "⚠️"
        else:
            emoji = "❌"
        
        formatted = f"""
{emoji} *{level}*

{message}
"""
        
        if context:
            formatted += f"\n*Context*: {context}\n"
        
        formatted += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return formatted
    
    def format_partial_tp_notification(self, tp_data: Dict[str, Any]) -> str:
        """Format partial take profit notification."""
        level = tp_data.get('level', 1)
        quantity = tp_data.get('quantity', 0.0)
        price = tp_data.get('price', 0.0)
        profit_pct = tp_data.get('profit_pct', 0.0)
        remaining = tp_data.get('remaining_quantity', 0.0)
        
        return f"""
🎯 *Partial Take Profit - Level {level}*

*Closed*: {quantity} BTC
*Price*: ${price:,.2f}
*Profit*: +{profit_pct*100:.1f}%
*Remaining*: {remaining} BTC

✅ Profit locked in, position partially closed.

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def format_trailing_stop_update(self, ts_data: Dict[str, Any]) -> str:
        """Format trailing stop update notification."""
        old_sl = ts_data.get('old_sl_price', 0.0)
        new_sl = ts_data.get('new_sl_price', 0.0)
        current_price = ts_data.get('current_price', 0.0)
        profit_pct = ts_data.get('profit_pct', 0.0)
        
        return f"""
🔄 *Trailing Stop Updated*

*Current Price*: ${current_price:,.2f}
*Profit*: +{profit_pct*100:.1f}%

*Stop Loss*:
  Old: ${old_sl:,.2f}
  New: ${new_sl:,.2f} ⬆️

🛡️ Stop loss moved up to lock in more profit!

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns
        -------
        bool
            True if connection successful, False otherwise
        """
        try:
            me = await self.bot.get_me()
            self.logger.info(f"✅ Connected to Telegram as @{me.username}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Telegram: {e}")
            return False
    
    # ===== Remote Control Command Formatters =====
    
    def format_status_response(self, status_info: Dict[str, Any]) -> str:
        """
        Format strategy status response for /status command.
        
        Parameters
        ----------
        status_info : dict
            Status information containing:
            - is_running: bool
            - is_paused: bool
            - instrument_id: str
            - current_price: float
            - equity: float
            - unrealized_pnl: float
            - last_signal: str
            - last_signal_time: str
            - uptime: str
        """
        is_running = status_info.get('is_running', False)
        is_paused = status_info.get('is_paused', False)
        
        # Status emoji
        if not is_running:
            status_emoji = "🔴"
            status_text = "STOPPED"
        elif is_paused:
            status_emoji = "⏸️"
            status_text = "PAUSED"
        else:
            status_emoji = "🟢"
            status_text = "RUNNING"
        
        msg = f"{status_emoji} *Strategy Status*\n\n"
        msg += f"*Status*: {status_text}\n"
        msg += f"*Instrument*: {status_info.get('instrument_id', 'N/A')}\n"
        msg += f"*Current Price*: ${status_info.get('current_price', 0):,.2f}\n"
        msg += f"*Equity*: ${status_info.get('equity', 0):,.2f}\n"
        
        pnl = status_info.get('unrealized_pnl', 0)
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
        msg += f"*Unrealized P&L*: {pnl_emoji} ${pnl:,.2f}\n\n"
        
        msg += f"*Last Signal*: {status_info.get('last_signal', 'N/A')}\n"
        msg += f"*Signal Time*: {status_info.get('last_signal_time', 'N/A')}\n"
        msg += f"*Uptime*: {status_info.get('uptime', 'N/A')}\n"
        
        return msg
    
    def format_position_response(self, position_info: Dict[str, Any]) -> str:
        """
        Format position information response for /position command.
        
        Parameters
        ----------
        position_info : dict
            Position information containing:
            - has_position: bool
            - side: LONG/SHORT
            - quantity: float
            - entry_price: float
            - current_price: float
            - unrealized_pnl: float
            - pnl_pct: float
            - sl_price: float (optional)
            - tp_price: float (optional)
        """
        if not position_info.get('has_position', False):
            return "ℹ️ *No Open Position*\n\nCurrently not holding any position."
        
        side = position_info.get('side', 'UNKNOWN')
        side_emoji = "🟢" if side == "LONG" else "🔴" if side == "SHORT" else "⚪"
        
        msg = f"{side_emoji} *Open Position*\n\n"
        msg += f"*Side*: {side}\n"
        msg += f"*Quantity*: {position_info.get('quantity', 0):.4f}\n"
        msg += f"*Entry Price*: ${position_info.get('entry_price', 0):,.2f}\n"
        msg += f"*Current Price*: ${position_info.get('current_price', 0):,.2f}\n\n"
        
        pnl = position_info.get('unrealized_pnl', 0)
        pnl_pct = position_info.get('pnl_pct', 0)
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
        msg += f"*Unrealized P&L*: {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)\n\n"
        
        # Add SL/TP if available
        sl_price = position_info.get('sl_price')
        tp_price = position_info.get('tp_price')
        
        if sl_price:
            msg += f"🛡️ *Stop Loss*: ${sl_price:,.2f}\n"
        if tp_price:
            msg += f"🎯 *Take Profit*: ${tp_price:,.2f}\n"
        
        return msg
    
    def format_pause_response(self, success: bool, message: str = "") -> str:
        """Format response for /pause command."""
        if success:
            return "⏸️ *Strategy Paused*\n\nTrading has been paused. No new orders will be placed.\nUse /resume to continue trading."
        else:
            return f"❌ *Failed to Pause*\n\n{message}"
    
    def format_resume_response(self, success: bool, message: str = "") -> str:
        """Format response for /resume command."""
        if success:
            return "▶️ *Strategy Resumed*\n\nTrading has been resumed. Strategy is now active."
        else:
            return f"❌ *Failed to Resume*\n\n{message}"
    
    def format_help_response(self) -> str:
        """Format help message with available commands."""
        msg = "🤖 *Available Commands*\n\n"
        msg += "*Query Commands*:\n"
        msg += "• `/status` - View strategy status\n"
        msg += "• `/position` - View current position\n"
        msg += "• `/help` - Show this help message\n\n"
        msg += "*Control Commands*:\n"
        msg += "• `/pause` - Pause trading (no new orders)\n"
        msg += "• `/resume` - Resume trading\n\n"
        msg += "💡 _Commands are case-insensitive_\n"
        return msg


# Convenience function for quick testing
async def test_telegram_bot(token: str, chat_id: str) -> bool:
    """
    Quick test function for Telegram bot.
    
    Parameters
    ----------
    token : str
        Bot token from @BotFather
    chat_id : str
        Chat ID to send test message to
    
    Returns
    -------
    bool
        True if test successful
    """
    try:
        bot = TelegramBot(token=token, chat_id=chat_id)
        
        # Test connection
        if not await bot.test_connection():
            return False
        
        # Send test message
        success = await bot.send_message(
            "🧪 *Test Message*\n\n"
            "Telegram bot is working correctly!\n"
            "Ready to send trading notifications."
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """
    Standalone test mode.
    
    Usage:
        python telegram_bot.py <token> <chat_id>
    """
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python telegram_bot.py <token> <chat_id>")
        sys.exit(1)
    
    token = sys.argv[1]
    chat_id = sys.argv[2]
    
    # Run test
    result = asyncio.run(test_telegram_bot(token, chat_id))
    
    if result:
        print("✅ Test successful!")
        sys.exit(0)
    else:
        print("❌ Test failed!")
        sys.exit(1)

