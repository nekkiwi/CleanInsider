# src/alpaca/trading_client.py
"""
Alpaca trading client for paper and live trading.
"""

import datetime
from typing import Dict, List, Optional

import pandas as pd

from src import config

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import (
        OrderClass,
        OrderSide,
        QueryOrderStatus,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARN] alpaca-py not installed. Trading functionality disabled.")


class AlpacaTradingClient:
    """
    Wrapper around Alpaca Trading API for paper and live trading.

    Supports:
    - Account info and positions
    - Market and limit orders
    - Order management (cancel, status)
    - Paper/live mode switching
    """

    def __init__(
        self, api_key: str = None, secret_key: str = None, paper_mode: bool = None
    ):
        """
        Initialize Alpaca trading client.

        Args:
            api_key: Alpaca API key (defaults to env var)
            secret_key: Alpaca secret key (defaults to env var)
            paper_mode: Use paper trading (defaults to config)
        """
        self.api_key = api_key or config.ALPACA_API_KEY
        self.secret_key = secret_key or config.ALPACA_SECRET_KEY
        self.paper_mode = paper_mode if paper_mode is not None else config.PAPER_MODE

        if not ALPACA_AVAILABLE:
            self.client = None
            return

        if not self.api_key or not self.secret_key:
            print(
                "[WARN] Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars."
            )
            self.client = None
            return

        try:
            self.client = TradingClient(
                api_key=self.api_key, secret_key=self.secret_key, paper=self.paper_mode
            )
            # Initialize data client for quotes/spreads
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.secret_key
            )
            mode = "paper" if self.paper_mode else "live"
            print(f"[INFO] Alpaca client initialized in {mode} mode")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Alpaca client: {e}")
            self.client = None
            self.data_client = None

    def is_connected(self) -> bool:
        """Check if client is connected and functional."""
        return self.client is not None

    def get_account(self) -> Optional[Dict]:
        """
        Get account information.

        Returns:
            Dict with account info or None if error
        """
        if not self.client:
            return None

        try:
            account = self.client.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "currency": account.currency,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "day_trade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            print(f"[ERROR] Failed to get account: {e}")
            return None

    def get_positions(self) -> Dict[str, Dict]:
        """
        Get all current positions.

        Returns:
            Dict of {ticker: position_info}
        """
        if not self.client:
            return {}

        try:
            positions = self.client.get_all_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "side": pos.side,
                }
            return result
        except Exception as e:
            print(f"[ERROR] Failed to get positions: {e}")
            return {}

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Position info dict or None
        """
        positions = self.get_positions()
        return positions.get(symbol)

    def get_spreads(
        self, symbols: List[str], default_spread: float = 0.005
    ) -> Dict[str, float]:
        """
        Get real-time bid-ask spreads from Alpaca.

        Args:
            symbols: List of stock tickers
            default_spread: Default spread (0.5%) if quote unavailable

        Returns:
            Dict of {ticker: spread_as_fraction}
        """
        if not self.client or not hasattr(self, "data_client") or not self.data_client:
            print("[WARN] Data client not available, using default spreads")
            return {s: default_spread for s in symbols}

        if not symbols:
            return {}

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

            spreads = {}
            for symbol in symbols:
                quote = quotes.get(symbol)
                if (
                    quote
                    and quote.ask_price
                    and quote.bid_price
                    and quote.ask_price > 0
                ):
                    # Calculate spread as fraction of mid price
                    mid = (quote.ask_price + quote.bid_price) / 2
                    spread = (quote.ask_price - quote.bid_price) / mid
                    spreads[symbol] = max(spread, 0.0001)  # Floor at 1 bp
                else:
                    spreads[symbol] = default_spread

            print(f"[INFO] Fetched live spreads for {len(spreads)} symbols")
            return spreads
        except Exception as e:
            print(f"[WARN] Failed to get spreads: {e}. Using defaults.")
            return {s: default_spread for s in symbols}

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices from Alpaca quotes.

        Args:
            symbols: List of stock tickers

        Returns:
            Dict of {ticker: mid_price}
        """
        if not self.client or not hasattr(self, "data_client") or not self.data_client:
            return {}

        if not symbols:
            return {}

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

            prices = {}
            for symbol in symbols:
                quote = quotes.get(symbol)
                if quote and quote.ask_price and quote.bid_price:
                    prices[symbol] = (quote.ask_price + quote.bid_price) / 2

            return prices
        except Exception as e:
            print(f"[WARN] Failed to get prices: {e}")
            return {}

    def get_atr(self, symbols: List[str], period: int = None) -> Dict[str, float]:
        """
        Compute the Average True Range (ATR) per symbol from recent daily bars.

        Fetches ~period+1 daily bars via the alpaca-py
        StockHistoricalDataClient.get_stock_bars and computes ATR as the mean of
        the true range over the lookback window. True range for bar i is
        max(high-low, |high-prev_close|, |low-prev_close|); the first bar has no
        previous close so it contributes no TR.

        Args:
            symbols: List of stock tickers.
            period: ATR lookback in bars (defaults to config.ATR_PERIOD).

        Returns:
            Dict of {symbol: atr}. Symbols with insufficient/missing data are
            OMITTED so the caller can apply its own fallback (e.g. a pct stop).
        """
        if period is None:
            period = config.ATR_PERIOD

        if not self.client or not getattr(self, "data_client", None):
            return {}

        if not symbols:
            return {}

        try:
            # Request a few extra bars so we reliably get `period` true ranges
            # even with holidays/missing sessions; we only use the most recent.
            request = StockBarsRequest(
                symbol_or_symbols=list(symbols),
                timeframe=TimeFrame.Day,
                limit=period + 5,
            )
            barset = self.data_client.get_stock_bars(request)
            data = getattr(barset, "data", barset)
        except Exception as e:  # noqa: BLE001 - network/SDK errors -> empty map
            print(f"[WARN] Failed to fetch bars for ATR: {e}. Using fallback.")
            return {}

        atr_map: Dict[str, float] = {}
        for symbol in symbols:
            bars = data.get(symbol) if hasattr(data, "get") else None
            if not bars or len(bars) < 2:
                # Need at least 2 bars to form one true range.
                continue

            # Use the most recent `period + 1` bars (need 1 extra for prev close).
            window = bars[-(period + 1) :]
            true_ranges = []
            for i in range(1, len(window)):
                high = float(window[i].high)
                low = float(window[i].low)
                prev_close = float(window[i - 1].close)
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close),
                )
                true_ranges.append(tr)

            if not true_ranges:
                continue

            atr = sum(true_ranges) / len(true_ranges)
            atr_map[symbol] = atr

        return atr_map

    def get_adv(self, symbols: List[str], period: int = 60) -> Dict[str, float]:
        """
        Compute live Average Dollar Volume (ADV) per symbol from recent daily bars.

        Mirrors get_atr's guarded bar-fetch. Fetches ~period daily bars via the
        alpaca-py StockHistoricalDataClient.get_stock_bars and returns the MEDIAN
        of Close*Volume over the window (median ignores one-off volume spikes that
        would inflate a mean). This is the live counterpart of the point-in-time
        ADV used to define the liquid training/backtest universe.

        Args:
            symbols: List of stock tickers.
            period: ADV lookback in daily bars (default 60, matching ADV_WINDOW).

        Returns:
            Dict of {symbol: adv}. Symbols with missing/insufficient bar data are
            OMITTED so the caller treats unknown liquidity as untradeable.
        """
        if not self.client or not getattr(self, "data_client", None):
            return {}

        if not symbols:
            return {}

        try:
            request = StockBarsRequest(
                symbol_or_symbols=list(symbols),
                timeframe=TimeFrame.Day,
                limit=period,
            )
            barset = self.data_client.get_stock_bars(request)
            data = getattr(barset, "data", barset)
        except Exception as e:  # noqa: BLE001 - network/SDK errors -> empty map
            print(f"[WARN] Failed to fetch bars for ADV: {e}. Using fallback.")
            return {}

        adv_map: Dict[str, float] = {}
        for symbol in symbols:
            bars = data.get(symbol) if hasattr(data, "get") else None
            if not bars:
                continue

            window = bars[-period:]
            dollar_vols = []
            for bar in window:
                close = getattr(bar, "close", None)
                volume = getattr(bar, "volume", None)
                if close is None or volume is None:
                    continue
                dollar_vols.append(float(close) * float(volume))

            if not dollar_vols:
                continue

            dollar_vols.sort()
            n = len(dollar_vols)
            mid = n // 2
            if n % 2 == 1:
                adv = dollar_vols[mid]
            else:
                adv = (dollar_vols[mid - 1] + dollar_vols[mid]) / 2.0
            adv_map[symbol] = adv

        return adv_map

    def place_market_order(
        self, symbol: str, qty: int, side: str = "buy"
    ) -> Optional[Dict]:
        """
        Place a market order.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'

        Returns:
            Order info dict or None if error
        """
        if not self.client:
            return None

        if qty <= 0:
            print(f"[WARN] Invalid quantity {qty} for {symbol}")
            return None

        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=TimeInForce.DAY
            )

            order = self.client.submit_order(order_request)

            return self._order_to_dict(order)
        except Exception as e:
            print(f"[ERROR] Failed to place market order for {symbol}: {e}")
            return None

    def place_limit_order(
        self, symbol: str, qty: int, limit_price: float, side: str = "buy"
    ) -> Optional[Dict]:
        """
        Place a limit order.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            limit_price: Limit price
            side: 'buy' or 'sell'

        Returns:
            Order info dict or None if error
        """
        if not self.client:
            return None

        if qty <= 0:
            print(f"[WARN] Invalid quantity {qty} for {symbol}")
            return None

        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )

            order = self.client.submit_order(order_request)

            return self._order_to_dict(order)
        except Exception as e:
            print(f"[ERROR] Failed to place limit order for {symbol}: {e}")
            return None

    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        entry_limit_price: float,
        tp_price: float,
        sl_price: float,
        side: str = "buy",
        client_order_id: str = None,
        sl_limit_price: float = None,
    ) -> Optional[Dict]:
        """
        Place a bracket order (entry + take-profit + stop-loss legs).

        Submits a single OrderClass.BRACKET limit entry with attached TP and SL
        child legs. Bracket legs require TimeInForce.GTC. All three prices are
        penny-rounded (2dp) before submission.

        VALIDATION: requires ``sl_price < entry_limit_price < tp_price`` strictly,
        evaluated AFTER rounding. Alpaca rejects the entire bracket if the legs
        are not on the correct side of the entry, so if the ordering is invalid
        (or collapses to equality after rounding) the order is SKIPPED: this logs
        a warning and returns None rather than submitting a doomed order.

        Args:
            symbol: Stock ticker.
            qty: Number of shares (must be > 0).
            entry_limit_price: Limit price for the entry leg.
            tp_price: Take-profit limit price (the profitable exit).
            sl_price: Stop-loss trigger price (the losing exit).
            side: 'buy' or 'sell' for the ENTRY leg (default 'buy').
            client_order_id: Deterministic client order id (idempotency key).
            sl_limit_price: Optional stop-loss limit price (stop-limit exit). When
                omitted the SL leg executes as a market order on trigger.

        Returns:
            Dict {entry_order_id, tp_leg_id, sl_leg_id, client_order_id} on
            success, or None if skipped/failed.
        """
        if not self.client:
            return None

        if qty <= 0:
            print(f"[WARN] Invalid quantity {qty} for {symbol}")
            return None

        # Penny-round all prices to 2dp (equities trade in cents).
        entry_r = round(float(entry_limit_price), 2)
        tp_r = round(float(tp_price), 2)
        sl_r = round(float(sl_price), 2)

        # VALIDATE sl < entry < tp strictly, AFTER rounding. Equality (collapse
        # after rounding) is also invalid -> skip.
        if not (sl_r < entry_r < tp_r):
            print(
                f"[WARN] Skipping bracket for {symbol}: invalid price ordering "
                f"after rounding (sl={sl_r}, entry={entry_r}, tp={tp_r}); "
                f"requires sl < entry < tp."
            )
            return None

        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            take_profit = TakeProfitRequest(limit_price=tp_r)
            sl_kwargs = {"stop_price": sl_r}
            if sl_limit_price is not None:
                sl_kwargs["limit_price"] = round(float(sl_limit_price), 2)
            stop_loss = StopLossRequest(**sl_kwargs)

            order_kwargs = dict(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                limit_price=entry_r,
                order_class=OrderClass.BRACKET,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
            if client_order_id:
                order_kwargs["client_order_id"] = client_order_id

            order_request = LimitOrderRequest(**order_kwargs)
            order = self.client.submit_order(order_request)

            return self._bracket_to_dict(order, client_order_id)
        except Exception as e:
            print(f"[ERROR] Failed to place bracket order for {symbol}: {e}")
            return None

    @staticmethod
    def _bracket_to_dict(order, client_order_id: str = None) -> Dict:
        """Extract entry + TP/SL leg ids from a submitted bracket order.

        The TP leg is the LIMIT child; the SL leg is the STOP / STOP_LIMIT child.
        Leg ids may be None if Alpaca has not yet materialised the children.
        """
        tp_leg_id = None
        sl_leg_id = None
        legs = getattr(order, "legs", None) or []
        for leg in legs:
            leg_type = str(getattr(leg, "type", "")).lower()
            if "stop" in leg_type:
                sl_leg_id = str(leg.id)
            elif "limit" in leg_type:
                tp_leg_id = str(leg.id)

        return {
            "entry_order_id": str(order.id),
            "tp_leg_id": tp_leg_id,
            "sl_leg_id": sl_leg_id,
            "client_order_id": client_order_id,
        }

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.client:
            return False

        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order info dict or None
        """
        if not self.client:
            return None

        try:
            order = self.client.get_order_by_id(order_id)
            return self._order_to_dict(order)
        except Exception as e:
            print(f"[ERROR] Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders.

        Returns:
            List of order info dicts
        """
        if not self.client:
            return []

        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(request)
            return [self._order_to_dict(order) for order in orders]
        except Exception as e:
            print(f"[ERROR] Failed to get open orders: {e}")
            return []

    def get_recent_orders(
        self, limit: int = 100, after: datetime.datetime = None
    ) -> List[Dict]:
        """
        Get recent orders.

        Args:
            limit: Max orders to return
            after: Only orders after this time

        Returns:
            List of order info dicts
        """
        if not self.client:
            return []

        try:
            request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            if after:
                request.after = after

            orders = self.client.get_orders(request)
            return [self._order_to_dict(order) for order in orders]
        except Exception as e:
            print(f"[ERROR] Failed to get recent orders: {e}")
            return []

    def _order_to_dict(self, order) -> Dict:
        """Convert Alpaca order object to dict."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else 0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status),
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "filled_avg_price": (
                float(order.filled_avg_price) if order.filled_avg_price else None
            ),
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }

    def execute_signals(
        self,
        signals_df: pd.DataFrame,
        use_limit_orders: bool = True,
        limit_buffer: float = 0.001,
    ) -> List[Dict]:
        """
        Execute trading signals.

        Args:
            signals_df: DataFrame with Ticker, shares, price columns
            use_limit_orders: Use limit orders instead of market
            limit_buffer: Buffer below mid for limit orders (0.1% default)

        Returns:
            List of order results
        """
        if not self.client:
            print("[WARN] Trading client not connected")
            return []

        if signals_df.empty:
            return []

        orders = []

        for _, row in signals_df.iterrows():
            ticker = row["Ticker"]
            shares = int(row.get("shares", 0))
            price = row.get("price") or row.get("Price")

            if shares <= 0:
                continue

            if use_limit_orders and price:
                # Set limit price slightly below current price
                limit_price = price * (1 - limit_buffer)
                order = self.place_limit_order(ticker, shares, limit_price, "buy")
            else:
                order = self.place_market_order(ticker, shares, "buy")

            if order:
                orders.append(order)
                print(
                    f"[ORDER] {ticker}: {shares} shares @ {order.get('limit_price', 'market')}"
                )

        return orders

    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Close entire position for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Order info dict or None
        """
        if not self.client:
            return None

        position = self.get_position(symbol)
        if not position:
            print(f"[INFO] No position to close for {symbol}")
            return None

        qty = int(abs(position["qty"]))
        return self.place_market_order(symbol, qty, "sell")

    def close_all_positions(self) -> List[Dict]:
        """
        Close all open positions.

        Returns:
            List of order results
        """
        if not self.client:
            return []

        orders = []
        positions = self.get_positions()

        for symbol in positions:
            order = self.close_position(symbol)
            if order:
                orders.append(order)

        return orders
