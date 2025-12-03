# src/alpaca/trading_client.py
"""
Alpaca trading client for paper and live trading.
"""

import datetime
from typing import Dict, List, Optional

import pandas as pd

from src import config

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderStatus,
        QueryOrderStatus
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
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
        self,
        api_key: str = None,
        secret_key: str = None,
        paper_mode: bool = None
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
            print("[WARN] Alpaca API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
            self.client = None
            return
        
        try:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_mode
            )
            # Initialize data client for quotes/spreads
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
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
    
    def get_spreads(self, symbols: List[str], default_spread: float = 0.005) -> Dict[str, float]:
        """
        Get real-time bid-ask spreads from Alpaca.
        
        Args:
            symbols: List of stock tickers
            default_spread: Default spread (0.5%) if quote unavailable
            
        Returns:
            Dict of {ticker: spread_as_fraction}
        """
        if not self.client or not hasattr(self, 'data_client') or not self.data_client:
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
                if quote and quote.ask_price and quote.bid_price and quote.ask_price > 0:
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
        if not self.client or not hasattr(self, 'data_client') or not self.data_client:
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
    
    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str = "buy"
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
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_request)
            
            return self._order_to_dict(order)
        except Exception as e:
            print(f"[ERROR] Failed to place market order for {symbol}: {e}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        side: str = "buy"
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
                limit_price=round(limit_price, 2)
            )
            
            order = self.client.submit_order(order_request)
            
            return self._order_to_dict(order)
        except Exception as e:
            print(f"[ERROR] Failed to place limit order for {symbol}: {e}")
            return None
    
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
        self,
        limit: int = 100,
        after: datetime.datetime = None
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
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit
            )
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
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        }
    
    def execute_signals(
        self,
        signals_df: pd.DataFrame,
        use_limit_orders: bool = True,
        limit_buffer: float = 0.001
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
                print(f"[ORDER] {ticker}: {shares} shares @ {order.get('limit_price', 'market')}")
        
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

