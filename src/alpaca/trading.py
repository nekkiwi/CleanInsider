import os
import pandas as pd

try:
    import alpaca_trade_api as alpaca
except Exception:
    alpaca = None


def execute_trades(signals_df: pd.DataFrame, notional_per_trade: float = 1000.0) -> int:
    if signals_df is None or signals_df.empty:
        return 0
    if alpaca is None:
        return 0
    api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('ALPACA_API_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    if not api_key or not api_secret:
        return 0
    client = alpaca.REST(api_key, api_secret, base_url=base_url)
    placed = 0
    for _, row in signals_df.iterrows():
        symbol = row.get('Ticker')
        if not symbol:
            continue
        price = float(row.get('Price') or 0) or 0.0
        try:
            if price > 0:
                qty = max(1, int(notional_per_trade // price))
                client.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
            else:
                client.submit_order(symbol=symbol, notional=notional_per_trade, side='buy', type='market', time_in_force='day')
            placed += 1
        except Exception:
            continue
    return placed


