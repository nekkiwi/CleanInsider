# src/alpaca/position_sizer.py
"""
Position sizing and risk management for trading.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config


class PositionSizer:
    """
    Calculates position sizes based on model predictions and risk constraints.
    
    Implements the same sizing logic used in training:
    - Min-max scaling of predicted returns to [0.25, 1.0]
    - Spread-based size haircut
    - Hard position and exposure limits
    """
    
    def __init__(
        self,
        max_position_size: float = None,
        max_total_exposure: float = None,
        min_position_dollars: float = None,
        max_spread_cost: float = None
    ):
        """
        Initialize position sizer with risk limits.
        
        Args:
            max_position_size: Max fraction of portfolio per position
            max_total_exposure: Max total portfolio fraction invested
            min_position_dollars: Minimum position size in dollars
            max_spread_cost: Maximum spread cost to allow trading
        """
        self.max_position_size = max_position_size or config.MAX_POSITION_SIZE
        self.max_total_exposure = max_total_exposure or config.MAX_TOTAL_EXPOSURE
        self.min_position_dollars = min_position_dollars or config.MIN_POSITION_DOLLARS
        self.max_spread_cost = max_spread_cost or config.MAX_SPREAD_COST
    
    def calculate_base_sizes(
        self,
        predicted_returns: pd.Series,
        min_size: float = 0.25,
        max_size: float = 1.0
    ) -> pd.Series:
        """
        Scale predicted returns to position sizes using min-max scaling.
        
        Args:
            predicted_returns: Series of predicted returns from regressor
            min_size: Minimum position size (fraction of max allocation)
            max_size: Maximum position size (fraction of max allocation)
            
        Returns:
            Series of position sizes in [min_size, max_size]
        """
        if predicted_returns.empty:
            return pd.Series(dtype=float)
        
        min_pred = predicted_returns.min()
        max_pred = predicted_returns.max()
        
        if max_pred == min_pred:
            return pd.Series((min_size + max_size) / 2, index=predicted_returns.index)
        
        scaled = (predicted_returns - min_pred) / (max_pred - min_pred)
        position_sizes = min_size + scaled * (max_size - min_size)
        
        return position_sizes
    
    def apply_spread_haircut(
        self,
        position_sizes: pd.Series,
        spreads: pd.Series,
        reference_spread: float = 0.005
    ) -> pd.Series:
        """
        Apply spread-based haircut to position sizes.
        
        Reduces position size proportionally to spread cost.
        
        Args:
            position_sizes: Base position sizes
            spreads: Estimated bid-ask spreads
            reference_spread: Reference spread for haircut calculation (0.5%)
            
        Returns:
            Adjusted position sizes
        """
        if position_sizes.empty or spreads.empty:
            return position_sizes
        
        # Align indices
        aligned_spreads = spreads.reindex(position_sizes.index).fillna(0.005)
        
        # Half spread is the one-way cost
        half_spreads = aligned_spreads * 0.5
        
        # Haircut: scale position by reference_spread / actual_spread, cap at 1
        haircut = (reference_spread / half_spreads).clip(upper=1.0)
        
        # Zero out positions with excessive spread costs
        high_cost_mask = half_spreads > self.max_spread_cost
        haircut.loc[high_cost_mask] = 0.0
        
        return position_sizes * haircut
    
    def calculate_dollar_sizes(
        self,
        position_sizes: pd.Series,
        portfolio_value: float,
        current_exposure: float = 0.0
    ) -> pd.Series:
        """
        Convert fractional position sizes to dollar amounts.
        
        Args:
            position_sizes: Fractional position sizes (0-1)
            portfolio_value: Total portfolio value in dollars
            current_exposure: Current invested amount in dollars
            
        Returns:
            Series of position sizes in dollars
        """
        if position_sizes.empty:
            return pd.Series(dtype=float)
        
        # Available capital considering exposure limit
        max_new_investment = (
            self.max_total_exposure * portfolio_value - current_exposure
        )
        max_new_investment = max(0, max_new_investment)
        
        # Cap individual positions
        max_single_position = self.max_position_size * portfolio_value
        
        # Calculate dollar sizes
        dollar_sizes = position_sizes * max_single_position
        
        # Ensure we don't exceed available capital
        if dollar_sizes.sum() > max_new_investment:
            scale_factor = max_new_investment / dollar_sizes.sum()
            dollar_sizes = dollar_sizes * scale_factor
        
        # Remove positions below minimum
        dollar_sizes[dollar_sizes < self.min_position_dollars] = 0.0
        
        return dollar_sizes
    
    def calculate_shares(
        self,
        dollar_sizes: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """
        Convert dollar sizes to number of shares.
        
        Args:
            dollar_sizes: Position sizes in dollars
            prices: Current stock prices
            
        Returns:
            Series of share counts (whole numbers)
        """
        if dollar_sizes.empty or prices.empty:
            return pd.Series(dtype=int)
        
        aligned_prices = prices.reindex(dollar_sizes.index)
        
        # Calculate shares, rounding down
        shares = (dollar_sizes / aligned_prices).fillna(0).astype(int)
        
        return shares
    
    def size_positions(
        self,
        signals_df: pd.DataFrame,
        portfolio_value: float,
        current_positions: Dict[str, float] = None,
        spreads: pd.Series = None
    ) -> pd.DataFrame:
        """
        Full pipeline to size positions from model signals.
        
        Args:
            signals_df: DataFrame with Ticker, predicted_return, position_size, Price
            portfolio_value: Total portfolio value
            current_positions: Dict of {ticker: current_value}
            spreads: Optional Series or Dict of spread estimates
            
        Returns:
            DataFrame with final position sizing
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        df = signals_df.copy()
        
        # Calculate current exposure
        current_exposure = sum((current_positions or {}).values())
        
        # Get base sizes from model
        if "position_size" in df.columns:
            original_sizes = df.set_index("Ticker")["position_size"]
        elif "predicted_return" in df.columns:
            original_sizes = self.calculate_base_sizes(
                df.set_index("Ticker")["predicted_return"]
            )
        else:
            original_sizes = pd.Series(0.5, index=df["Ticker"])
        
        # Apply spread haircut if spreads provided
        base_sizes = original_sizes.copy()
        spread_haircuts = pd.Series(1.0, index=df["Ticker"])
        
        if spreads is not None:
            # Convert dict to Series if needed
            if isinstance(spreads, dict):
                spreads = pd.Series(spreads)
            base_sizes = self.apply_spread_haircut(original_sizes, spreads)
            # Calculate haircut ratios
            for t in df["Ticker"]:
                orig = original_sizes.get(t, 1.0)
                final = base_sizes.get(t, 0.0)
                spread_haircuts[t] = final / max(orig, 0.001) if orig > 0 else 0
        
        # Calculate dollar and share sizes
        dollar_sizes = self.calculate_dollar_sizes(
            base_sizes,
            portfolio_value,
            current_exposure
        )
        
        prices = df.set_index("Ticker")["Price"] if "Price" in df.columns else pd.Series()
        
        if not prices.empty:
            shares = self.calculate_shares(dollar_sizes, prices)
        else:
            shares = pd.Series(dtype=int)
        
        # Build result dataframe
        cols = ["Ticker"]
        if "Filing Date" in df.columns:
            cols.append("Filing Date")
        result = df[cols].copy()
        
        result["base_size"] = base_sizes.reindex(df["Ticker"]).values
        result["dollar_size"] = dollar_sizes.reindex(df["Ticker"]).values
        result["shares"] = shares.reindex(df["Ticker"]).values if not shares.empty else 0
        result["spread_haircut"] = spread_haircuts.reindex(df["Ticker"]).values
        
        if "Price" in df.columns:
            result["price"] = df["Price"].values
        
        if "predicted_return" in df.columns:
            result["predicted_return"] = df["predicted_return"].values
        
        if "confidence" in df.columns:
            result["confidence"] = df["confidence"].values
        
        # Filter out zero-size positions
        result = result[result["dollar_size"] > 0].copy()
        
        return result.sort_values("dollar_size", ascending=False)


