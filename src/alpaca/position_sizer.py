# src/alpaca/position_sizer.py
"""
Position sizing and risk management for trading.
"""

from typing import Dict

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
        max_spread_cost: float = None,
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
        max_size: float = 1.0,
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
        reference_spread: float = 0.005,
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
        current_exposure: float = 0.0,
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

    def calculate_shares(self, dollar_sizes: pd.Series, prices: pd.Series) -> pd.Series:
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

    def apply_kelly_cap(self, dollar_sizes: pd.Series, *args, **kwargs) -> pd.Series:
        """
        Fractional-Kelly cap on dollar allocations.

        NO-OP for now. A proper Kelly fraction needs realized win-rate / payoff
        statistics, which only become available once the trade ledger exists
        (Stage 5). It is wired as a pass-through in the vol_target path so the
        composition is ready: when the ledger lands, this will scale allocations
        by min(estimated_kelly, config.KELLY_FRACTION_CAP). Until then it returns
        its input unchanged.

        Args:
            dollar_sizes: Per-symbol dollar allocations.

        Returns:
            The same Series, unchanged.
        """
        return dollar_sizes

    def calculate_vol_target_sizes(
        self,
        predicted_returns: pd.Series,
        prices: pd.Series,
        atr_map: Dict[str, float],
        strategy_sl: float,
        portfolio_value: float,
        current_exposure_dollars: float = 0.0,
        spreads: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """
        Volatility-targeted position sizing.

        Composition (each factor has a distinct job):
        - ATR (risk scalar): a fixed dollar risk budget per name is divided by a
          volatility-derived stop distance, so a fixed dollar loss is taken at
          the stop regardless of the name's volatility. risk_budget =
          portfolio_value * RISK_PER_TRADE_PCT; stop_distance = ATR_STOP_MULT *
          ATR; shares = risk_budget / stop_distance; base_dollar = shares*price.
        - min-max conviction (signal/ranking): the SAME calculate_base_sizes
          mapping of predicted_returns into [0.25, 1.0] used by the legacy path
          and by training, so the relative ranking/scaling of names is identical
          to the conviction signal — only the dollar translation changes here.
        - spread haircut (cost): the existing apply_spread_haircut logic
          (reference_spread / half_spread, capped at 1, zeroed above
          MAX_SPREAD_COST) when spreads are provided.

        dollar_i = base_dollar_i * conviction_i * spread_haircut_i, then capped at
        MAX_POSITION_SIZE * portfolio_value per name. Finally, if
        current_exposure_dollars + sum(dollars) exceeds TARGET_NET_EXPOSURE *
        portfolio_value, all allocations are scaled down proportionally so total
        net exposure equals exactly TARGET_NET_EXPOSURE * portfolio_value.

        ATR-MISSING FALLBACK: when a symbol has no ATR (omitted by get_atr) OR a
        non-positive ATR, the stop distance falls back to a percentage stop
        abs(strategy_sl) * price. This ties the stop distance to the bracket SL
        (the same stop the eventual bracket order will use), and guarantees
        stop_distance > 0 so shares are always finite (never inf/NaN).

        Args:
            predicted_returns: Per-symbol predicted returns (Series indexed by symbol).
            prices: Per-symbol prices (Series indexed by symbol).
            atr_map: Dict of {symbol: atr}; missing symbols use the pct-stop fallback.
            strategy_sl: Strategy stop-loss as a (negative) fraction, e.g. -0.05.
            portfolio_value: Total portfolio value in dollars.
            current_exposure_dollars: Existing invested dollars (Stage 5 will
                source this from the ledger / Alpaca positions).
            spreads: Optional Series/Dict of spread estimates for the haircut.

        Returns:
            Dict with "dollars" (Series) and "shares" (Series), indexed by symbol.
        """
        if predicted_returns.empty:
            return {
                "dollars": pd.Series(dtype=float),
                "shares": pd.Series(dtype=int),
            }

        index = predicted_returns.index
        prices = prices.reindex(index)

        risk_budget = portfolio_value * config.RISK_PER_TRADE_PCT

        # Per-name stop distance: ATR-based when available & positive, else a
        # pct stop tied to the bracket SL. Always > 0 -> finite shares.
        stop_distance = pd.Series(index=index, dtype=float)
        for sym in index:
            price = prices.get(sym)
            atr = atr_map.get(sym)
            if atr is not None and atr > 0:
                sd = config.ATR_STOP_MULT * atr
            else:
                sd = abs(strategy_sl) * (price if pd.notna(price) else 0.0)
            # Guard: never allow a zero/NaN stop distance (would give inf shares).
            if not (sd > 0):
                # last-resort tiny floor relative to price (or 1 cent)
                sd = max(abs(price) * 1e-4, 0.01) if pd.notna(price) else 0.01
            stop_distance[sym] = sd

        base_shares = risk_budget / stop_distance
        base_dollars = base_shares * prices

        # Conviction multiplier: reuse the min-max mapping for signal parity.
        conviction = self.calculate_base_sizes(predicted_returns)
        conviction = conviction.reindex(index)

        # Spread haircut (cost): reuse the existing logic.
        if spreads is not None:
            if isinstance(spreads, dict):
                spreads = pd.Series(spreads)
            ones = pd.Series(1.0, index=index)
            haircut = self.apply_spread_haircut(ones, spreads)
            haircut = haircut.reindex(index).fillna(1.0)
        else:
            haircut = pd.Series(1.0, index=index)

        dollars = base_dollars * conviction * haircut

        # Per-name cap at MAX_POSITION_SIZE * portfolio_value.
        per_name_cap = self.max_position_size * portfolio_value
        dollars = dollars.clip(upper=per_name_cap)

        # Kelly cap (no-op pass-through for now; ready for Stage 5).
        dollars = self.apply_kelly_cap(dollars)

        # Batch scaling to TARGET_NET_EXPOSURE, respecting current exposure.
        max_new = (
            config.TARGET_NET_EXPOSURE * portfolio_value - current_exposure_dollars
        )
        max_new = max(0.0, max_new)
        total = dollars.sum()
        if total > max_new and total > 0:
            dollars = dollars * (max_new / total)

        # Shares (floor). Guard against zero/NaN prices.
        safe_prices = prices.replace(0, pd.NA)
        shares = (dollars / safe_prices).fillna(0)
        shares = shares.replace([float("inf"), float("-inf")], 0).astype(int)

        return {"dollars": dollars, "shares": shares}

    def size_positions(
        self,
        signals_df: pd.DataFrame,
        portfolio_value: float,
        current_positions: Dict[str, float] = None,
        spreads: pd.Series = None,
        method: str = None,
    ) -> pd.DataFrame:
        """
        Full pipeline to size positions from model signals.

        Args:
            signals_df: DataFrame with Ticker, predicted_return, position_size, Price
            portfolio_value: Total portfolio value
            current_positions: Dict of {ticker: current_value}
            spreads: Optional Series or Dict of spread estimates
            method: Sizing method — "vol_target" (ATR risk-budget) or "minmax"
                (legacy conviction-fraction). Defaults to config.SIZING_METHOD.

        Returns:
            DataFrame with final position sizing
        """
        if method is None:
            method = config.SIZING_METHOD

        if signals_df.empty:
            return pd.DataFrame()

        if method == "vol_target":
            return self._size_positions_vol_target(
                signals_df, portfolio_value, current_positions, spreads
            )

        return self._size_positions_minmax(
            signals_df, portfolio_value, current_positions, spreads
        )

    def _size_positions_vol_target(
        self,
        signals_df: pd.DataFrame,
        portfolio_value: float,
        current_positions: Dict[str, float] = None,
        spreads: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Build the sized-positions DataFrame using calculate_vol_target_sizes.

        Expects signals_df to carry Ticker, predicted_return, Price, and an
        optional per-row "atr" column (from AlpacaTradingClient.get_atr) plus the
        strategy stop-loss either as a "strategy_sl" column or via the predicted
        returns' own scale. Missing ATR falls back to a pct stop (see
        calculate_vol_target_sizes).
        """
        df = signals_df.copy()
        current_exposure = sum((current_positions or {}).values())

        idx = df.set_index("Ticker")

        if "predicted_return" in idx.columns:
            predicted_returns = idx["predicted_return"]
        elif "position_size" in idx.columns:
            predicted_returns = idx["position_size"]
        else:
            predicted_returns = pd.Series(0.5, index=idx.index)

        prices = idx["Price"] if "Price" in idx.columns else pd.Series(dtype=float)

        # ATR map from an "atr" column if present (omit NaN/<=0 -> fallback).
        atr_map: Dict[str, float] = {}
        if "atr" in idx.columns:
            for t, v in idx["atr"].items():
                if pd.notna(v) and v > 0:
                    atr_map[t] = float(v)

        # Strategy SL: per-row column, else default strategy's SL.
        if "strategy_sl" in idx.columns:
            strategy_sl = float(idx["strategy_sl"].iloc[0])
        else:
            strategy_sl = config.DEFAULT_STRATEGY[2]

        if isinstance(spreads, dict):
            spreads = pd.Series(spreads)

        sized = self.calculate_vol_target_sizes(
            predicted_returns,
            prices,
            atr_map,
            strategy_sl=strategy_sl,
            portfolio_value=portfolio_value,
            current_exposure_dollars=current_exposure,
            spreads=spreads,
        )
        dollar_sizes = sized["dollars"]
        shares = sized["shares"]

        cols = ["Ticker"]
        if "Filing Date" in df.columns:
            cols.append("Filing Date")
        result = df[cols].copy()

        result["dollar_size"] = dollar_sizes.reindex(df["Ticker"]).values
        result["shares"] = shares.reindex(df["Ticker"]).values
        if "Price" in df.columns:
            result["price"] = df["Price"].values
        if "predicted_return" in df.columns:
            result["predicted_return"] = df["predicted_return"].values
        if "confidence" in df.columns:
            result["confidence"] = df["confidence"].values

        # Drop below-minimum and zero-size positions.
        result.loc[result["dollar_size"] < self.min_position_dollars, "dollar_size"] = (
            0.0
        )
        result = result[result["dollar_size"] > 0].copy()

        return result.sort_values("dollar_size", ascending=False)

    def _size_positions_minmax(
        self,
        signals_df: pd.DataFrame,
        portfolio_value: float,
        current_positions: Dict[str, float] = None,
        spreads: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Legacy conviction-fraction sizing (preserved exactly for parity).

        Min-max conviction -> spread haircut -> fraction of MAX_POSITION_SIZE ->
        exposure cap. This is the pre-Stage-4 size_positions body unchanged.
        """
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
            base_sizes, portfolio_value, current_exposure
        )

        prices = (
            df.set_index("Ticker")["Price"] if "Price" in df.columns else pd.Series()
        )

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
        result["shares"] = (
            shares.reindex(df["Ticker"]).values if not shares.empty else 0
        )
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
