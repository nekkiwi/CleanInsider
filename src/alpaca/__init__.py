# src/alpaca/__init__.py
"""
Alpaca trading module for CleanInsider.

This module provides:
- inference: Load models and generate predictions
- live_features: Real-time feature scraping for new insider purchases
- trading_client: Alpaca API wrapper for paper/live trading
- position_sizer: Risk management and position sizing
- google_drive: Model sync and trade logging
"""

from .inference import EnsemblePredictor
from .live_features import LiveFeatureGenerator
from .trading_client import AlpacaTradingClient
from .position_sizer import PositionSizer
from .google_drive import GoogleDriveClient

__all__ = [
    "EnsemblePredictor",
    "LiveFeatureGenerator",
    "AlpacaTradingClient",
    "PositionSizer",
    "GoogleDriveClient",
]


