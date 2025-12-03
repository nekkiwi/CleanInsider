# src/alpaca/live_features.py
"""
Live feature generation for real-time inference.
"""

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.scrapers.feature_scraper.scrape_openinsider import scrape_openinsider
from src.scrapers.feature_scraper.load_annual_statements import generate_annual_statements
from src.scrapers.feature_scraper.load_technical_indicators import generate_technical_indicators
from src.scrapers.feature_scraper.load_macro_features import generate_macro_features
# Spread calculation is done inline since the module function is designed for batch processing


class LiveFeatureGenerator:
    """
    Generates features for live trading by scraping recent insider purchases
    and generating the same features used in training.
    """
    
    def __init__(
        self,
        num_weeks: int = None,
        stooq_path: Path = None,
        edgar_path: Path = None
    ):
        """
        Initialize the live feature generator.
        
        Args:
            num_weeks: Number of weeks of insider data to scrape
            stooq_path: Path to Stooq database
            edgar_path: Path to SEC EDGAR parquet files
        """
        self.num_weeks = num_weeks or config.LIVE_SCRAPE_WEEKS
        self.stooq_path = stooq_path or config.STOOQ_DATABASE_PATH
        self.edgar_path = edgar_path or config.EDGAR_DOWNLOAD_PATH
        
    def scrape_recent_insider_data(self) -> pd.DataFrame:
        """
        Scrape recent insider purchase data from OpenInsider.
        
        Returns:
            DataFrame with insider trading events
        """
        print(f"[INFO] Scraping {self.num_weeks} weeks of insider data...")
        insider_df = scrape_openinsider(num_weeks=self.num_weeks)
        
        if insider_df.empty:
            print("[WARN] No insider data scraped")
            return pd.DataFrame()
        
        insider_df["Filing Date"] = pd.to_datetime(insider_df["Filing Date"])
        print(f"[INFO] Scraped {len(insider_df)} insider events")
        
        return insider_df
    
    def generate_features_for_events(
        self,
        events_df: pd.DataFrame,
        include_fundamentals: bool = True,
        include_technicals: bool = True,
        include_macro: bool = True
    ) -> pd.DataFrame:
        """
        Generate full feature set for given insider events.
        
        Args:
            events_df: DataFrame with Ticker, Filing Date, and base insider features
            include_fundamentals: Whether to include SEC fundamental data
            include_technicals: Whether to include technical indicators
            include_macro: Whether to include macro features
            
        Returns:
            DataFrame with all features merged
        """
        if events_df.empty:
            return pd.DataFrame()
        
        merged_df = events_df.copy()
        
        # Generate and merge annual statements (fundamentals)
        if include_fundamentals:
            try:
                print("[INFO] Generating fundamental features...")
                annual_path = Path(config.FEATURES_OUTPUT_PATH) / "components" / "temp_annual.parquet"
                generate_annual_statements(
                    events_df, 
                    annual_path,
                    sec_parquet_dir=self.edgar_path,
                    request_header=config.REQUESTS_HEADER
                )
                
                if annual_path.exists():
                    annual_df = pd.read_parquet(annual_path)
                    annual_df["Filing Date"] = pd.to_datetime(annual_df["Filing Date"])
                    merged_df = pd.merge(
                        merged_df, 
                        annual_df, 
                        on=["Ticker", "Filing Date"], 
                        how="left"
                    )
                    annual_path.unlink()  # Clean up temp file
            except Exception as e:
                print(f"[WARN] Failed to generate fundamental features: {e}")
        
        # Generate and merge technical indicators
        if include_technicals:
            try:
                print("[INFO] Generating technical features...")
                tech_path = Path(config.FEATURES_OUTPUT_PATH) / "components" / "temp_tech.parquet"
                generate_technical_indicators(
                    events_df,
                    self.stooq_path,
                    tech_path
                )
                
                if tech_path.exists():
                    tech_df = pd.read_parquet(tech_path)
                    tech_df["Filing Date"] = pd.to_datetime(tech_df["Filing Date"])
                    merged_df = pd.merge(
                        merged_df,
                        tech_df,
                        on=["Ticker", "Filing Date"],
                        how="left"
                    )
                    tech_path.unlink()
            except Exception as e:
                print(f"[WARN] Failed to generate technical features: {e}")
        
        # Generate and merge macro features
        if include_macro:
            try:
                print("[INFO] Generating macro features...")
                macro_path = Path(config.FEATURES_OUTPUT_PATH) / "components" / "temp_macro.parquet"
                generate_macro_features(
                    events_df,
                    self.stooq_path,
                    macro_path
                )
                
                if macro_path.exists():
                    macro_df = pd.read_parquet(macro_path)
                    macro_df["Filing Date"] = pd.to_datetime(macro_df["Filing Date"])
                    merged_df = pd.merge(
                        merged_df,
                        macro_df,
                        on="Filing Date",
                        how="left"
                    )
                    macro_path.unlink()
            except Exception as e:
                print(f"[WARN] Failed to generate macro features: {e}")
        
        # Add corwin_schultz_spread column with default value
        # In live inference, actual spread haircuts are applied using live Alpaca quotes
        # This placeholder ensures feature count matches trained models
        if "corwin_schultz_spread" not in merged_df.columns:
            merged_df["corwin_schultz_spread"] = 0.005  # 50 bps default placeholder
        
        print(f"[INFO] Generated features for {len(merged_df)} events with {merged_df.shape[1]} columns")
        return merged_df
    
    def generate_live_features(
        self,
        min_date: datetime.datetime = None
    ) -> pd.DataFrame:
        """
        Full pipeline to scrape and generate features for live inference.
        
        Args:
            min_date: Only include events after this date (default: 7 days ago)
            
        Returns:
            DataFrame ready for model inference
        """
        if min_date is None:
            min_date = datetime.datetime.now() - datetime.timedelta(days=7)
        
        # Scrape recent insider data
        insider_df = self.scrape_recent_insider_data()
        
        if insider_df.empty:
            return pd.DataFrame()
        
        # Filter to recent events only
        insider_df = insider_df[insider_df["Filing Date"] >= min_date].copy()
        
        if insider_df.empty:
            print(f"[INFO] No insider events after {min_date.date()}")
            return pd.DataFrame()
        
        print(f"[INFO] Processing {len(insider_df)} events after {min_date.date()}")
        
        # Generate all features
        features_df = self.generate_features_for_events(insider_df)
        
        return features_df
    
    def filter_to_tradable(
        self,
        features_df: pd.DataFrame,
        min_price: float = 1.0,
        min_volume: float = 100000
    ) -> pd.DataFrame:
        """
        Filter features to only include tradable stocks.
        
        Args:
            features_df: Full feature dataframe
            min_price: Minimum stock price
            min_volume: Minimum average volume
            
        Returns:
            Filtered dataframe
        """
        if features_df.empty:
            return features_df
        
        df = features_df.copy()
        
        # Filter by price if available
        if "Price" in df.columns:
            df = df[df["Price"] >= min_price]
        
        # Filter by volume if available
        if "Volume" in df.columns:
            df = df[df["Volume"] >= min_volume]
        
        print(f"[INFO] Filtered to {len(df)} tradable events")
        return df


