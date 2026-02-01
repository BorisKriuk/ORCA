#!/usr/bin/env python3
"""
Data Loading Utilities for Spectral Crisis Detection

Handles:
- API data fetching with caching
- Multi-asset price data alignment
- Returns computation
"""

import os
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from config import DataConfig

warnings.filterwarnings('ignore')


class EODHDLoader:
    """Load market data from EODHD API with caching"""
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, api_key: str, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.session = requests.Session()
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data for a single symbol with caching"""
        
        cache_file = symbol.replace('.', '_').replace('/', '_')
        cache_path = self.cache_dir / f"{cache_file}.pkl"
        
        # Try cache first
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    if len(df) > 100:
                        return df
            except Exception:
                pass
        
        # Fetch from API
        try:
            params = {
                'api_token': self.api_key,
                'fmt': 'json',
                'from': start_date,
                'to': end_date
            }
            
            url = f"{self.BASE_URL}/eod/{symbol}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Cache the result
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            return pd.DataFrame()


def load_multi_asset_data(config: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load diverse set of assets for correlation analysis.
    
    Returns:
        prices: DataFrame of adjusted close prices
        returns: DataFrame of daily returns
    """
    
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY") or os.getenv("API_KEY")
    
    if not api_key:
        raise ValueError("EODHD_API_KEY not found in environment!")
    
    loader = EODHDLoader(api_key, config.cache_dir)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=config.years * 365)).strftime('%Y-%m-%d')
    
    print(f"Loading {len(config.symbols)} assets...")
    
    data_dict = {}
    for symbol, name in config.symbols.items():
        df = loader.get_data(symbol, start_date, end_date)
        if not df.empty and 'adjusted_close' in df.columns:
            data_dict[name] = df['adjusted_close']
            print(f"  ✓ {name}: {len(df)} days")
        else:
            print(f"  ✗ {name}: failed")
    
    # Combine into single DataFrame
    prices = pd.DataFrame(data_dict)
    prices = prices.dropna(how='all')
    
    # Forward fill small gaps, then drop remaining NaN
    prices = prices.ffill(limit=5)
    prices = prices.dropna()
    
    # Compute returns
    returns = prices.pct_change().dropna()
    
    print(f"\nCombined dataset: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, returns


def compute_target_variable(
    prices: pd.DataFrame,
    horizon: int = 10,
    percentile_threshold: float = 0.9,
    lookback_window: int = 252
) -> pd.Series:
    """
    Compute the target variable: Extreme volatility in next N days.
    
    Target = 1 if realized volatility over next `horizon` days
    exceeds the `percentile_threshold` of the trailing `lookback_window`.
    
    Args:
        prices: Price DataFrame (uses first column or 'SP500' as market proxy)
        horizon: Prediction horizon in days
        percentile_threshold: Threshold percentile (0.9 = top 10%)
        lookback_window: Window for computing percentile threshold
    
    Returns:
        Binary target series
    """
    
    # Use SP500 as market proxy
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()
    
    # Realized volatility (annualized)
    realized_vol = returns.rolling(20).std() * np.sqrt(252)
    
    # Future volatility (what we're predicting)
    future_vol = realized_vol.shift(-horizon)
    
    # Dynamic threshold based on trailing percentile
    vol_threshold = realized_vol.rolling(lookback_window).quantile(percentile_threshold)
    
    # Target: 1 if future vol exceeds threshold
    target = (future_vol > vol_threshold).astype(int)
    target.name = f'vol_extreme_{horizon}d'
    
    return target