#!/usr/bin/env python3
"""
Configuration for Spectral Crisis Detection Experiment
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Main configuration"""
    
    # API
    api_key: str = field(default_factory=lambda: os.getenv("EODHD_API_KEY") or os.getenv("API_KEY"))
    
    # Data
    years: int = 15
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("results")
    
    # Walk-forward validation
    n_splits: int = 5
    train_years: float = 3
    test_months: int = 6
    gap_days: int = 10
    
    # Model parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 30
    rf_min_samples_split: int = 60
    random_seed: int = 42
    
    # Spectral parameters
    correlation_windows: tuple = (60, 120)
    ewm_halflife: int = 30
    graph_thresholds: tuple = (0.3, 0.5, 0.7)
    dynamics_lookbacks: tuple = (5, 10, 20)
    
    # Asset universe
    symbols: Dict[str, str] = field(default_factory=lambda: {
        'SPY.US': 'SP500', 'QQQ.US': 'Nasdaq100', 'IWM.US': 'Russell2000',
        'XLF.US': 'Financials', 'XLE.US': 'Energy', 'XLK.US': 'Technology',
        'XLV.US': 'Healthcare', 'XLU.US': 'Utilities', 'XLP.US': 'ConsumerStaples',
        'XLY.US': 'ConsumerDisc', 'XLI.US': 'Industrials', 'XLB.US': 'Materials',
        'XLRE.US': 'RealEstate', 'EFA.US': 'DevIntl', 'EEM.US': 'EmergingMkts',
        'VGK.US': 'Europe', 'EWJ.US': 'Japan', 'TLT.US': 'LongTreasury',
        'IEF.US': 'IntermTreasury', 'LQD.US': 'InvGradeCorp', 'HYG.US': 'HighYield',
        'GLD.US': 'Gold', 'USO.US': 'Oil', 'UUP.US': 'USDollar', 'VNQ.US': 'REITs',
    })
    
    def __post_init__(self):
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        if not self.api_key:
            raise ValueError("EODHD_API_KEY not found in environment!")