#!/usr/bin/env python3
"""
SOTA Benchmark Models for Crisis Prediction

Benchmarks:
1. Absorption Ratio (Kritzman et al., 2011)
2. Turbulence Index (Kritzman & Li, 2010)
3. GARCH(1,1) Volatility
4. HAR-RV (Corsi, 2009)
5. Random Forest with Traditional Features
6. Logistic Regression
"""

from typing import Dict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler

from config import Config


class BaseModel(ABC):
    """Base class for all models"""
    
    name: str = "BaseModel"
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# =============================================================================
# BENCHMARK FEATURE COMPUTATION
# =============================================================================

def compute_absorption_ratio(returns: pd.DataFrame, window: int = 60, n_top: int = 5) -> pd.Series:
    """Absorption Ratio - Kritzman et al. (2011)"""
    ar_series = pd.Series(index=returns.index, dtype=float)
    n_assets = len(returns.columns)
    n_top = min(n_top, max(1, n_assets // 5))
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        corr = window_returns.corr().values
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 1.0)
        
        try:
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)
            total = np.sum(eigenvalues)
            if total > 0:
                ar_series.iloc[i] = np.sum(eigenvalues[:n_top]) / total
        except Exception:
            pass
    
    return ar_series


def compute_turbulence(returns: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Turbulence Index - Kritzman & Li (2010)"""
    turb_series = pd.Series(index=returns.index, dtype=float)
    
    for i in range(lookback, len(returns)):
        historical = returns.iloc[i-lookback:i]
        current = returns.iloc[i].values
        
        mu = historical.mean().values
        cov = historical.cov().values
        
        try:
            cov_reg = cov + np.eye(len(cov)) * 1e-6
            cov_inv = np.linalg.inv(cov_reg)
            diff = current - mu
            turb_series.iloc[i] = diff @ cov_inv @ diff
        except Exception:
            pass
    
    return turb_series


def compute_garch_volatility(returns: pd.Series, alpha: float = 0.1, beta: float = 0.85) -> pd.Series:
    """GARCH(1,1) Volatility"""
    omega = (1 - alpha - beta) * returns.var()
    garch_vol = pd.Series(index=returns.index, dtype=float)
    var_t = returns.var()
    
    for i in range(1, len(returns)):
        r_prev = returns.iloc[i-1]
        var_t = omega + alpha * r_prev**2 + beta * var_t
        garch_vol.iloc[i] = np.sqrt(var_t) * np.sqrt(252)
    
    return garch_vol


def compute_har_features(returns: pd.Series) -> pd.DataFrame:
    """HAR-RV features - Corsi (2009)"""
    rv_d = returns.rolling(1).std() * np.sqrt(252)
    rv_w = returns.rolling(5).std() * np.sqrt(252)
    rv_m = returns.rolling(22).std() * np.sqrt(252)
    
    return pd.DataFrame({
        'rv_d': rv_d,
        'rv_w': rv_w,
        'rv_m': rv_m
    }, index=returns.index)


# =============================================================================
# BENCHMARK MODELS
# =============================================================================

class AbsorptionRatioModel(BaseModel):
    """Absorption Ratio based prediction"""
    name = "Absorption Ratio"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


class TurbulenceModel(BaseModel):
    """Turbulence Index based prediction"""
    name = "Turbulence Index"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


class GARCHModel(BaseModel):
    """GARCH based prediction"""
    name = "GARCH(1,1)"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


class HARRVModel(BaseModel):
    """HAR-RV based prediction"""
    name = "HAR-RV"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


class RandomForestModel(BaseModel):
    """Random Forest with traditional features"""
    name = "Random Forest"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            min_samples_split=self.config.rf_min_samples_split,
            class_weight='balanced_subsample',
            random_state=self.config.random_seed,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline"""
    name = "Logistic Regression"
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        self.model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        return self.model.predict_proba(X_scaled)[:, 1]


# =============================================================================
# PREPARE BENCHMARK FEATURES
# =============================================================================

def prepare_benchmark_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Prepare feature sets for each benchmark"""
    
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    market_returns = market.pct_change()
    
    print("Computing benchmark features...")
    
    features = {}
    
    # Absorption Ratio
    ar = compute_absorption_ratio(returns, window=60, n_top=5)
    features['absorption_ratio'] = pd.DataFrame({
        'ar': ar,
        'ar_ma5': ar.rolling(5).mean(),
        'ar_ma20': ar.rolling(20).mean(),
        'ar_zscore': (ar - ar.rolling(60).mean()) / (ar.rolling(60).std() + 1e-10),
        'ar_roc_5d': ar.pct_change(5),
    })
    print(f"  Absorption Ratio: {len(features['absorption_ratio'].columns)} features")
    
    # Turbulence
    turb = compute_turbulence(returns, lookback=252)
    features['turbulence'] = pd.DataFrame({
        'turb': turb,
        'turb_ma5': turb.rolling(5).mean(),
        'turb_ma20': turb.rolling(20).mean(),
        'turb_zscore': (turb - turb.rolling(60).mean()) / (turb.rolling(60).std() + 1e-10),
        'turb_roc_5d': turb.pct_change(5),
    })
    print(f"  Turbulence: {len(features['turbulence'].columns)} features")
    
    # GARCH
    garch = compute_garch_volatility(market_returns)
    features['garch'] = pd.DataFrame({
        'garch': garch,
        'garch_ma5': garch.rolling(5).mean(),
        'garch_zscore': (garch - garch.rolling(60).mean()) / (garch.rolling(60).std() + 1e-10),
        'garch_roc_5d': garch.pct_change(5),
    })
    print(f"  GARCH: {len(features['garch'].columns)} features")
    
    # HAR-RV
    har = compute_har_features(market_returns)
    har['rv_ratio'] = har['rv_d'] / (har['rv_m'] + 1e-10)
    features['har_rv'] = har
    print(f"  HAR-RV: {len(features['har_rv'].columns)} features")
    
    return features