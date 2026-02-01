#!/usr/bin/env python3
"""
Correlation Graph Eigenvalue Crisis Detector (CGECD)
====================================================

Novel approach: Use spectral properties of dynamic correlation networks
to detect regime shifts and predict crisis events.

Key insight: During crises, normally uncorrelated assets become correlated,
causing measurable changes in the correlation matrix eigenvalue spectrum
BEFORE the full crisis manifests.
"""

import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from config import Config
from metrics import compute_metrics, Metrics

warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """Load market data from EODHD API with caching"""
    
    BASE_URL = "https://eodhd.com/api"
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_file = symbol.replace('.', '_').replace('/', '_')
        cache_path = self.config.cache_dir / f"{cache_file}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    if len(df) > 100:
                        return df
            except Exception:
                pass
        
        try:
            params = {
                'api_token': self.config.api_key,
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
            
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            return pd.DataFrame()


def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load multi-asset price data and compute returns"""
    
    loader = DataLoader(config)
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
    
    prices = pd.DataFrame(data_dict)
    prices = prices.dropna(how='all').ffill(limit=5).dropna()
    returns = prices.pct_change().dropna()
    
    print(f"\nDataset: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices, returns


# =============================================================================
# CORRELATION GRAPH CONSTRUCTION
# =============================================================================

class CorrelationGraphBuilder:
    """Build and analyze dynamic correlation graphs."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.n_assets = len(returns.columns)
    
    def compute_rolling_correlation(self, window: int = 60) -> Dict[pd.Timestamp, np.ndarray]:
        """Compute rolling correlation matrices"""
        corr_matrices = {}
        
        for i in range(window, len(self.returns)):
            date = self.returns.index[i]
            window_returns = self.returns.iloc[i-window:i]
            corr = window_returns.corr().values
            corr = np.nan_to_num(corr, nan=0)
            np.fill_diagonal(corr, 1.0)
            corr_matrices[date] = corr
        
        return corr_matrices
    
    def compute_ewm_correlation(self, halflife: int = 30) -> Dict[pd.Timestamp, np.ndarray]:
        """Compute exponentially weighted correlation matrices"""
        corr_matrices = {}
        n = self.n_assets
        
        for i in range(60, len(self.returns)):
            date = self.returns.index[i]
            window_returns = self.returns.iloc[:i]
            
            ewm_cov = window_returns.ewm(halflife=halflife).cov().iloc[-n:]
            ewm_cov = ewm_cov.values.reshape(n, n)
            
            std = np.sqrt(np.diag(ewm_cov))
            std_outer = np.outer(std, std)
            std_outer[std_outer == 0] = 1
            corr = ewm_cov / std_outer
            
            corr = np.nan_to_num(corr, nan=0)
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -1, 1)
            corr_matrices[date] = corr
        
        return corr_matrices


# =============================================================================
# SPECTRAL FEATURE EXTRACTION
# =============================================================================

class SpectralFeatureExtractor:
    """Extract features from eigenvalue decomposition of correlation matrices."""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
    
    def extract_eigenvalue_features(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """Extract features from eigenvalue decomposition."""
        
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 1e-10)
        except Exception:
            return self._default_features()
        
        n = len(eigenvalues)
        total_var = np.sum(eigenvalues)
        
        features = {}
        
        # Primary eigenvalues
        features['lambda_1'] = eigenvalues[0]
        features['lambda_1_ratio'] = eigenvalues[0] / total_var
        features['lambda_2'] = eigenvalues[1] if n > 1 else 0
        features['lambda_3'] = eigenvalues[2] if n > 2 else 0
        features['spectral_gap'] = eigenvalues[0] / (eigenvalues[1] + 1e-10) if n > 1 else n
        
        # Absorption ratios
        for k in [1, 3, 5]:
            features[f'absorption_ratio_{k}'] = np.sum(eigenvalues[:min(k, n)]) / total_var
        
        # Eigenvalue entropy
        normalized_eig = eigenvalues / total_var
        entropy = -np.sum(normalized_eig * np.log(normalized_eig + 1e-10))
        max_entropy = np.log(n)
        features['eigenvalue_entropy'] = entropy
        features['normalized_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
        
        # Effective rank
        features['effective_rank'] = np.exp(entropy)
        features['effective_rank_ratio'] = features['effective_rank'] / n
        
        # Marchenko-Pastur excess
        mp_upper = (1 + np.sqrt(1)) ** 2
        features['mp_excess'] = max(0, eigenvalues[0] - mp_upper)
        
        # Higher order statistics
        features['eigenvalue_std'] = np.std(eigenvalues)
        features['eigenvalue_skew'] = stats.skew(eigenvalues)
        features['eigenvalue_kurt'] = stats.kurtosis(eigenvalues)
        features['tail_eigenvalue_mean'] = np.mean(eigenvalues[-5:]) if n >= 5 else np.mean(eigenvalues)
        
        # Condition number
        features['condition_number'] = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        features['log_condition_number'] = np.log(features['condition_number'] + 1)
        
        return features
    
    def extract_eigenvector_features(self, corr_matrix: np.ndarray) -> Dict[str, float]:
        """Extract features from first eigenvector."""
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
        except Exception:
            return {}
        
        features = {}
        v1 = np.abs(eigenvectors[:, 0])
        v1 = v1 / (np.sum(v1) + 1e-10)
        
        features['v1_entropy'] = -np.sum(v1 * np.log(v1 + 1e-10))
        features['v1_max'] = np.max(v1)
        features['v1_min'] = np.min(v1)
        features['v1_std'] = np.std(v1)
        features['v1_herfindahl'] = np.sum(v1 ** 2)
        
        n = len(v1)
        if n >= 4:
            sorted_loadings = np.sort(v1)
            features['loading_dispersion'] = np.mean(sorted_loadings[-n//4:]) - np.mean(sorted_loadings[:n//4])
        
        return features
    
    def _default_features(self) -> Dict[str, float]:
        """Default features when eigendecomposition fails"""
        return {
            'lambda_1': 1.0, 'lambda_1_ratio': 1.0 / self.n_assets,
            'lambda_2': 0.0, 'lambda_3': 0.0, 'spectral_gap': 1.0,
            'absorption_ratio_1': 1.0 / self.n_assets,
            'absorption_ratio_3': 3.0 / self.n_assets,
            'absorption_ratio_5': 5.0 / self.n_assets,
            'eigenvalue_entropy': np.log(self.n_assets),
            'normalized_entropy': 1.0,
            'effective_rank': self.n_assets, 'effective_rank_ratio': 1.0,
            'mp_excess': 0.0, 'eigenvalue_std': 0.0,
            'eigenvalue_skew': 0.0, 'eigenvalue_kurt': 0.0,
            'tail_eigenvalue_mean': 1.0,
            'condition_number': 1.0, 'log_condition_number': 0.0,
        }


# =============================================================================
# GRAPH TOPOLOGY FEATURES
# =============================================================================

class GraphTopologyExtractor:
    """Extract features from the network topology of the correlation graph."""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
    
    def extract_topology_features(
        self, 
        corr_matrix: np.ndarray,
        thresholds: List[float] = [0.3, 0.5, 0.7]
    ) -> Dict[str, float]:
        """Extract graph topology features at multiple thresholds."""
        
        n = self.n_assets
        features = {}
        
        for thresh in thresholds:
            adj = (np.abs(corr_matrix) > thresh).astype(float)
            np.fill_diagonal(adj, 0)
            suffix = f"_t{int(thresh*100)}"
            
            n_edges = np.sum(adj) / 2
            max_edges = n * (n - 1) / 2
            features[f'edge_density{suffix}'] = n_edges / max_edges if max_edges > 0 else 0
            
            degrees = np.sum(adj, axis=1)
            features[f'degree_mean{suffix}'] = np.mean(degrees)
            features[f'degree_std{suffix}'] = np.std(degrees)
            features[f'degree_max{suffix}'] = np.max(degrees)
            features[f'isolated_nodes{suffix}'] = np.sum(degrees == 0)
            
            if np.max(degrees) > 0:
                features[f'centralization{suffix}'] = np.sum(np.max(degrees) - degrees) / ((n - 1) * (n - 2))
            else:
                features[f'centralization{suffix}'] = 0
            
            # Clustering coefficient
            triangles = 0
            triplets = 0
            for i in range(n):
                neighbors = np.where(adj[i] > 0)[0]
                k = len(neighbors)
                if k >= 2:
                    triplets += k * (k - 1) / 2
                    for j in range(len(neighbors)):
                        for m in range(j + 1, len(neighbors)):
                            if adj[neighbors[j], neighbors[m]] > 0:
                                triangles += 1
            features[f'clustering_coef{suffix}'] = triangles / triplets if triplets > 0 else 0
        
        # Correlation statistics
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        features['mean_abs_corr'] = np.mean(np.abs(upper_tri))
        features['median_abs_corr'] = np.median(np.abs(upper_tri))
        features['max_abs_corr'] = np.max(np.abs(upper_tri))
        features['corr_std'] = np.std(upper_tri)
        features['corr_skew'] = stats.skew(upper_tri)
        features['frac_corr_above_50'] = np.mean(np.abs(upper_tri) > 0.5)
        features['frac_corr_above_70'] = np.mean(np.abs(upper_tri) > 0.7)
        
        return features


# =============================================================================
# DYNAMICS FEATURES
# =============================================================================

def compute_dynamics_features(
    feature_df: pd.DataFrame,
    lookbacks: tuple = (5, 10, 20)
) -> pd.DataFrame:
    """Compute rate of change and acceleration of key features."""
    
    key_features = [
        'lambda_1', 'lambda_1_ratio', 'absorption_ratio_1',
        'eigenvalue_entropy', 'effective_rank', 'mean_abs_corr',
        'edge_density_t50', 'clustering_coef_t50'
    ]
    
    dynamics = pd.DataFrame(index=feature_df.index)
    
    for feat in key_features:
        col = f'{feat}_60d'
        if col not in feature_df.columns:
            continue
        
        series = feature_df[col]
        
        for lb in lookbacks:
            dynamics[f'{feat}_roc_{lb}d'] = series.pct_change(lb)
            dynamics[f'{feat}_diff_{lb}d'] = series.diff(lb)
            
            rolling_mean = series.rolling(lb * 2).mean()
            rolling_std = series.rolling(lb * 2).std()
            dynamics[f'{feat}_zscore_{lb}d'] = (series - rolling_mean) / (rolling_std + 1e-10)
        
        dynamics[f'{feat}_accel'] = series.diff().diff()
        dynamics[f'{feat}_pct_252d'] = series.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 10 else 0.5
        )
    
    return dynamics


# =============================================================================
# BUILD ALL FEATURES
# =============================================================================

def build_spectral_features(returns: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build complete spectral feature set."""
    
    print("Building correlation matrices...")
    graph_builder = CorrelationGraphBuilder(returns)
    
    corr_60d = graph_builder.compute_rolling_correlation(window=60)
    corr_120d = graph_builder.compute_rolling_correlation(window=120)
    corr_ewm = graph_builder.compute_ewm_correlation(halflife=config.ewm_halflife)
    
    n_assets = len(returns.columns)
    spectral_extractor = SpectralFeatureExtractor(n_assets)
    topology_extractor = GraphTopologyExtractor(n_assets)
    
    print("Extracting spectral features...")
    all_features = []
    
    for date in corr_60d.keys():
        row = {'date': date}
        
        # 60d window features
        eig_feats = spectral_extractor.extract_eigenvalue_features(corr_60d[date])
        for k, v in eig_feats.items():
            row[f'{k}_60d'] = v
        
        evec_feats = spectral_extractor.extract_eigenvector_features(corr_60d[date])
        for k, v in evec_feats.items():
            row[f'{k}_60d'] = v
        
        topo_feats = topology_extractor.extract_topology_features(
            corr_60d[date], list(config.graph_thresholds)
        )
        for k, v in topo_feats.items():
            row[f'{k}_60d'] = v
        
        # 120d window features
        if date in corr_120d:
            eig_feats_120 = spectral_extractor.extract_eigenvalue_features(corr_120d[date])
            for k, v in eig_feats_120.items():
                row[f'{k}_120d'] = v
        
        # EWM features
        if date in corr_ewm:
            eig_feats_ewm = spectral_extractor.extract_eigenvalue_features(corr_ewm[date])
            for k, v in eig_feats_ewm.items():
                row[f'{k}_ewm'] = v
        
        all_features.append(row)
    
    feature_df = pd.DataFrame(all_features)
    feature_df.set_index('date', inplace=True)
    feature_df = feature_df.sort_index()
    
    print("Computing dynamics features...")
    dynamics_df = compute_dynamics_features(feature_df, config.dynamics_lookbacks)
    
    full_features = pd.concat([feature_df, dynamics_df], axis=1)
    print(f"Total spectral features: {len(full_features.columns)}")
    
    return full_features


def build_traditional_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build traditional technical features."""
    
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    returns = market.pct_change()
    
    features = pd.DataFrame(index=prices.index)
    
    # Returns
    for window in [1, 5, 10, 20, 60]:
        features[f'return_{window}d'] = market.pct_change(window)
    
    # Volatility
    for window in [5, 10, 20, 60]:
        features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
    
    features['vol_ratio_5_20'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)
    
    # Momentum
    for window in [10, 20, 50]:
        sma = market.rolling(window).mean()
        features[f'price_to_sma_{window}'] = market / sma - 1
    
    # RSI
    delta = market.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Drawdown
    for window in [20, 60]:
        rolling_max = market.rolling(window).max()
        features[f'drawdown_{window}d'] = (market - rolling_max) / rolling_max
    
    # Higher moments
    for window in [20, 60]:
        features[f'skewness_{window}d'] = returns.rolling(window).skew()
        features[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
    
    print(f"Total traditional features: {len(features.columns)}")
    
    return features


# =============================================================================
# MODELS
# =============================================================================

class CGECDModel:
    """Our novel method: CGECD with Random Forest"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.name = "CGECD (Ours)"
    
    def fit(self, X: np.ndarray, y: np.ndarray):
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# =============================================================================
# WALK-FORWARD EVALUATION
# =============================================================================

def walk_forward_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    model_class,
    config: Config
) -> Dict:
    """Walk-forward cross-validation"""
    
    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    common_idx = common_idx.intersection(target.dropna().index)
    
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    
    train_size = int(config.train_years * 252)
    test_size = int(config.test_months * 21)
    gap = config.gap_days
    
    if len(X) < train_size + gap + test_size:
        return {'error': 'Insufficient data'}
    
    available = len(X) - train_size - gap - test_size
    step = max(test_size, available // config.n_splits)
    
    all_probs = []
    all_actuals = []
    all_dates = []
    
    for fold in range(config.n_splits):
        start = fold * step
        train_end = start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, len(X))
        
        if test_end > len(X):
            break
        
        X_train = X.iloc[start:train_end].values
        y_train = y.iloc[start:train_end].values
        X_test = X.iloc[test_start:test_end].values
        y_test = y.iloc[test_start:test_end].values
        
        try:
            model = model_class(config)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            
            all_probs.extend(probs)
            all_actuals.extend(y_test)
            all_dates.extend(y.iloc[test_start:test_end].index.tolist())
        except Exception as e:
            print(f"    Fold {fold} failed: {e}")
            continue
    
    if len(all_probs) == 0:
        return {'error': 'All folds failed'}
    
    all_probs = np.array(all_probs)
    all_actuals = np.array(all_actuals)
    all_preds = (all_probs >= 0.5).astype(int)
    
    metrics = compute_metrics(all_actuals, all_preds, all_probs)
    
    return {
        'metrics': metrics,
        'probabilities': all_probs,
        'actuals': all_actuals,
        'predictions': all_preds,
        'dates': all_dates
    }