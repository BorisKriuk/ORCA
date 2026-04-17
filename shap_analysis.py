#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for CGECD
===========================================

Trains the CGECD model on both tasks (rally + crash detection),
computes SHAP values, and produces publication-quality visualizations
comparing which features drive each prediction task.

Outputs (results/ directory):
  shap_beeswarm_rally.png       — SHAP beeswarm for rally detection
  shap_beeswarm_crash.png       — SHAP beeswarm for crash detection
  shap_bar_comparison.png       — Top-20 features side-by-side comparison
  shap_feature_overlap.png      — Venn-style overlap of top features
  shap_spectral_vs_trad.png     — Spectral vs traditional contribution breakdown
  shap_scatter_top6.png         — SHAP dependence plots for top features
  shap_heatmap_comparison.png   — Signed mean SHAP heatmap (rally vs crash)
  shap_values_rally.csv         — Raw SHAP values for rally task
  shap_values_crash.csv         — Raw SHAP values for crash task
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.preprocessing import RobustScaler

from config import Config
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, CGECDModel
)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'sans-serif',
})

RALLY_COLOR = '#2196F3'   # blue
CRASH_COLOR = '#E53935'   # red
SPECTRAL_COLOR = '#7E57C2'  # purple
TRAD_COLOR = '#FF9800'    # orange


def classify_feature(name):
    """Classify a feature as 'spectral' or 'traditional'."""
    spectral_keywords = [
        'lambda', 'spectral', 'absorption', 'eigenvalue', 'effective_rank',
        'mp_excess', 'condition_number', 'log_condition', 'v1_',
        'edge_density', 'degree_', 'clustering_coef', 'centralization',
        'isolated_nodes', 'mean_abs_corr', 'median_abs_corr', 'max_abs_corr',
        'corr_std', 'corr_skew', 'frac_corr_above', 'loading_dispersion',
        'herfindahl', 'tail_eigenvalue', 'normalized_entropy',
        '_roc_', '_diff_', '_zscore_', '_accel', '_pct_252d',
    ]
    # dynamics features derived from spectral base features
    dynamics_base = [
        'lambda_1_roc', 'lambda_1_ratio_roc', 'absorption_ratio_1_roc',
        'eigenvalue_entropy_roc', 'effective_rank_roc', 'mean_abs_corr_roc',
        'edge_density_t50_roc', 'clustering_coef_t50_roc',
        'lambda_1_diff', 'lambda_1_ratio_diff', 'absorption_ratio_1_diff',
        'eigenvalue_entropy_diff', 'effective_rank_diff', 'mean_abs_corr_diff',
        'edge_density_t50_diff', 'clustering_coef_t50_diff',
        'lambda_1_zscore', 'lambda_1_ratio_zscore', 'absorption_ratio_1_zscore',
        'eigenvalue_entropy_zscore', 'effective_rank_zscore', 'mean_abs_corr_zscore',
        'edge_density_t50_zscore', 'clustering_coef_t50_zscore',
        'lambda_1_accel', 'lambda_1_ratio_accel', 'absorption_ratio_1_accel',
        'eigenvalue_entropy_accel', 'effective_rank_accel', 'mean_abs_corr_accel',
        'edge_density_t50_accel', 'clustering_coef_t50_accel',
    ]
    name_lower = name.lower()
    for kw in spectral_keywords:
        if kw in name_lower:
            return 'spectral'
    for db in dynamics_base:
        if db in name_lower:
            return 'spectral'
    return 'traditional'


def prettify_name(name, max_len=35):
    """Make feature names more readable for plots."""
    name = name.replace('_60d', ' (60d)').replace('_120d', ' (120d)')
    name = name.replace('_ewm', ' (EWM)')
    name = name.replace('_', ' ')
    if len(name) > max_len:
        name = name[:max_len-1] + '.'
    return name


def train_and_get_shap(features, target, config, task_name, n_background=300):
    """Train CGECD model on full aligned data and compute SHAP values."""

    common_idx = features.dropna(thresh=int(len(features.columns) * 0.5)).index
    common_idx = common_idx.intersection(target.dropna().index)

    X = features.loc[common_idx]
    y = target.loc[common_idx]

    # Fill remaining NaN for SHAP
    X = X.fillna(0)

    print(f"  [{task_name}] Training on {len(X)} samples, {len(X.columns)} features ...")

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=X.columns
    )
    X_scaled = X_scaled.replace([np.inf, -np.inf], 0)

    model = CGECDModel(config)
    model.fit(X_scaled.values, y.values)

    # Use TreeExplainer for Random Forest (exact, fast)
    explainer = shap.TreeExplainer(model.model)

    # Subsample for speed if large
    if len(X_scaled) > 3000:
        idx = np.random.RandomState(42).choice(len(X_scaled), 3000, replace=False)
        X_shap = X_scaled.iloc[idx]
    else:
        X_shap = X_scaled

    print(f"  [{task_name}] Computing SHAP values on {len(X_shap)} samples ...")
    shap_values = explainer.shap_values(X_shap.values)

    # For binary classifier, shap_values may be list [class_0, class_1] or 3D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X_shap.index)

    print(f"  [{task_name}] Done. Mean |SHAP| range: "
          f"{np.abs(shap_values).mean(axis=0).min():.4f} — "
          f"{np.abs(shap_values).mean(axis=0).max():.4f}")

    return shap_df, X_shap, explainer


def plot_beeswarm(shap_df, X_data, title, save_path, top_n=25, color_map='RdBu_r'):
    """Publication-quality SHAP beeswarm plot."""

    mean_abs = np.abs(shap_df.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    shap_top = shap_df.iloc[:, top_idx]
    X_top = X_data.iloc[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, 0.4 * top_n + 1.5))

    feature_names = [prettify_name(c) for c in shap_top.columns]

    shap.summary_plot(
        shap_top.values,
        X_top.values,
        feature_names=feature_names,
        plot_type='dot',
        show=False,
        max_display=top_n,
        plot_size=None,
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bar_comparison(shap_rally, shap_crash, save_path, top_n=20):
    """Side-by-side horizontal bar chart: top features for rally vs crash."""

    rally_imp = np.abs(shap_rally.values).mean(axis=0)
    crash_imp = np.abs(shap_crash.values).mean(axis=0)

    rally_imp_s = pd.Series(rally_imp, index=shap_rally.columns)
    crash_imp_s = pd.Series(crash_imp, index=shap_crash.columns)

    # Union of top features from both tasks
    top_rally = set(rally_imp_s.nlargest(top_n).index)
    top_crash = set(crash_imp_s.nlargest(top_n).index)
    all_top = sorted(top_rally | top_crash, key=lambda f: rally_imp_s[f] + crash_imp_s[f], reverse=True)[:top_n]

    rally_vals = [rally_imp_s[f] for f in all_top]
    crash_vals = [crash_imp_s[f] for f in all_top]
    labels = [prettify_name(f) for f in all_top]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.45 * top_n + 2), sharey=True)

    y_pos = np.arange(len(all_top))

    # Color by spectral vs traditional
    rally_colors = [SPECTRAL_COLOR if classify_feature(f) == 'spectral' else TRAD_COLOR for f in all_top]
    crash_colors = [SPECTRAL_COLOR if classify_feature(f) == 'spectral' else TRAD_COLOR for f in all_top]

    ax1.barh(y_pos, rally_vals, color=rally_colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('Rally Detection', fontsize=13, fontweight='bold', color=RALLY_COLOR)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.barh(y_pos, crash_vals, color=crash_colors, edgecolor='white', linewidth=0.5, height=0.7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('Crash Detection', fontsize=13, fontweight='bold', color=CRASH_COLOR)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SPECTRAL_COLOR, label='Spectral / Graph'),
        Patch(facecolor=TRAD_COLOR, label='Traditional'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('SHAP Feature Importance: Rally vs Crash Detection',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_overlap(shap_rally, shap_crash, save_path, top_n=20):
    """Visualize overlap and uniqueness of top features between tasks."""

    rally_imp = pd.Series(np.abs(shap_rally.values).mean(axis=0), index=shap_rally.columns)
    crash_imp = pd.Series(np.abs(shap_crash.values).mean(axis=0), index=shap_crash.columns)

    top_rally = set(rally_imp.nlargest(top_n).index)
    top_crash = set(crash_imp.nlargest(top_n).index)

    only_rally = top_rally - top_crash
    only_crash = top_crash - top_rally
    shared = top_rally & top_crash

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, max(len(only_rally), len(only_crash), len(shared)) + 3)
    ax.axis('off')

    # Column headers
    cols = [
        (0.15, f'Rally Only ({len(only_rally)})', RALLY_COLOR, only_rally, rally_imp),
        (1.15, f'Shared ({len(shared)})', '#4CAF50', shared, rally_imp),
        (2.15, f'Crash Only ({len(only_crash)})', CRASH_COLOR, only_crash, crash_imp),
    ]

    for x_start, header, color, feat_set, imp_series in cols:
        ax.text(x_start + 0.35, max(len(only_rally), len(only_crash), len(shared)) + 2,
                header, ha='center', va='center', fontsize=14, fontweight='bold', color=color)

        sorted_feats = sorted(feat_set, key=lambda f: imp_series[f], reverse=True)
        for i, f in enumerate(sorted_feats):
            ftype = classify_feature(f)
            bg_color = SPECTRAL_COLOR if ftype == 'spectral' else TRAD_COLOR
            y = max(len(only_rally), len(only_crash), len(shared)) + 0.8 - i * 1.1
            rect = FancyBboxPatch((x_start, y - 0.35), 0.7, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor=bg_color, alpha=0.15, edgecolor=bg_color, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x_start + 0.35, y, prettify_name(f, 30),
                    ha='center', va='center', fontsize=8.5,
                    color='#222222')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SPECTRAL_COLOR, alpha=0.3, edgecolor=SPECTRAL_COLOR, label='Spectral / Graph'),
        Patch(facecolor=TRAD_COLOR, alpha=0.3, edgecolor=TRAD_COLOR, label='Traditional'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=True)

    fig.suptitle(f'Top-{top_n} Feature Overlap: Rally vs Crash Detection',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spectral_vs_traditional(shap_rally, shap_crash, save_path):
    """Stacked bar: total SHAP contribution from spectral vs traditional features."""

    data = {}
    for task_name, shap_df in [('Rally', shap_rally), ('Crash', shap_crash)]:
        mean_abs = pd.Series(np.abs(shap_df.values).mean(axis=0), index=shap_df.columns)
        spectral_total = sum(mean_abs[f] for f in shap_df.columns if classify_feature(f) == 'spectral')
        trad_total = sum(mean_abs[f] for f in shap_df.columns if classify_feature(f) == 'traditional')
        total = spectral_total + trad_total
        data[task_name] = {
            'spectral': spectral_total,
            'traditional': trad_total,
            'spectral_pct': spectral_total / total * 100,
            'trad_pct': trad_total / total * 100,
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute SHAP contribution
    ax = axes[0]
    tasks = list(data.keys())
    x = np.arange(len(tasks))
    width = 0.5

    spec_vals = [data[t]['spectral'] for t in tasks]
    trad_vals = [data[t]['traditional'] for t in tasks]

    bars1 = ax.bar(x, spec_vals, width, label='Spectral / Graph', color=SPECTRAL_COLOR, edgecolor='white')
    bars2 = ax.bar(x, trad_vals, width, bottom=spec_vals, label='Traditional', color=TRAD_COLOR, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=13)
    ax.set_ylabel('Sum of Mean |SHAP value|', fontsize=12)
    ax.set_title('Absolute SHAP Contribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add value labels
    for i, t in enumerate(tasks):
        ax.text(i, spec_vals[i] / 2, f'{spec_vals[i]:.3f}', ha='center', va='center',
                fontweight='bold', color='white', fontsize=11)
        ax.text(i, spec_vals[i] + trad_vals[i] / 2, f'{trad_vals[i]:.3f}', ha='center', va='center',
                fontweight='bold', color='white', fontsize=11)

    # Right: percentage breakdown
    ax = axes[1]
    spec_pcts = [data[t]['spectral_pct'] for t in tasks]
    trad_pcts = [data[t]['trad_pct'] for t in tasks]

    bars1 = ax.barh(x, spec_pcts, height=0.5, label='Spectral / Graph', color=SPECTRAL_COLOR, edgecolor='white')
    bars2 = ax.barh(x, trad_pcts, height=0.5, left=spec_pcts, label='Traditional', color=TRAD_COLOR, edgecolor='white')

    ax.set_yticks(x)
    ax.set_yticklabels(tasks, fontsize=13)
    ax.set_xlabel('% of Total SHAP Importance', fontsize=12)
    ax.set_title('Relative SHAP Contribution', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add percentage labels
    for i, t in enumerate(tasks):
        ax.text(spec_pcts[i] / 2, i, f'{spec_pcts[i]:.1f}%', ha='center', va='center',
                fontweight='bold', color='white', fontsize=12)
        ax.text(spec_pcts[i] + trad_pcts[i] / 2, i, f'{trad_pcts[i]:.1f}%', ha='center', va='center',
                fontweight='bold', color='white', fontsize=12)

    fig.suptitle('Spectral vs Traditional Feature Contribution (SHAP)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scatter_top_features(shap_rally, shap_crash, X_rally, X_crash, save_path, top_n=6):
    """SHAP dependence scatter plots for top features, rally vs crash side by side."""

    # Pick top features by combined importance
    rally_imp = pd.Series(np.abs(shap_rally.values).mean(axis=0), index=shap_rally.columns)
    crash_imp = pd.Series(np.abs(shap_crash.values).mean(axis=0), index=shap_crash.columns)
    combined_imp = rally_imp + crash_imp
    top_feats = combined_imp.nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(top_n, 2, figsize=(14, 3 * top_n))

    for i, feat in enumerate(top_feats):
        feat_idx = list(shap_rally.columns).index(feat)
        pretty = prettify_name(feat)

        # Rally
        ax = axes[i, 0]
        sc = ax.scatter(
            X_rally.iloc[:, feat_idx], shap_rally.iloc[:, feat_idx],
            c=X_rally.iloc[:, feat_idx], cmap='coolwarm', s=3, alpha=0.4
        )
        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_ylabel('SHAP value', fontsize=9)
        if i == 0:
            ax.set_title('Rally Detection', fontsize=13, fontweight='bold', color=RALLY_COLOR)
        ax.text(0.02, 0.95, pretty, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Crash
        ax = axes[i, 1]
        sc = ax.scatter(
            X_crash.iloc[:, feat_idx], shap_crash.iloc[:, feat_idx],
            c=X_crash.iloc[:, feat_idx], cmap='coolwarm', s=3, alpha=0.4
        )
        ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
        if i == 0:
            ax.set_title('Crash Detection', fontsize=13, fontweight='bold', color=CRASH_COLOR)
        ax.text(0.02, 0.95, pretty, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[-1, 0].set_xlabel('Feature value (scaled)')
    axes[-1, 1].set_xlabel('Feature value (scaled)')

    fig.suptitle('SHAP Dependence: How Top Features Drive Rally vs Crash Predictions',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_signed_heatmap(shap_rally, shap_crash, save_path, top_n=25):
    """Heatmap of signed mean SHAP values — shows direction of effect."""

    rally_mean = shap_rally.mean(axis=0)
    crash_mean = shap_crash.mean(axis=0)

    # Top features by absolute importance across both tasks
    combined_abs = np.abs(shap_rally.values).mean(axis=0) + np.abs(shap_crash.values).mean(axis=0)
    combined_abs_s = pd.Series(combined_abs, index=shap_rally.columns)
    top_feats = combined_abs_s.nlargest(top_n).index.tolist()

    heatmap_data = pd.DataFrame({
        'Rally (signed mean SHAP)': [rally_mean[f] for f in top_feats],
        'Crash (signed mean SHAP)': [crash_mean[f] for f in top_feats],
    }, index=[prettify_name(f) for f in top_feats])

    # Add feature type annotation
    feat_types = [classify_feature(f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 2))

    vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
    sns.heatmap(
        heatmap_data, annot=True, fmt='.4f', cmap='RdBu_r',
        center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.5, linecolor='white',
        ax=ax, cbar_kws={'label': 'Mean SHAP value'}
    )

    # Color y-tick labels by feature type
    for i, (label, ftype) in enumerate(zip(ax.get_yticklabels(), feat_types)):
        label.set_color(SPECTRAL_COLOR if ftype == 'spectral' else TRAD_COLOR)
        label.set_fontweight('bold')

    ax.set_title('Signed Mean SHAP: Direction of Feature Effects\n(purple = spectral, orange = traditional)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary(shap_rally, shap_crash):
    """Print text summary of key findings."""

    rally_imp = pd.Series(np.abs(shap_rally.values).mean(axis=0), index=shap_rally.columns)
    crash_imp = pd.Series(np.abs(shap_crash.values).mean(axis=0), index=shap_crash.columns)

    print("\n" + "=" * 80)
    print("  SHAP ANALYSIS SUMMARY")
    print("=" * 80)

    for task_name, imp in [("RALLY", rally_imp), ("CRASH", crash_imp)]:
        top10 = imp.nlargest(10)
        n_spectral = sum(1 for f in top10.index if classify_feature(f) == 'spectral')
        total_spectral = sum(imp[f] for f in imp.index if classify_feature(f) == 'spectral')
        total_all = imp.sum()

        print(f"\n  {task_name} DETECTION — Top 10 features:")
        for i, (feat, val) in enumerate(top10.items(), 1):
            ftype = classify_feature(feat)
            tag = "[S]" if ftype == 'spectral' else "[T]"
            print(f"    {i:>2}. {tag} {prettify_name(feat, 45):<48} {val:.4f}")

        print(f"\n    Spectral in top-10: {n_spectral}/10")
        print(f"    Spectral share of total importance: {total_spectral/total_all:.1%}")

    # Overlap analysis
    top20_rally = set(rally_imp.nlargest(20).index)
    top20_crash = set(crash_imp.nlargest(20).index)
    shared = top20_rally & top20_crash
    only_rally = top20_rally - top20_crash
    only_crash = top20_crash - top20_rally

    print(f"\n  TOP-20 FEATURE OVERLAP:")
    print(f"    Shared:      {len(shared)}/20")
    print(f"    Rally-only:  {len(only_rally)}")
    print(f"    Crash-only:  {len(only_crash)}")

    if shared:
        print(f"\n    Shared features:")
        for f in sorted(shared, key=lambda x: rally_imp[x] + crash_imp[x], reverse=True):
            print(f"      {prettify_name(f, 45):<48} R={rally_imp[f]:.4f}  C={crash_imp[f]:.4f}")

    print("\n" + "=" * 80)


def load_data_from_cache(config):
    """Load data from cached pickle files (no API key needed)."""
    import pickle
    cache_dir = config.cache_dir
    data_dict = {}

    print(f"Loading {len(config.symbols)} assets from cache ...")
    for symbol, name in config.symbols.items():
        cache_file = symbol.replace('.', '_').replace('/', '_') + '.pkl'
        cache_path = cache_dir / cache_file
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            if not df.empty and 'adjusted_close' in df.columns:
                data_dict[name] = df['adjusted_close']
                print(f"  ✓ {name}: {len(df)} days")
            else:
                print(f"  ✗ {name}: no adjusted_close")
        else:
            print(f"  ✗ {name}: cache miss ({cache_file})")

    prices = pd.DataFrame(data_dict)
    prices = prices.dropna(how='all').ffill(limit=5).dropna()
    returns = prices.pct_change().dropna()

    print(f"\nDataset: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices, returns


def run():
    """Main entry point."""
    print("=" * 80)
    print("  CGECD — SHAP Feature Importance Analysis")
    print("=" * 80)

    # Bypass API key requirement — we load from cache
    import os
    os.environ.setdefault('EODHD_API_KEY', 'cache_only')
    cfg = Config()
    out = cfg.output_dir

    # Load data from cache
    print("\n[1/5] Loading data ...")
    prices, returns = load_data_from_cache(cfg)

    print("\n[2/5] Building features ...")
    spectral = build_spectral_features(returns, cfg)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)

    print(f"  Combined: {len(combined.columns)} features "
          f"({len(spectral.columns)} spectral + {len(traditional.columns)} traditional)")

    # Targets
    print("\n[3/5] Computing targets ...")
    all_targets = compute_all_targets(prices)
    rally_target = all_targets['up_3pct_10d']
    crash_target = all_targets['drawdown_7pct_10d']

    r_rate = rally_target.dropna().mean()
    c_rate = crash_target.dropna().mean()
    print(f"  Rally (up >3% in 10d):       {r_rate:.1%}  ({int(rally_target.sum())} events)")
    print(f"  Crash (drawdown >7% in 10d): {c_rate:.1%}  ({int(crash_target.sum())} events)")

    # Train and compute SHAP
    print("\n[4/5] Training models and computing SHAP values ...")
    shap_rally, X_rally, _ = train_and_get_shap(combined, rally_target, cfg, "Rally")
    shap_crash, X_crash, _ = train_and_get_shap(combined, crash_target, cfg, "Crash")

    # Save raw SHAP values
    shap_rally.to_csv(out / 'shap_values_rally.csv')
    shap_crash.to_csv(out / 'shap_values_crash.csv')
    print(f"  Saved: {out / 'shap_values_rally.csv'}")
    print(f"  Saved: {out / 'shap_values_crash.csv'}")

    # Generate all plots
    print("\n[5/5] Generating visualizations ...")

    plot_beeswarm(
        shap_rally, X_rally,
        'SHAP Feature Importance — Rally Detection (Up >3% in 10 days)',
        out / 'shap_beeswarm_rally.png', top_n=25
    )

    plot_beeswarm(
        shap_crash, X_crash,
        'SHAP Feature Importance — Crash Detection (Drawdown >7% in 10 days)',
        out / 'shap_beeswarm_crash.png', top_n=25
    )

    plot_bar_comparison(shap_rally, shap_crash, out / 'shap_bar_comparison.png', top_n=20)

    plot_feature_overlap(shap_rally, shap_crash, out / 'shap_feature_overlap.png', top_n=20)

    plot_spectral_vs_traditional(shap_rally, shap_crash, out / 'shap_spectral_vs_trad.png')

    plot_scatter_top_features(
        shap_rally, shap_crash, X_rally, X_crash,
        out / 'shap_scatter_top6.png', top_n=6
    )

    plot_signed_heatmap(shap_rally, shap_crash, out / 'shap_heatmap_comparison.png', top_n=25)

    # Print summary
    print_summary(shap_rally, shap_crash)

    print(f"\n  All outputs saved to {out}/")
    print("=" * 80)


if __name__ == "__main__":
    run()
