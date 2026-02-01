#!/usr/bin/env python3
"""
CGECD: Correlation Graph Eigenvalue Crisis Detector

MAIN EXPERIMENT: Compare our novel CGECD against SOTA benchmarks
on TWO specific prediction tasks:
1. Large up move 3d (>3%)
2. Large down move 10d (>3%)

Output: Clean comparison tables
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import Config
from metrics import compute_metrics, Metrics
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    CGECDModel, walk_forward_evaluate
)
from benchmarks import (
    prepare_benchmark_features,
    AbsorptionRatioModel, TurbulenceModel, GARCHModel, HARRVModel,
    RandomForestModel, LogisticRegressionModel
)


def print_results_table(task_name: str, results: List[Dict], positive_rate: float):
    """Print a clean results table for one task"""
    
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"Positive rate: {positive_rate:.1%}")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'AUC-ROC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    
    # Sort by AUC-ROC descending
    results_sorted = sorted(results, key=lambda x: x['auc_roc'], reverse=True)
    
    for r in results_sorted:
        marker = " ***" if r['is_ours'] else ""
        print(f"{r['model']:<30} {r['auc_roc']:>10.3f} {r['precision']:>9.1%} {r['recall']:>9.1%} {r['f1']:>9.1%}{marker}")
    
    print("-" * 80)
    print("*** = Our method (CGECD)")


def run_experiment():
    """Run the main experiment comparing CGECD vs SOTA benchmarks"""
    
    print("=" * 80)
    print("CORRELATION GRAPH EIGENVALUE CRISIS DETECTOR (CGECD)")
    print("Comparison against SOTA Benchmarks")
    print("=" * 80)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[1/4] Loading data...")
    prices, returns = load_data(config)
    
    # =========================================================================
    # BUILD FEATURES
    # =========================================================================
    print("\n[2/4] Building features...")
    
    # Our spectral features
    print("\n  Building CGECD spectral features...")
    spectral_features = build_spectral_features(returns, config)
    
    # Traditional features
    print("\n  Building traditional features...")
    traditional_features = build_traditional_features(prices)
    
    # Combined features for our method
    cgecd_features = pd.concat([spectral_features, traditional_features], axis=1)
    
    # Benchmark features
    print("\n  Building benchmark features...")
    benchmark_features = prepare_benchmark_features(prices, returns)
    
    print(f"\n  Feature counts:")
    print(f"    CGECD (spectral + traditional): {len(cgecd_features.columns)}")
    print(f"    Traditional only: {len(traditional_features.columns)}")
    for name, df in benchmark_features.items():
        print(f"    {name}: {len(df.columns)}")
    
    # =========================================================================
    # DEFINE TARGETS
    # =========================================================================
    print("\n[3/4] Computing targets...")
    
    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    
    # Task 1: Large up move 3d (>3%)
    future_ret_3d = market.pct_change(3).shift(-3)
    target_up_3pct_3d = (future_ret_3d > 0.03).astype(int)
    
    # Task 2: Large down move 10d (>3%)
    future_ret_10d = market.pct_change(10).shift(-10)
    target_down_3pct_10d = (future_ret_10d < -0.03).astype(int)
    
    tasks = {
        'Large UP move 3d (>3%)': target_up_3pct_3d,
        'Large DOWN move 10d (>3%)': target_down_3pct_10d,
    }
    
    for name, target in tasks.items():
        pos_rate = target.dropna().mean()
        n_pos = target.dropna().sum()
        print(f"  {name}: {pos_rate:.1%} positive ({int(n_pos)} events)")
    
    # =========================================================================
    # RUN EXPERIMENTS
    # =========================================================================
    print("\n[4/4] Running experiments...")
    
    all_task_results = {}
    
    for task_name, target in tasks.items():
        print(f"\n  Evaluating: {task_name}")
        
        task_results = []
        pos_rate = target.dropna().mean()
        
        # --- OUR METHOD: CGECD ---
        print("    Testing CGECD (Ours)...")
        res = walk_forward_evaluate(cgecd_features, target, CGECDModel, config)
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'CGECD (Ours)',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': True
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 1: Absorption Ratio ---
        print("    Testing Absorption Ratio...")
        res = walk_forward_evaluate(
            benchmark_features['absorption_ratio'], target, AbsorptionRatioModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'Absorption Ratio (Kritzman)',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 2: Turbulence Index ---
        print("    Testing Turbulence Index...")
        res = walk_forward_evaluate(
            benchmark_features['turbulence'], target, TurbulenceModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'Turbulence Index (Kritzman)',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 3: GARCH ---
        print("    Testing GARCH...")
        res = walk_forward_evaluate(
            benchmark_features['garch'], target, GARCHModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'GARCH(1,1)',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 4: HAR-RV ---
        print("    Testing HAR-RV...")
        res = walk_forward_evaluate(
            benchmark_features['har_rv'], target, HARRVModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'HAR-RV (Corsi)',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 5: RF with Traditional Features ---
        print("    Testing RF + Traditional...")
        res = walk_forward_evaluate(
            traditional_features, target, RandomForestModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'RF + Traditional Features',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        # --- BENCHMARK 6: Logistic Regression ---
        print("    Testing Logistic Regression...")
        res = walk_forward_evaluate(
            traditional_features, target, LogisticRegressionModel, config
        )
        if 'error' not in res:
            m = res['metrics']
            task_results.append({
                'model': 'Logistic Regression',
                'auc_roc': m.auc_roc,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'is_ours': False
            })
            print(f"      AUC: {m.auc_roc:.3f}")
        
        all_task_results[task_name] = (task_results, pos_rate)
    
    # =========================================================================
    # PRINT RESULTS TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for task_name, (results, pos_rate) in all_task_results.items():
        print_results_table(task_name, results, pos_rate)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for task_name, (results, pos_rate) in all_task_results.items():
        our_result = next((r for r in results if r['is_ours']), None)
        if our_result:
            other_results = [r for r in results if not r['is_ours']]
            if other_results:
                best_benchmark = max(other_results, key=lambda x: x['auc_roc'])
                diff = our_result['auc_roc'] - best_benchmark['auc_roc']
                
                if diff > 0.02:
                    print(f"\n✓ {task_name}:")
                    print(f"  CGECD (AUC={our_result['auc_roc']:.3f}) beats {best_benchmark['model']} (AUC={best_benchmark['auc_roc']:.3f}) by +{diff:.3f}")
                elif diff < -0.02:
                    print(f"\n✗ {task_name}:")
                    print(f"  {best_benchmark['model']} (AUC={best_benchmark['auc_roc']:.3f}) beats CGECD (AUC={our_result['auc_roc']:.3f}) by +{-diff:.3f}")
                else:
                    print(f"\n≈ {task_name}:")
                    print(f"  CGECD (AUC={our_result['auc_roc']:.3f}) ≈ {best_benchmark['model']} (AUC={best_benchmark['auc_roc']:.3f})")
    
    # Save results
    results_data = []
    for task_name, (results, pos_rate) in all_task_results.items():
        for r in results:
            results_data.append({
                'Task': task_name,
                'Model': r['model'],
                'AUC-ROC': r['auc_roc'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1': r['f1'],
                'Is_Ours': r['is_ours']
            })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(config.output_dir / 'experiment_results.csv', index=False)
    
    print(f"\nResults saved to: {config.output_dir}/experiment_results.csv")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results_df


if __name__ == "__main__":
    results = run_experiment()