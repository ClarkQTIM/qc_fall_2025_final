"""
statistical_analysis.py

Comprehensive statistical analysis of quantum-classical hybrid ML experiments.

Tests 6 key hypotheses:
1. P1 vs P3: Classical vs Quantum Direct Classification
2. P2a vs P2b vs P2c: Feature Extraction Methods
3. P2d Noise Sweet Spot: Optimal noise levels for density matrix
4. P1 Dropout Sweep: Optimal dropout for classical
5. Best P2 vs P3: Feature extraction vs direct classification
6. Noise Type Comparison: Which noise type is best for P2d?

Usage:
    python statistical_analysis.py \
        --results_dir /path/to/results \
        --output_dir /path/to/analysis
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Load all metrics.json files from results directory.
    
    Returns:
        DataFrame with columns: pipeline, config, run_id, auc, accuracy, etc.
    """
    all_data = []
    
    for pipeline_dir in sorted(results_dir.glob("p*")):
        if not pipeline_dir.is_dir():
            continue
        
        pipeline_name = pipeline_dir.name
        
        # Extract pipeline type and config
        if pipeline_name.startswith("p1_"):
            # P1: Classical with dropout
            # e.g., p1_classical_dropout_0.5
            parts = pipeline_name.split('_')
            if len(parts) >= 4:
                dropout_val = parts[3]
                pipeline_type = "P1"
                config = f"dropout_{dropout_val}"
        
        elif pipeline_name.startswith("p2a_"):
            # P2a: Probabilities
            pipeline_type = "P2a"
            config = "probabilities"
        
        elif pipeline_name.startswith("p2b_"):
            # P2b: Complex amplitudes
            pipeline_type = "P2b"
            config = "complex_amplitudes"
        
        elif pipeline_name.startswith("p2c_"):
            # P2c: Pauli expectations
            pipeline_type = "P2c"
            config = "pauli"
        
        elif pipeline_name.startswith("p2d_"):
            # P2d: Density matrix with noise
            # e.g., p2d_density_matrix_bit_flip_0.05
            parts = pipeline_name.split('_')
            if len(parts) >= 5:
                noise_type = parts[3]
                if len(parts) >= 6:
                    noise_type = f"{parts[3]}_{parts[4]}"  # amp_damp or bit_flip
                noise_level = parts[-1]
                pipeline_type = "P2d"
                config = f"{noise_type}_{noise_level}"
        
        elif pipeline_name.startswith("p3_"):
            # P3: Quantum classifier
            # e.g., p3_no_noise or p3_bit_flip_0.05
            if "no_noise" in pipeline_name:
                pipeline_type = "P3"
                config = "no_noise"
            else:
                parts = pipeline_name.split('_')
                if len(parts) >= 3:
                    noise_type = parts[1]
                    if len(parts) >= 4:
                        noise_type = f"{parts[1]}_{parts[2]}" if parts[2] not in ["0.01", "0.03", "0.05", "0.1", "0.2", "0.3"] else parts[1]
                    noise_level = parts[-1]
                    pipeline_type = "P3"
                    config = f"{noise_type}_{noise_level}"
        else:
            continue
        
        # Load all runs
        for run_dir in sorted(pipeline_dir.glob("run_*")):
            metrics_file = run_dir / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                run_id = int(run_dir.name.split('_')[1])
                
                row = {
                    'pipeline_type': pipeline_type,
                    'pipeline_name': pipeline_name,
                    'config': config,
                    'run_id': run_id,
                    'auc': metrics.get('auc', np.nan),
                    'accuracy': metrics.get('accuracy', np.nan),
                    'final_train_loss': metrics.get('final_train_loss', np.nan),
                    'best_val_loss': metrics.get('best_val_loss', np.nan),
                    'epochs_trained': metrics.get('epochs_trained', np.nan)
                }
                
                all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    # Add parsed noise info for P2d and P3
    def parse_noise_info(row):
        if row['pipeline_type'] in ['P2d', 'P3'] and row['config'] != 'no_noise':
            parts = row['config'].split('_')
            if len(parts) >= 2:
                # Handle amp_damp, bit_flip, depol
                if parts[0] in ['amp', 'bit']:
                    noise_type = f"{parts[0]}_{parts[1]}"
                    noise_level = float(parts[2]) if len(parts) > 2 else np.nan
                else:
                    noise_type = parts[0]
                    noise_level = float(parts[1]) if len(parts) > 1 else np.nan
                return pd.Series({'noise_type': noise_type, 'noise_level': noise_level})
        return pd.Series({'noise_type': None, 'noise_level': None})
    
    df[['noise_type', 'noise_level']] = df.apply(parse_noise_info, axis=1)
    
    return df


def test_1_p1_vs_p3(df: pd.DataFrame, output_dir: Path):
    """
    Test 1: P1 (Classical) vs P3 (Quantum Classifier)
    
    Tests both:
    - 1a: No noise versions only
    - 1b: Best configuration (including noise)
    """
    print("\n" + "="*80)
    print("TEST 1: Classical vs Quantum")
    print("="*80)
    
    # ===== TEST 1A: NO NOISE =====
    print("\n--- Test 1a: No Noise Comparison ---")
    
    # Get best P1 config (highest mean AUC) among no-noise configs
    p1_data = df[df['pipeline_type'] == 'P1'].copy()
    p1_means = p1_data.groupby('config')['auc'].mean()
    best_p1_config = p1_means.idxmax()
    
    p1_best_no_noise = p1_data[p1_data['config'] == best_p1_config]['auc'].values
    
    # Get P3 no-noise
    p3_no_noise = df[(df['pipeline_type'] == 'P3') & (df['config'] == 'no_noise')]['auc'].values
    
    results_1a = None
    if len(p1_best_no_noise) > 0 and len(p3_no_noise) > 0:
        # Statistics
        p1_mean, p1_std = np.mean(p1_best_no_noise), np.std(p1_best_no_noise, ddof=1)
        p3_mean, p3_std = np.mean(p3_no_noise), np.std(p3_no_noise, ddof=1)
        
        t_stat, p_value = ttest_ind(p1_best_no_noise, p3_no_noise)
        
        pooled_std = np.sqrt(((len(p1_best_no_noise)-1)*p1_std**2 + (len(p3_no_noise)-1)*p3_std**2) / 
                            (len(p1_best_no_noise) + len(p3_no_noise) - 2))
        cohens_d = (p1_mean - p3_mean) / pooled_std
        
        dropout_label = f"Dropout p={best_p1_config.split('_')[1]}"
        
        print(f"P1 ({dropout_label}): {p1_mean:.4f} ± {p1_std:.4f} (n={len(p1_best_no_noise)})")
        print(f"P3 (No Noise):       {p3_mean:.4f} ± {p3_std:.4f} (n={len(p3_no_noise)})")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
        print(f"Cohen's d: {cohens_d:.4f}")
        
        # Plot Test 1a
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_to_plot = [p1_best_no_noise, p3_no_noise]
        labels = [f'P1 Classical\n({dropout_label})', 'P3 Quantum\n(No Noise)']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=50, color='darkblue')
        
        sig_text = f"p = {p_value:.4f}" if p_value >= 0.001 else f"p < 0.001"
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'test1a_p1_vs_p3_no_noise.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        results_1a = {
            'test': 'P1 vs P3 (no noise)',
            'p1_config': best_p1_config,
            'p1_mean': p1_mean,
            'p1_std': p1_std,
            'p1_n': len(p1_best_no_noise),
            'p3_mean': p3_mean,
            'p3_std': p3_std,
            'p3_n': len(p3_no_noise),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    # ===== TEST 1B: BEST (INCLUDING NOISE) =====
    print("\n--- Test 1b: Best Configuration (including noise) ---")
    
    # Find best P1 (any config)
    p1_all_means = p1_data.groupby('config')['auc'].mean()
    best_p1_config_all = p1_all_means.idxmax()
    p1_best_all = p1_data[p1_data['config'] == best_p1_config_all]['auc'].values
    
    # Find best P3 (any config including noise)
    p3_data = df[df['pipeline_type'] == 'P3'].copy()
    p3_all_means = p3_data.groupby('config')['auc'].mean()
    best_p3_config = p3_all_means.idxmax()
    p3_best_all = p3_data[p3_data['config'] == best_p3_config]['auc'].values
    
    results_1b = None
    if len(p1_best_all) > 0 and len(p3_best_all) > 0:
        p1_mean_all, p1_std_all = np.mean(p1_best_all), np.std(p1_best_all, ddof=1)
        p3_mean_all, p3_std_all = np.mean(p3_best_all), np.std(p3_best_all, ddof=1)
        
        t_stat_all, p_value_all = ttest_ind(p1_best_all, p3_best_all)
        
        pooled_std_all = np.sqrt(((len(p1_best_all)-1)*p1_std_all**2 + (len(p3_best_all)-1)*p3_std_all**2) / 
                                (len(p1_best_all) + len(p3_best_all) - 2))
        cohens_d_all = (p1_mean_all - p3_mean_all) / pooled_std_all
        
        # Format labels
        p1_label_all = f"Dropout p={best_p1_config_all.split('_')[1]}"
        if best_p3_config == 'no_noise':
            p3_label_all = "No Noise"
        else:
            # Parse noise config
            parts = best_p3_config.split('_')
            if len(parts) >= 2:
                noise_type = parts[0].replace('_', ' ').title()
                if parts[0] in ['amp', 'bit']:
                    noise_type = f"{parts[0].title()} {parts[1].title()}"
                    noise_level = parts[2] if len(parts) > 2 else "?"
                else:
                    noise_level = parts[1] if len(parts) > 1 else "?"
                p3_label_all = f"{noise_type} p={noise_level}"
            else:
                p3_label_all = best_p3_config
        
        print(f"P1 Best ({p1_label_all}): {p1_mean_all:.4f} ± {p1_std_all:.4f} (n={len(p1_best_all)})")
        print(f"P3 Best ({p3_label_all}): {p3_mean_all:.4f} ± {p3_std_all:.4f} (n={len(p3_best_all)})")
        print(f"t-statistic: {t_stat_all:.4f}, p-value: {p_value_all:.4e}")
        print(f"Cohen's d: {cohens_d_all:.4f}")
        
        # Plot Test 1b
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_to_plot = [p1_best_all, p3_best_all]
        labels = [f'P1 Classical\n({p1_label_all})', f'P3 Quantum\n({p3_label_all})']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightcoral', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=50, color='darkred')
        
        sig_text = f"p = {p_value_all:.4f}" if p_value_all >= 0.001 else f"p < 0.001"
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'test1b_p1_vs_p3_best.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        results_1b = {
            'test': 'P1 vs P3 (best)',
            'p1_config': best_p1_config_all,
            'p1_mean': p1_mean_all,
            'p1_std': p1_std_all,
            'p1_n': len(p1_best_all),
            'p3_config': best_p3_config,
            'p3_mean': p3_mean_all,
            'p3_std': p3_std_all,
            'p3_n': len(p3_best_all),
            't_statistic': t_stat_all,
            'p_value': p_value_all,
            'cohens_d': cohens_d_all,
            'significant': p_value_all < 0.05
        }
    
    # Return combined results
    return {'test_1a': results_1a, 'test_1b': results_1b}


def test_2_p2_variants(df: pd.DataFrame, output_dir: Path):
    """
    Test 2: Compare P2a vs P2b (feature extraction methods, no noise)
    Note: P2c (Pauli) excluded from analysis
    """
    print("\n" + "="*80)
    print("TEST 2: P2a vs P2b (Feature Extraction Methods)")
    print("="*80)
    
    p2a_data = df[df['pipeline_type'] == 'P2a']['auc'].values
    p2b_data = df[df['pipeline_type'] == 'P2b']['auc'].values
    
    groups = []
    labels = []
    
    if len(p2a_data) > 0:
        groups.append(p2a_data)
        labels.append('P2a\n(Probabilities)')
    if len(p2b_data) > 0:
        groups.append(p2b_data)
        labels.append('P2b\n(Complex Amplitudes)')
    
    if len(groups) < 2:
        print("⚠️  Insufficient data for Test 2")
        return None
    
    # T-test (only 2 groups now)
    t_stat, p_value = ttest_ind(groups[0], groups[1])
    
    # Summary stats
    stats_summary = []
    for i, (data, label) in enumerate(zip(groups, labels)):
        stats_summary.append({
            'method': label.replace('\n', ' '),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'n': len(data)
        })
    
    print(f"\nt-test: t={t_stat:.4f}, p={p_value:.4e}")
    print("\nSummary:")
    for stat in stats_summary:
        print(f"  {stat['method']}: {stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['n']})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    # Add individual points
    for i, data in enumerate(groups):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=50, color='darkgreen')
    
    # Add p-value to text box
    sig_text = f"p = {p_value:.4f}" if p_value >= 0.001 else f"p < 0.001"
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test2_p2_variants.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'test': 'P2 variants',
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'summary': stats_summary
    }
    
    return results


def test_3_p2d_noise_sweet_spot(df: pd.DataFrame, output_dir: Path):
    """
    Test 3: P2d Noise Sweet Spot - Find optimal noise level for each noise type
    """
    print("\n" + "="*80)
    print("TEST 3: P2d Noise Sweet Spot Analysis")
    print("="*80)
    
    p2d_data = df[df['pipeline_type'] == 'P2d'].copy()
    
    if len(p2d_data) == 0:
        print("⚠️  No P2d data found")
        return None
    
    noise_types = p2d_data['noise_type'].dropna().unique()
    
    results = {}
    
    fig, axes = plt.subplots(1, len(noise_types), figsize=(6*len(noise_types), 5))
    if len(noise_types) == 1:
        axes = [axes]
    
    for idx, noise_type in enumerate(sorted(noise_types)):
        noise_data = p2d_data[p2d_data['noise_type'] == noise_type].copy()
        
        # Group by noise level
        grouped = noise_data.groupby('noise_level')['auc'].agg(['mean', 'std', 'count']).reset_index()
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        
        # Find optimal
        optimal_idx = grouped['mean'].idxmax()
        optimal_level = grouped.loc[optimal_idx, 'noise_level']
        optimal_auc = grouped.loc[optimal_idx, 'mean']
        
        # ANOVA across noise levels
        noise_levels = sorted(noise_data['noise_level'].unique())
        groups = [noise_data[noise_data['noise_level'] == level]['auc'].values 
                 for level in noise_levels]
        
        f_stat, p_value = f_oneway(*groups)
        
        results[noise_type] = {
            'noise_type': noise_type,
            'optimal_level': optimal_level,
            'optimal_auc': optimal_auc,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print(f"\n{noise_type}:")
        print(f"  Optimal: p={optimal_level} (AUC={optimal_auc:.4f})")
        print(f"  ANOVA: F={f_stat:.4f}, p={p_value:.4e}")
        
        # Plot
        ax = axes[idx]
        ax.errorbar(grouped['noise_level'], grouped['mean'], yerr=grouped['sem'],
                   fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        # Highlight optimal
        ax.plot(optimal_level, optimal_auc, 'g*', markersize=20, 
               label=f'Optimal: p={optimal_level}', zorder=10)
        
        # Add noise type as subplot label
        noise_label = noise_type.replace('_', ' ').title()
        ax.text(0.5, 0.95, noise_label, transform=ax.transAxes,
               fontsize=13, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Noise Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test3_p2d_noise_sweet_spot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def test_4_p1_dropout_sweep(df: pd.DataFrame, output_dir: Path):
    """
    Test 4: P1 Dropout Sweep - Find optimal dropout
    """
    print("\n" + "="*80)
    print("TEST 4: P1 Dropout Sweep")
    print("="*80)
    
    p1_data = df[df['pipeline_type'] == 'P1'].copy()
    
    if len(p1_data) == 0:
        print("⚠️  No P1 data found")
        return None
    
    # Extract dropout values from config
    p1_data['dropout'] = p1_data['config'].str.extract(r'dropout_(\d+\.\d+)')[0].astype(float)
    
    # Group by dropout
    grouped = p1_data.groupby('dropout')['auc'].agg(['mean', 'std', 'count']).reset_index()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    
    # Find optimal
    optimal_idx = grouped['mean'].idxmax()
    optimal_dropout = grouped.loc[optimal_idx, 'dropout']
    optimal_auc = grouped.loc[optimal_idx, 'mean']
    
    # ANOVA
    dropout_levels = sorted(p1_data['dropout'].unique())
    groups = [p1_data[p1_data['dropout'] == d]['auc'].values for d in dropout_levels]
    f_stat, p_value = f_oneway(*groups)
    
    # Format dropout label
    dropout_str = f"p={optimal_dropout}" if optimal_dropout != 0.0 else "p=0"
    
    print(f"\nOptimal dropout: {dropout_str} (AUC={optimal_auc:.4f})")
    print(f"ANOVA: F={f_stat:.4f}, p={p_value:.4e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(grouped['dropout'], grouped['mean'], yerr=grouped['sem'],
               fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10,
               color='blue', label='Mean ± SEM')
    
    # Highlight optimal
    ax.plot(optimal_dropout, optimal_auc, 'r*', markersize=25,
           label=f'Optimal: {dropout_str}', zorder=10)
    
    # Add p-value
    sig_text = f"ANOVA p = {p_value:.4f}" if p_value >= 0.001 else f"ANOVA p < 0.001"
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Dropout Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test4_p1_dropout.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'test': 'P1 dropout',
        'optimal_dropout': optimal_dropout,
        'optimal_auc': optimal_auc,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'summary': grouped.to_dict('records')
    }
    
    return results


def test_5_best_p2_vs_p3(df: pd.DataFrame, output_dir: Path):
    """
    Test 5: Best P2 variant vs P3
    """
    print("\n" + "="*80)
    print("TEST 5: Best P2 vs P3")
    print("="*80)
    
    # Find best P2 variant (among P2a, P2b, P2c - no noise)
    p2_variants = df[df['pipeline_type'].isin(['P2a', 'P2b', 'P2c'])].copy()
    
    if len(p2_variants) == 0:
        print("⚠️  No P2 data found")
        return None
    
    p2_means = p2_variants.groupby('pipeline_type')['auc'].mean()
    best_p2_type = p2_means.idxmax()
    best_p2_auc = p2_means.max()
    
    best_p2_data = p2_variants[p2_variants['pipeline_type'] == best_p2_type]['auc'].values
    
    # Get P3 no-noise
    p3_data = df[(df['pipeline_type'] == 'P3') & (df['config'] == 'no_noise')]['auc'].values
    
    if len(p3_data) == 0:
        print("⚠️  No P3 data found")
        return None
    
    # Statistics
    p2_mean, p2_std = np.mean(best_p2_data), np.std(best_p2_data, ddof=1)
    p3_mean, p3_std = np.mean(p3_data), np.std(p3_data, ddof=1)
    
    t_stat, p_value = ttest_ind(best_p2_data, p3_data)
    
    # Effect size
    pooled_std = np.sqrt(((len(best_p2_data)-1)*p2_std**2 + (len(p3_data)-1)*p3_std**2) / 
                        (len(best_p2_data) + len(p3_data) - 2))
    cohens_d = (p2_mean - p3_mean) / pooled_std
    
    print(f"\nBest P2: {best_p2_type} ({p2_mean:.4f} ± {p2_std:.4f}, n={len(best_p2_data)})")
    print(f"P3:                  ({p3_mean:.4f} ± {p3_std:.4f}, n={len(p3_data)})")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = [best_p2_data, p3_data]
    labels = [f'{best_p2_type}\n(Feature Extraction)', 'P3\n(Direct Classification)']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2))
    
    # Add points
    for i, data in enumerate(data_to_plot):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=50, color='darkred')
    
    # Add p-value
    sig_text = f"p = {p_value:.4f}" if p_value >= 0.001 else f"p < 0.001"
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test5_best_p2_vs_p3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'test': 'Best P2 vs P3',
        'best_p2_type': best_p2_type,
        'p2_mean': p2_mean,
        'p2_std': p2_std,
        'p2_n': len(best_p2_data),
        'p3_mean': p3_mean,
        'p3_std': p3_std,
        'p3_n': len(p3_data),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }
    
    return results


def test_6_noise_type_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Test 6: Compare different noise types at their optimal levels (P2d)
    """
    print("\n" + "="*80)
    print("TEST 6: Noise Type Comparison (P2d at optimal levels)")
    print("="*80)
    
    p2d_data = df[df['pipeline_type'] == 'P2d'].copy()
    
    if len(p2d_data) == 0:
        print("⚠️  No P2d data found")
        return None
    
    # Find optimal level for each noise type
    noise_types = p2d_data['noise_type'].dropna().unique()
    
    optimal_data = []
    optimal_configs = {}
    
    for noise_type in noise_types:
        noise_subset = p2d_data[p2d_data['noise_type'] == noise_type]
        means = noise_subset.groupby('noise_level')['auc'].mean()
        optimal_level = means.idxmax()
        optimal_auc = means.max()
        
        optimal_subset = noise_subset[noise_subset['noise_level'] == optimal_level]['auc'].values
        optimal_data.append(optimal_subset)
        optimal_configs[noise_type] = {
            'optimal_level': optimal_level,
            'mean_auc': optimal_auc,
            'data': optimal_subset
        }
    
    labels = [f"{nt.replace('_', ' ').title()}\n(p={optimal_configs[nt]['optimal_level']})" 
             for nt in sorted(noise_types)]
    
    # ANOVA across noise types at optimal levels
    f_stat, p_value = f_oneway(*optimal_data)
    
    print(f"\nANOVA across noise types: F={f_stat:.4f}, p={p_value:.4e}")
    
    for nt in sorted(noise_types):
        config = optimal_configs[nt]
        print(f"  {nt}: p={config['optimal_level']}, AUC={config['mean_auc']:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(optimal_data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightyellow', alpha=0.7),
                    medianprops=dict(color='orange', linewidth=2))
    
    # Add points
    for i, data in enumerate(optimal_data):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=50, color='darkorange')
    
    ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test6_noise_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'test': 'Noise type comparison',
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'optimal_configs': {k: {'optimal_level': v['optimal_level'], 
                               'mean_auc': v['mean_auc']} 
                          for k, v in optimal_configs.items()}
    }
    
    return results


def test_7_best_classical_vs_best_quantum_overall(df: pd.DataFrame, output_dir: Path):
    """
    Test 7: ULTIMATE SHOWDOWN - Best Classical vs Best Quantum (all variants, all configs)
    
    Compares:
    - Best P1 (any dropout config)
    vs
    - Best Quantum OVERALL (best among ALL P2 variants with/without noise, P3 with/without noise)
    """
    print("\n" + "="*80)
    print("TEST 7: ULTIMATE SHOWDOWN - Best Classical vs Best Quantum Overall")
    print("="*80)
    
    # Find best P1 configuration
    p1_data = df[df['pipeline_type'] == 'P1'].copy()
    p1_means = p1_data.groupby('config')['auc'].mean()
    best_p1_config = p1_means.idxmax()
    best_p1_auc = p1_means.max()
    p1_best = p1_data[p1_data['config'] == best_p1_config]['auc'].values
    
    # Find best quantum configuration across ALL quantum pipelines
    quantum_data = df[df['pipeline_type'].isin(['P2a', 'P2b', 'P2d', 'P3'])].copy()
    
    if len(quantum_data) == 0:
        print("⚠️  No quantum data found")
        return None
    
    # Group by pipeline_type and config, find best overall
    quantum_means = quantum_data.groupby(['pipeline_type', 'config'])['auc'].mean()
    best_quantum_idx = quantum_means.idxmax()
    best_quantum_type = best_quantum_idx[0]
    best_quantum_config = best_quantum_idx[1]
    best_quantum_auc = quantum_means.max()
    
    quantum_best = quantum_data[
        (quantum_data['pipeline_type'] == best_quantum_type) & 
        (quantum_data['config'] == best_quantum_config)
    ]['auc'].values
    
    if len(p1_best) == 0 or len(quantum_best) == 0:
        print("⚠️  Insufficient data for Test 7")
        return None
    
    # Statistics
    p1_mean, p1_std = np.mean(p1_best), np.std(p1_best, ddof=1)
    quantum_mean, quantum_std = np.mean(quantum_best), np.std(quantum_best, ddof=1)
    
    t_stat, p_value = ttest_ind(p1_best, quantum_best)
    
    pooled_std = np.sqrt(((len(p1_best)-1)*p1_std**2 + (len(quantum_best)-1)*quantum_std**2) / 
                        (len(p1_best) + len(quantum_best) - 2))
    cohens_d = (p1_mean - quantum_mean) / pooled_std
    
    # Format labels
    p1_label = f"Dropout p={best_p1_config.split('_')[1]}"
    
    # Format quantum label
    if best_quantum_config == 'no_noise':
        quantum_config_label = "No Noise"
    elif best_quantum_config == 'probabilities':
        quantum_config_label = "Probabilities"
    elif best_quantum_config == 'complex_amplitudes':
        quantum_config_label = "Complex Amplitudes"
    else:
        # Parse noise config
        parts = best_quantum_config.split('_')
        if len(parts) >= 2:
            noise_type = parts[0].replace('_', ' ').title()
            if parts[0] in ['amp', 'bit']:
                noise_type = f"{parts[0].title()} {parts[1].title()}"
                noise_level = parts[2] if len(parts) > 2 else "?"
            else:
                noise_level = parts[1] if len(parts) > 1 else "?"
            quantum_config_label = f"{noise_type} p={noise_level}"
        else:
            quantum_config_label = best_quantum_config
    
    quantum_full_label = f"{best_quantum_type} ({quantum_config_label})"
    
    print(f"\n{'='*80}")
    print(f"BEST CLASSICAL:")
    print(f"  P1 ({p1_label}): {p1_mean:.4f} ± {p1_std:.4f} (n={len(p1_best)})")
    print(f"\nBEST QUANTUM:")
    print(f"  {quantum_full_label}: {quantum_mean:.4f} ± {quantum_std:.4f} (n={len(quantum_best)})")
    print(f"\nCOMPARISON:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Significant: {'YES ✓' if p_value < 0.05 else 'NO'}")
    
    winner = "Classical" if p1_mean > quantum_mean else "Quantum"
    print(f"\n🏆 WINNER: {winner}")
    print(f"{'='*80}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = [p1_best, quantum_best]
    
    # Format labels more compactly
    p1_short = p1_label.replace("Dropout p=", "p=")
    quantum_short = quantum_full_label
    
    # For long labels, split into multiple lines intelligently
    if len(quantum_full_label) > 25:
        # Split at parenthesis for better formatting
        if '(' in quantum_full_label:
            parts = quantum_full_label.split('(')
            quantum_short = f"{parts[0].strip()}\n({parts[1]}"
        else:
            quantum_short = quantum_full_label
    
    labels = [f'Best Classical\nP1 ({p1_short})', f'Best Quantum\n{quantum_short}']
    
    # Use gold colors for the ultimate showdown
    colors = ['gold', 'mediumorchid']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    medianprops=dict(color='darkred', linewidth=3))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add points
    for i, (data, color) in enumerate(zip(data_to_plot, colors)):
        x = np.random.normal(i+1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=60, color=color, edgecolors='black', linewidth=0.5)
    
    # Add p-value
    sig_text = f"p = {p_value:.4f}" if p_value >= 0.001 else f"p < 0.001"
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=2))
    
    ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits to give boxes more room (don't auto-scale too tightly)
    y_min = min(np.min(p1_best), np.min(quantum_best))
    y_max = max(np.max(p1_best), np.max(quantum_best))
    y_range = y_max - y_min
    
    # Add padding (20% on each side)
    if y_range < 0.1:  # If very tight range, ensure minimum spread
        y_range = max(y_range, 0.05)
    
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.3 * y_range)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test7_ultimate_showdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    results = {
        'test': 'Ultimate Showdown',
        'p1_config': best_p1_config,
        'p1_mean': p1_mean,
        'p1_std': p1_std,
        'p1_n': len(p1_best),
        'quantum_type': best_quantum_type,
        'quantum_config': best_quantum_config,
        'quantum_mean': quantum_mean,
        'quantum_std': quantum_std,
        'quantum_n': len(quantum_best),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'winner': winner
    }
    
    return results


def create_summary_report(all_results: dict, output_dir: Path):
    """Create markdown summary report."""
    
    report = []
    report.append("# Statistical Analysis Summary\n")
    report.append("="*80 + "\n")
    
    for test_name, result in all_results.items():
        if result is None:
            continue
        
        report.append(f"\n## {test_name}\n")
        
        if 'p_value' in result:
            sig = "✓ SIGNIFICANT" if result['significant'] else "✗ Not significant"
            report.append(f"**Result:** {sig} (p={result['p_value']:.4e})\n")
        
        if test_name == "Test 1":
            if 'test_1a' in result and result['test_1a']:
                report.append("\n### Test 1a: No Noise Comparison\n")
                r = result['test_1a']
                report.append(f"- P1 ({r['p1_config']}): {r['p1_mean']:.4f} ± {r['p1_std']:.4f}\n")
                report.append(f"- P3 (No Noise): {r['p3_mean']:.4f} ± {r['p3_std']:.4f}\n")
                report.append(f"- Cohen's d: {r['cohens_d']:.4f}\n")
            
            if 'test_1b' in result and result['test_1b']:
                report.append("\n### Test 1b: Best Configuration\n")
                r = result['test_1b']
                report.append(f"- P1 Best ({r['p1_config']}): {r['p1_mean']:.4f} ± {r['p1_std']:.4f}\n")
                report.append(f"- P3 Best ({r['p3_config']}): {r['p3_mean']:.4f} ± {r['p3_std']:.4f}\n")
                report.append(f"- Cohen's d: {r['cohens_d']:.4f}\n")
        
        elif test_name == "Test 2":
            report.append("\n**Summary:**\n")
            for s in result['summary']:
                report.append(f"- {s['method']}: {s['mean']:.4f} ± {s['std']:.4f}\n")
        
        elif test_name == "Test 3":
            report.append("\n**Optimal noise levels:**\n")
            for noise_type, res in result.items():
                if isinstance(res, dict):
                    report.append(f"- {noise_type}: p={res['optimal_level']}, AUC={res['optimal_auc']:.4f}\n")
        
        elif test_name == "Test 4":
            report.append(f"- Optimal dropout: {result['optimal_dropout']}\n")
            report.append(f"- AUC: {result['optimal_auc']:.4f}\n")
        
        elif test_name == "Test 5":
            report.append(f"- Best P2: {result['best_p2_type']} ({result['p2_mean']:.4f} ± {result['p2_std']:.4f})\n")
            report.append(f"- P3: {result['p3_mean']:.4f} ± {result['p3_std']:.4f}\n")
            report.append(f"- Cohen's d: {result['cohens_d']:.4f}\n")
        
        elif test_name == "Test 6":
            report.append("\n**Optimal configurations:**\n")
            for nt, config in result['optimal_configs'].items():
                report.append(f"- {nt}: p={config['optimal_level']}, AUC={config['mean_auc']:.4f}\n")
        
        elif test_name == "Test 7":
            report.append(f"- Best Classical: P1 ({result['p1_config']}) - {result['p1_mean']:.4f} ± {result['p1_std']:.4f}\n")
            report.append(f"- Best Quantum: {result['quantum_type']} ({result['quantum_config']}) - {result['quantum_mean']:.4f} ± {result['quantum_std']:.4f}\n")
            report.append(f"- Cohen's d: {result['cohens_d']:.4f}\n")
            report.append(f"- **🏆 WINNER: {result['winner']}**\n")
    
    report_text = "".join(report)
    
    with open(output_dir / 'statistical_summary.md', 'w') as f:
        f.write(report_text)
    
    print(f"\n💾 Summary report saved to: {output_dir / 'statistical_summary.md'}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of quantum ML experiments")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory for analysis')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " STATISTICAL ANALYSIS - QUANTUM ML EXPERIMENTS".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    # Load all data
    print(f"\n📥 Loading data from: {results_dir}")
    df = load_all_metrics(results_dir)
    
    if df.empty:
        print("❌ No data found!")
        return
    
    print(f"✅ Loaded {len(df)} experiments")
    print(f"   Pipelines: {sorted(df['pipeline_type'].unique())}")
    print(f"   Total runs: {df['run_id'].max()}")
    
    # Save raw data
    df.to_csv(output_dir / 'all_metrics.csv', index=False)
    print(f"💾 Saved raw data to: {output_dir / 'all_metrics.csv'}")
    
    # Run all tests
    all_results = {}
    
    all_results['Test 1'] = test_1_p1_vs_p3(df, output_dir)
    all_results['Test 2'] = test_2_p2_variants(df, output_dir)
    all_results['Test 3'] = test_3_p2d_noise_sweet_spot(df, output_dir)
    all_results['Test 4'] = test_4_p1_dropout_sweep(df, output_dir)
    all_results['Test 5'] = test_5_best_p2_vs_p3(df, output_dir)
    all_results['Test 6'] = test_6_noise_type_comparison(df, output_dir)
    all_results['Test 7'] = test_7_best_classical_vs_best_quantum_overall(df, output_dir)
    
    # Save results
    import json
    with open(output_dir / 'statistical_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for k, v in all_results.items():
            if v is not None:
                json_results[k] = v
        json.dump(json_results, f, indent=4, default=str)
    
    print(f"\n💾 Saved results to: {output_dir / 'statistical_results.json'}")
    
    # Create summary
    create_summary_report(all_results, output_dir)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " ANALYSIS COMPLETE!".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print(f"\n📁 All outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
    
# python src/statistical_analysis.py \
#     --results_dir /scratch90/chris/qc2025_final_proj/results_pneumoniamnist_n10_dual_lr \
#     --output_dir /scratch90/chris/qc2025_final_proj/analysis_pneumoniamnist_n10_dual_lr