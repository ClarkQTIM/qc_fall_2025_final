"""
visualize_analysis.py

Comprehensive visualizations for Quantum-Classical Hybrid Study:
- Pipeline 1: Classical + Dropout
- Pipeline 2a: Probabilities (|αᵢ|²) - no noise
- Pipeline 2b: Complex Amplitudes (Re + Im) - no noise  
- Pipeline 2d: Density Matrix (full ρ) - WITH noise
- Pipeline 3: Quantum Classifier + Noise

Note: P2c (Pauli Expectations) was explored but excluded from final analysis.

Usage:
    python visualize_analysis.py \
        --results_dir /path/to/results \
        --output_dir /path/to/analysis

Generates comprehensive visualizations for all pipelines.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_all_results(results_dir):
    """Load all experiment results from directory structure."""
    results = []
    results_path = Path(results_dir)
    
    print("📂 Scanning for experiment results...")
    
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Parse experiment type
        if exp_name.startswith("p1_classical_dropout_"):
            pipeline = "P1"
            method = "Classical"
            transform = None
            noise_type = "Dropout"
            level = float(exp_name.split("_")[-1])
            
        elif exp_name == "p2a_probabilities":
            pipeline = "P2a"
            method = "Quantum Feature"
            transform = "probabilities"
            noise_type = "No Noise"
            level = 0.0
            
        elif exp_name == "p2b_complex_amplitudes":
            pipeline = "P2b"
            method = "Quantum Feature"
            transform = "complex_amplitudes"
            noise_type = "No Noise"
            level = 0.0
            
        elif exp_name == "p2c_pauli_expectations":
            pipeline = "P2c"
            method = "Quantum Feature"
            transform = "pauli"
            noise_type = "No Noise"
            level = 0.0
        
        elif exp_name.startswith("p2d_density_matrix_bit_flip_"):
            pipeline = "P2d"
            method = "Quantum Feature"
            transform = "density_matrix"
            noise_type = "Bit Flip"
            level = float(exp_name.split("_")[-1])
        
        elif exp_name.startswith("p2d_density_matrix_amp_damp_"):
            pipeline = "P2d"
            method = "Quantum Feature"
            transform = "density_matrix"
            noise_type = "Amp Damping"
            level = float(exp_name.split("_")[-1])
        
        elif exp_name.startswith("p2d_density_matrix_depol_"):
            pipeline = "P2d"
            method = "Quantum Feature"
            transform = "density_matrix"
            noise_type = "Depolarizing"
            level = float(exp_name.split("_")[-1])
            
        elif exp_name.startswith("p3_") or exp_name.startswith("p3_quantum_class"):
            # Parse P3 experiments
            if exp_name in ["p3_quantum_class_no_noise", "p3_no_noise"]:
                pipeline = "P3"
                method = "Quantum Classifier"
                transform = None
                noise_type = "No Noise"
                level = 0.0
            elif "_bit_flip_" in exp_name or exp_name.startswith("p3_bit_flip_"):
                pipeline = "P3"
                method = "Quantum Classifier"
                transform = None
                noise_type = "Bit Flip"
                level = float(exp_name.split("_")[-1])
            elif "_amp_damp_" in exp_name or exp_name.startswith("p3_amp_damp_"):
                pipeline = "P3"
                method = "Quantum Classifier"
                transform = None
                noise_type = "Amp Damping"
                level = float(exp_name.split("_")[-1])
            elif "_depol_" in exp_name or exp_name.startswith("p3_depol_"):
                pipeline = "P3"
                method = "Quantum Classifier"
                transform = None
                noise_type = "Depolarizing"
                level = float(exp_name.split("_")[-1])
            else:
                print(f"⚠️  Skipping unknown P3 directory: {exp_name}")
                continue
        else:
            print(f"⚠️  Skipping unknown directory: {exp_name}")
            continue
        
        # Load all runs for this experiment
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            
            run_id = int(run_dir.name.split("_")[-1])
            metrics_file = run_dir / "metrics.json"
            
            if not metrics_file.exists():
                print(f"⚠️  Missing metrics: {metrics_file}")
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            results.append({
                'pipeline': pipeline,
                'method': method,
                'transform': transform,
                'noise_type': noise_type,
                'level': level,
                'run': run_id,
                'test_auc': metrics.get('auc', np.nan),
                'test_accuracy': metrics.get('accuracy', np.nan),
                'train_loss': metrics.get('final_train_loss', np.nan),
                'val_loss': metrics.get('best_val_loss', np.nan),
                'epochs_trained': metrics.get('epochs_trained', np.nan)
            })
    
    df = pd.DataFrame(results)
    print(f"✅ Loaded {len(df)} experiment results")
    print(f"   Pipelines: {sorted(df['pipeline'].unique())}")
    print(f"   Methods: {df['method'].unique()}")
    print(f"   Transforms: {df['transform'].dropna().unique()}")
    print(f"   Noise types: {df['noise_type'].unique()}")
    
    return df


def plot_p2_no_noise_comparison(df, output_dir):
    """
    P2 No-Noise Comparison: P2a vs P2b vs P2c
    
    Compares pure state feature extraction methods.
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get data for each transform (no noise only)
    transforms_data = [
        ('P2a', 'Probabilities\n(|αᵢ|²)\n512 features', 'gray'),
        ('P2b', 'Complex Amplitudes\n(Re + Im)\n1024 features', 'blue'),
        ('P2c', 'Pauli Expectations\n(⟨X⟩,⟨Y⟩,⟨Z⟩)\n27 features', 'orange')
    ]
    
    means = []
    stds = []
    labels = []
    colors = []
    
    for pipeline, label, color in transforms_data:
        data = df[(df['pipeline'] == pipeline) & (df['noise_type'] == 'No Noise')]['test_auc']
        if len(data) > 0:
            means.append(data.mean())
            stds.append(data.std())
            labels.append(label)
            colors.append(color)
        else:
            print(f"⚠️  No data for {pipeline}")
    
    if len(means) == 0:
        print("❌ No P2 no-noise data found!")
        return
    
    # Create bar chart
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight best
    if len(means) > 0:
        best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.set_title('Pipeline 2 (Pure States): Feature Extraction Method Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0.4, 0.95])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for P2a (baseline)
    if len(means) > 0:
        p2a_mean = means[0]
        ax.axhline(p2a_mean, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                  label=f'P2a Baseline (No Phase Info): {p2a_mean:.3f}')
        ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_p2_no_noise_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 1_p2_no_noise_comparison.png")


def plot_p2d_vs_noise(df, output_dir):
    """
    P2d Performance vs Noise Level
    
    Shows how density matrix features degrade with increasing noise.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    noise_types = ['Bit Flip', 'Amp Damping', 'Depolarizing']
    colors = ['red', 'green', 'purple']
    
    p2d_df = df[df['pipeline'] == 'P2d']
    
    if len(p2d_df) == 0:
        print("⚠️  No P2d data found for noise analysis")
        return
    
    for idx, (noise_type, color) in enumerate(zip(noise_types, colors)):
        ax = axes[idx]
        data = p2d_df[p2d_df['noise_type'] == noise_type]
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No data for {noise_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Aggregate by noise level
        agg = data.groupby('level')['test_auc'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
        agg = agg.sort_values('level')
        
        # Plot
        ax.plot(agg['level'], agg['mean'], 
                color=color, marker='o', linestyle='-',
                linewidth=2.5, markersize=8, label=noise_type)
        
        ax.fill_between(agg['level'], 
                        agg['mean'] - agg['sem'], 
                        agg['mean'] + agg['sem'],
                        color=color, alpha=0.2)
        
        # Mark optimal
        optimal_idx = agg['mean'].idxmax()
        optimal_level = agg.loc[optimal_idx, 'level']
        optimal_auc = agg.loc[optimal_idx, 'mean']
        ax.scatter([optimal_level], [optimal_auc], 
                  color='gold', s=300, marker='*', 
                  edgecolors='black', linewidths=2, zorder=4,
                  label=f'Optimal: p={optimal_level}')
        
        ax.set_xlabel('Noise Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
        ax.set_title(f'P2d: {noise_type}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 0.95])
    
    plt.suptitle('Pipeline 2d (Mixed States): Density Matrix Performance vs Noise', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_p2d_vs_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 2_p2d_vs_noise.png")


def plot_p2b_vs_p2d_comparison(df, output_dir):
    """
    P2b vs P2d Direct Comparison
    
    Compares the best no-noise approach (P2b) with best noise approach (P2d).
    """
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # P2b: No noise, complex amplitudes
    p2b_data = df[(df['pipeline'] == 'P2b') & (df['noise_type'] == 'No Noise')]['test_auc']
    
    # P2d: Best noise configuration across all noise types
    p2d_data = df[df['pipeline'] == 'P2d']
    
    if len(p2b_data) == 0 or len(p2d_data) == 0:
        print("⚠️  Missing data for P2b vs P2d comparison")
        return
    
    # Find best P2d configuration
    p2d_best = p2d_data.groupby(['noise_type', 'level'])['test_auc'].mean()
    best_config = p2d_best.idxmax()
    p2d_best_data = p2d_data[
        (p2d_data['noise_type'] == best_config[0]) & 
        (p2d_data['level'] == best_config[1])
    ]['test_auc']
    
    # Prepare data
    conditions = [
        'P2b\nComplex Amplitudes\n(No Noise)\n1,024 features',
        f'P2d\nDensity Matrix\n({best_config[0]}, p={best_config[1]})\n262,144 features'
    ]
    means = [p2b_data.mean(), p2d_best_data.mean()]
    stds = [p2b_data.std(), p2d_best_data.std()]
    colors = ['blue', 'red']
    
    # Create bar chart
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
               f'{mean:.3f}\n±{std:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Highlight better performer
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.set_title('Pure State (P2b) vs Mixed State (P2d) Feature Extraction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim([0.4, 0.95])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add comparison text
    diff = means[1] - means[0]
    if diff > 0:
        comparison = f"P2d outperforms P2b by {diff:.3f} AUC"
        color = 'green'
    else:
        comparison = f"P2b outperforms P2d by {abs(diff):.3f} AUC"
        color = 'blue'
    
    ax.text(0.5, 0.95, comparison, 
           transform=ax.transAxes, ha='center', va='top',
           fontsize=12, fontweight='bold', color=color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_p2b_vs_p2d_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 3_p2b_vs_p2d_comparison.png")


def plot_all_pipelines_comparison(df, output_dir):
    """
    Comprehensive comparison of all pipelines at their best.
    """
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    results = []
    
    # P1: Best dropout
    if 'P1' in df['pipeline'].values:
        p1_best_level = df[df['pipeline'] == 'P1'].groupby('level')['test_auc'].mean().idxmax()
        p1_data = df[(df['pipeline'] == 'P1') & (df['level'] == p1_best_level)]['test_auc']
        if len(p1_data) > 0:
            results.append(('P1 Classical\n(best dropout)', p1_data.mean(), p1_data.std(), 'lightblue'))
    
    # P2a: No noise
    if 'P2a' in df['pipeline'].values:
        p2a_data = df[(df['pipeline'] == 'P2a') & (df['noise_type'] == 'No Noise')]['test_auc']
        if len(p2a_data) > 0:
            results.append(('P2a Probabilities\n(no noise)', p2a_data.mean(), p2a_data.std(), 'gray'))
    
    # P2b: No noise
    if 'P2b' in df['pipeline'].values:
        p2b_data = df[(df['pipeline'] == 'P2b') & (df['noise_type'] == 'No Noise')]['test_auc']
        if len(p2b_data) > 0:
            results.append(('P2b Complex Amps\n(no noise)', p2b_data.mean(), p2b_data.std(), 'blue'))
    
    # P2c: No noise
    if 'P2c' in df['pipeline'].values:
        p2c_data = df[(df['pipeline'] == 'P2c') & (df['noise_type'] == 'No Noise')]['test_auc']
        if len(p2c_data) > 0:
            results.append(('P2c Pauli\n(no noise)', p2c_data.mean(), p2c_data.std(), 'orange'))
    
    # P2d: Best noise config
    if 'P2d' in df['pipeline'].values:
        p2d_data = df[df['pipeline'] == 'P2d']
        if len(p2d_data) > 0:
            p2d_best = p2d_data.groupby(['noise_type', 'level'])['test_auc'].mean()
            best_config = p2d_best.idxmax()
            p2d_best_data = p2d_data[
                (p2d_data['noise_type'] == best_config[0]) & 
                (p2d_data['level'] == best_config[1])
            ]['test_auc']
            if len(p2d_best_data) > 0:
                results.append((f'P2d Density Matrix\n({best_config[0][:3]})', 
                              p2d_best_data.mean(), p2d_best_data.std(), 'red'))
    
    # P3: Best noise config
    if 'P3' in df['pipeline'].values:
        p3_data = df[df['pipeline'] == 'P3']
        if len(p3_data) > 0:
            p3_best = p3_data.groupby(['noise_type', 'level'])['test_auc'].mean()
            best_config = p3_best.idxmax()
            p3_best_data = p3_data[
                (p3_data['noise_type'] == best_config[0]) & 
                (p3_data['level'] == best_config[1])
            ]['test_auc']
            if len(p3_best_data) > 0:
                results.append((f'P3 Classifier\n({best_config[0][:3]})', 
                              p3_best_data.mean(), p3_best_data.std(), 'green'))
    
    if len(results) == 0:
        print("⚠️  No data for all pipelines comparison")
        return
    
    # Unpack results
    labels, means, stds, colors = zip(*results)
    
    # Create bar chart
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
               f'{mean:.3f}\n±{std:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    ax.set_ylabel('Test AUC', fontsize=13, fontweight='bold')
    ax.set_title('All Pipelines: Best Configuration Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim([0.4, 0.95])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_all_pipelines_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved: 4_all_pipelines_comparison.png")


def generate_summary_table(df, output_dir):
    """Generate comprehensive summary table."""
    
    # Calculate aggregates
    summary = df.groupby(['pipeline', 'method', 'transform', 'noise_type', 'level']).agg({
        'test_auc': ['mean', 'std', 'count'],
        'test_accuracy': 'mean',
        'epochs_trained': 'mean'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={
        'test_auc_mean': 'mean_auc',
        'test_auc_std': 'std_auc',
        'test_auc_count': 'n_runs',
        'test_accuracy_mean': 'mean_accuracy',
        'epochs_trained_mean': 'mean_epochs'
    })
    
    # Sort
    summary = summary.sort_values(['pipeline', 'noise_type', 'level'])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    summary.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✅ Saved: results_summary.csv")
    
    # Create formatted text table
    txt_path = os.path.join(output_dir, 'results_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("╔" + "="*120 + "╗\n")
        f.write("║" + " "*40 + "COMPREHENSIVE RESULTS - ALL PIPELINES" + " "*43 + "║\n")
        f.write("╠" + "="*120 + "╣\n")
        f.write("║ Pipeline | Transform           | Noise Type     | Level | Mean AUC | Std AUC | Mean Acc | Epochs ║\n")
        f.write("╠" + "="*120 + "╣\n")
        
        for _, row in summary.iterrows():
            trans = str(row['transform']) if pd.notna(row['transform']) else "N/A"
            f.write(f"║ {row['pipeline']:<8} | {trans:<19} | {row['noise_type']:<14} | "
                   f"{row['level']:>5.2f} | {row['mean_auc']:>8.4f} | {row['std_auc']:>7.4f} | "
                   f"{row['mean_accuracy']:>8.2f} | {row['mean_epochs']:>6.1f} ║\n")
        
        f.write("╚" + "="*120 + "╝\n")
        
        # Add best configs
        f.write("\n\n" + "="*120 + "\n")
        f.write("BEST CONFIGURATIONS BY PIPELINE\n")
        f.write("="*120 + "\n\n")
        
        for pipeline in sorted(df['pipeline'].unique()):
            pipe_data = summary[summary['pipeline'] == pipeline]
            if len(pipe_data) == 0:
                continue
            
            best_row = pipe_data.loc[pipe_data['mean_auc'].idxmax()]
            f.write(f"\n{pipeline}:\n")
            f.write(f"  Transform: {best_row['transform']}\n")
            f.write(f"  Noise: {best_row['noise_type']}, Level: {best_row['level']:.3f}\n")
            f.write(f"  AUC: {best_row['mean_auc']:.4f} ± {best_row['std_auc']:.4f}\n")
    
    print(f"✅ Saved: results_summary.txt")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize quantum-classical hybrid experiments")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("QUANTUM-CLASSICAL HYBRID ANALYSIS - VISUALIZATION")
    print("="*70 + "\n")
    
    # Load all results
    df = load_all_results(args.results_dir)
    
    if len(df) == 0:
        print("❌ No results found!")
        return
    
    print(f"\n📊 Generating visualizations...\n")
    
    # Generate all visualizations
    plot_p2_no_noise_comparison(df, args.output_dir)
    plot_p2d_vs_noise(df, args.output_dir)
    plot_p2b_vs_p2d_comparison(df, args.output_dir)
    plot_all_pipelines_comparison(df, args.output_dir)
    summary = generate_summary_table(df, args.output_dir)
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  1. 1_p2_no_noise_comparison.png     (P2a vs P2b vs P2c)")
    print("  2. 2_p2d_vs_noise.png               (P2d noise analysis)")
    print("  3. 3_p2b_vs_p2d_comparison.png      (Pure vs Mixed states)")
    print("  4. 4_all_pipelines_comparison.png   (Complete overview)")
    print("  5. results_summary.csv               (Numerical results)")
    print("  6. results_summary.txt               (Formatted table)")
    print("\n")


if __name__ == "__main__":
    main()


# python src/visualize_analysis.py \
#     --results_dir /scratch90/chris/qc2025_final_proj/results_pneumoniamnist_n10 \
#     --output_dir /scratch90/chris/qc2025_final_proj/analysis_pneumoniamnist_n10