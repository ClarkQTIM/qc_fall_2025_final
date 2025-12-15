"""
analyze_quantum_features.py

Diagnose whether VQC produces distinguishable quantum states for different classes.

This script:
1. Loads a trained P2a/P2b model from checkpoint
2. Extracts raw quantum features (amplitudes, phases, probabilities)
3. Saves features to CSV for each class
4. Generates visualizations and statistical analysis
5. Determines if the problem is VQC (identical states) or MLP (can't learn)

Usage:
    python analyze_quantum_features.py \
        --model_path /path/to/best_model.pth \
        --dataset retinamnist \
        --save_dir /path/to/analysis \
        --n_samples 100
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Add parent directory to path to import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model import HybridModel
from src.loader import get_medmnist_loaders


def load_model_from_checkpoint(model_dir, device):
    """
    Load trained model from checkpoint directory.
    
    Args:
        model_dir: Directory containing best_model.pth and config.json
        device: torch device
    
    Returns:
        model: Loaded HybridModel
        config: Configuration dictionary
    """
    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"📋 Loaded configuration:")
    print(f"   Dataset: {config.get('dataset', 'retinamnist')}")
    print(f"   Pipeline: {'P2a' if config['feature_transform'] == 'probabilities' else 'P2b' if config['feature_transform'] == 'complex_amplitudes' else config['feature_transform']}")
    print(f"   VQC Mode: {config['vqc_mode']}")
    print(f"   N Qubits: {config['n_qubits']}")
    print(f"   Layers: {config['num_layers']}")
    
    # Recreate model with exact same architecture
    model = HybridModel(
        num_classes=2,  # Binary classification
        use_quantum=config['use_quantum'],
        vqc_mode=config['vqc_mode'],
        n_qubits=config['n_qubits'],
        num_layers=config['num_layers'],
        vqc_type=config['vqc_type'],
        shots=config.get('shots', None),
        bit_flip_prob=config.get('bit_flip_prob', 0.0),
        amp_damp_prob=config.get('amp_damp_prob', 0.0),
        depol_prob=config.get('depol_prob', 0.0),
        dropout=config['dropout'],
        feature_transform=config['feature_transform']
    ).to(device)
    
    # Load weights
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth not found in {model_dir}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Loaded model weights from {model_path}")
    
    return model, config


def extract_quantum_features(model, dataloader, device, max_samples_per_class=None):
    """
    Extract raw quantum features from VQC for each image.
    
    Args:
        model: Trained HybridModel
        dataloader: Test data loader
        device: torch device
        max_samples_per_class: Maximum samples per class (None = all)
    
    Returns:
        features_dict: {
            'class_0': {
                'amplitudes': np.array,  # [N, 512]
                'phases': np.array,      # [N, 512]
                'probabilities': np.array, # [N, 512]
                'image_indices': list
            },
            'class_1': {...}
        }
    """
    feature_transform = model.feature_transform
    print(f"\n🔬 Extracting quantum features ({feature_transform})...")
    
    # Storage for each class
    features_dict = {
        'class_0': {
            'amplitudes': [],
            'phases': [],
            'probabilities': [],
            'image_indices': []
        },
        'class_1': {
            'amplitudes': [],
            'phases': [],
            'probabilities': [],
            'image_indices': []
        }
    }
    
    class_counts = {'class_0': 0, 'class_1': 0}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
            images = images.to(device)
            labels = labels.squeeze().cpu().numpy()
            
            # Get CNN features
            cnn_features = model.feature_extractor(images)
            batch_size = cnn_features.shape[0]
            
            # Extract quantum state for each image
            for i in range(batch_size):
                label = int(labels[i])
                class_key = f'class_{label}'
                
                # Check if we've reached max samples for this class
                if max_samples_per_class and class_counts[class_key] >= max_samples_per_class:
                    continue
                
                # Get quantum state for this image
                x = cnn_features[i].double()
                
                if model.vqc_type == "basic":
                    quantum_output = model.qnode(x, model.vqc_weights)
                elif model.vqc_type == "expressive":
                    quantum_output = model.qnode(x, model.single_qubit_weights, 
                                                model.entangling_weights)
                
                # Extract amplitudes and phases from statevector
                if feature_transform in ['probabilities', 'complex_amplitudes']:
                    # quantum_output is the statevector [512] complex
                    statevector = quantum_output.cpu().numpy()
                    
                    # Amplitudes (|α|)
                    amplitudes = np.abs(statevector)
                    
                    # Phases (arg(α))
                    phases = np.angle(statevector)
                    
                    # Probabilities (|α|²)
                    probabilities = amplitudes ** 2
                    
                    features_dict[class_key]['amplitudes'].append(amplitudes)
                    features_dict[class_key]['phases'].append(phases)
                    features_dict[class_key]['probabilities'].append(probabilities)
                    features_dict[class_key]['image_indices'].append(batch_idx * dataloader.batch_size + i)
                    
                    class_counts[class_key] += 1
                
                else:
                    print(f"⚠️  Feature transform '{feature_transform}' not supported for this analysis")
                    return None
                
                # Stop if both classes have enough samples
                if max_samples_per_class and all(count >= max_samples_per_class for count in class_counts.values()):
                    break
            
            if max_samples_per_class and all(count >= max_samples_per_class for count in class_counts.values()):
                break
    
    # Convert lists to arrays
    for class_key in features_dict.keys():
        features_dict[class_key]['amplitudes'] = np.array(features_dict[class_key]['amplitudes'])
        features_dict[class_key]['phases'] = np.array(features_dict[class_key]['phases'])
        features_dict[class_key]['probabilities'] = np.array(features_dict[class_key]['probabilities'])
    
    print(f"\n📊 Extracted features:")
    print(f"   Class 0: {len(features_dict['class_0']['amplitudes'])} samples")
    print(f"   Class 1: {len(features_dict['class_1']['amplitudes'])} samples")
    
    return features_dict


def save_features_to_csv(features_dict, save_dir):
    """
    Save quantum features to CSV files.
    
    Args:
        features_dict: Dictionary of features per class
        save_dir: Directory to save CSVs
    """
    print(f"\n💾 Saving features to CSV...")
    
    for class_key, features in features_dict.items():
        n_samples = len(features['amplitudes'])
        n_features = features['amplitudes'].shape[1]  # 512 features (basis states)
        
        # Create DataFrame with all features
        data = {
            'image_idx': features['image_indices'],
            'class': [int(class_key.split('_')[1])] * n_samples
        }
        
        # Add amplitude columns
        for i in range(n_features):
            data[f'amplitude_{i}'] = features['amplitudes'][:, i]
        
        # Add phase columns
        for i in range(n_features):
            data[f'phase_{i}'] = features['phases'][:, i]
        
        # Add probability columns
        for i in range(n_features):
            data[f'probability_{i}'] = features['probabilities'][:, i]
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, f'quantum_features_{class_key}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved {csv_path}")
    
    # Save combined CSV
    all_data = []
    for class_key, features in features_dict.items():
        n_samples = len(features['amplitudes'])
        n_features = features['amplitudes'].shape[1]  # 512 features
        class_label = int(class_key.split('_')[1])
        
        for sample_idx in range(n_samples):
            row = {
                'image_idx': features['image_indices'][sample_idx],
                'class': class_label
            }
            
            for i in range(n_features):
                row[f'amplitude_{i}'] = features['amplitudes'][sample_idx, i]
                row[f'phase_{i}'] = features['phases'][sample_idx, i]
                row[f'probability_{i}'] = features['probabilities'][sample_idx, i]
            
            all_data.append(row)
    
    df_combined = pd.DataFrame(all_data)
    csv_combined_path = os.path.join(save_dir, 'quantum_features_combined.csv')
    df_combined.to_csv(csv_combined_path, index=False)
    print(f"   ✅ Saved {csv_combined_path}")


def compute_statistics(features_dict, save_dir):
    """
    Compute statistical separability metrics.
    
    Args:
        features_dict: Dictionary of features per class
        save_dir: Directory to save results
    """
    print(f"\n📈 Computing statistical separability metrics...")
    
    class0_probs = features_dict['class_0']['probabilities']
    class1_probs = features_dict['class_1']['probabilities']
    
    class0_amps = features_dict['class_0']['amplitudes']
    class1_amps = features_dict['class_1']['amplitudes']
    
    class0_phases = features_dict['class_0']['phases']
    class1_phases = features_dict['class_1']['phases']
    
    results = []
    
    # === PROBABILITY ANALYSIS ===
    results.append("="*80)
    results.append("PROBABILITY ANALYSIS (|α|²)")
    results.append("="*80)
    
    # Mean probabilities per class
    class0_prob_mean = class0_probs.mean(axis=0)
    class1_prob_mean = class1_probs.mean(axis=0)
    
    results.append(f"\nClass 0 mean probability: {class0_prob_mean.mean():.6f} ± {class0_prob_mean.std():.6f}")
    results.append(f"Class 1 mean probability: {class1_prob_mean.mean():.6f} ± {class1_prob_mean.std():.6f}")
    
    # Per-feature difference
    prob_diff = np.abs(class0_prob_mean - class1_prob_mean)
    results.append(f"\nMean per-feature |difference|: {prob_diff.mean():.6f}")
    results.append(f"Max per-feature |difference|:  {prob_diff.max():.6f}")
    results.append(f"Features with >0.001 diff:     {(prob_diff > 0.001).sum()} / {len(prob_diff)}")
    
    # Statistical test (t-test on flattened probabilities)
    class0_flat = class0_probs.flatten()
    class1_flat = class1_probs.flatten()
    t_stat, p_value = stats.ttest_ind(class0_flat, class1_flat)
    results.append(f"\nT-test (all probabilities):")
    results.append(f"   t-statistic: {t_stat:.4f}")
    results.append(f"   p-value:     {p_value:.4e}")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(class0_flat, class1_flat)
    results.append(f"\nKolmogorov-Smirnov test:")
    results.append(f"   KS statistic: {ks_stat:.4f}")
    results.append(f"   p-value:      {ks_pvalue:.4e}")
    
    # === AMPLITUDE ANALYSIS ===
    results.append("\n" + "="*80)
    results.append("AMPLITUDE ANALYSIS (|α|)")
    results.append("="*80)
    
    class0_amp_mean = class0_amps.mean(axis=0)
    class1_amp_mean = class1_amps.mean(axis=0)
    
    results.append(f"\nClass 0 mean amplitude: {class0_amp_mean.mean():.6f} ± {class0_amp_mean.std():.6f}")
    results.append(f"Class 1 mean amplitude: {class1_amp_mean.mean():.6f} ± {class1_amp_mean.std():.6f}")
    
    amp_diff = np.abs(class0_amp_mean - class1_amp_mean)
    results.append(f"\nMean per-feature |difference|: {amp_diff.mean():.6f}")
    results.append(f"Max per-feature |difference|:  {amp_diff.max():.6f}")
    
    # === PHASE ANALYSIS ===
    results.append("\n" + "="*80)
    results.append("PHASE ANALYSIS (arg(α))")
    results.append("="*80)
    
    # Circular statistics for phases
    class0_phase_mean = stats.circmean(class0_phases, axis=0, low=-np.pi, high=np.pi)
    class1_phase_mean = stats.circmean(class1_phases, axis=0, low=-np.pi, high=np.pi)
    
    results.append(f"\nClass 0 mean phase (circular): {class0_phase_mean.mean():.4f} rad")
    results.append(f"Class 1 mean phase (circular): {class1_phase_mean.mean():.4f} rad")
    
    # Angular difference
    phase_diff = np.abs(np.angle(np.exp(1j * class0_phase_mean) / np.exp(1j * class1_phase_mean)))
    results.append(f"\nMean angular difference: {phase_diff.mean():.4f} rad")
    results.append(f"Max angular difference:  {phase_diff.max():.4f} rad")
    
    # === DISTANCE METRICS ===
    results.append("\n" + "="*80)
    results.append("SEPARABILITY METRICS")
    results.append("="*80)
    
    # Intra-class distances (using probabilities)
    def compute_mean_pairwise_distance(data):
        n_samples = len(data)
        if n_samples < 2:
            return 0.0
        distances = []
        for i in range(min(n_samples, 100)):  # Sample 100 pairs to avoid O(n²)
            for j in range(i+1, min(n_samples, 100)):
                dist = np.linalg.norm(data[i] - data[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0.0
    
    intra_class0_dist = compute_mean_pairwise_distance(class0_probs)
    intra_class1_dist = compute_mean_pairwise_distance(class1_probs)
    
    # Inter-class distance
    inter_class_dists = []
    for i in range(min(len(class0_probs), 100)):
        for j in range(min(len(class1_probs), 100)):
            dist = np.linalg.norm(class0_probs[i] - class1_probs[j])
            inter_class_dists.append(dist)
    inter_class_dist = np.mean(inter_class_dists)
    
    results.append(f"\nIntra-class distance (Class 0): {intra_class0_dist:.6f}")
    results.append(f"Intra-class distance (Class 1): {intra_class1_dist:.6f}")
    results.append(f"Inter-class distance:            {inter_class_dist:.6f}")
    
    mean_intra = (intra_class0_dist + intra_class1_dist) / 2
    separability_ratio = inter_class_dist / mean_intra if mean_intra > 0 else 0
    results.append(f"\nSeparability ratio (inter/intra): {separability_ratio:.4f}")
    
    # === INTERPRETATION ===
    results.append("\n" + "="*80)
    results.append("INTERPRETATION")
    results.append("="*80)
    
    if prob_diff.max() < 0.001 and separability_ratio < 1.1:
        results.append("\n❌ PROBLEM: VQC produces nearly IDENTICAL states for both classes!")
        results.append("   The quantum features are not distinguishable.")
        results.append("   → VQC is not encoding class information properly")
        results.append("\n   RECOMMENDATION:")
        results.append("   • Increase VQC capacity (more layers, expressive circuit)")
        results.append("   • Try different encoding strategies")
        results.append("   • Check if CNN features themselves are discriminative")
    
    elif prob_diff.max() > 0.01 and separability_ratio > 1.5:
        results.append("\n✅ VQC produces DIFFERENT states for each class!")
        results.append("   The quantum features are statistically separable.")
        results.append("   → Problem is likely the MLP (cannot learn from quantum features)")
        results.append("\n   RECOMMENDATION:")
        results.append("   • Try simpler classifier (linear instead of 2-layer MLP)")
        results.append("   • Use feature selection (only discriminative qubits)")
        results.append("   • Reduce feature dimensionality before classification")
    
    else:
        results.append("\n⚠️  VQC produces SLIGHTLY DIFFERENT states")
        results.append("   Classes are weakly separable in quantum feature space.")
        results.append(f"   Max difference: {prob_diff.max():.6f} (threshold: 0.01 for 'strong')")
        results.append(f"   Separability: {separability_ratio:.2f} (threshold: 1.5 for 'strong')")
        results.append("   → VQC encodes SOME class info, but signal is WEAK")
        results.append("\n   DUAL PROBLEM:")
        results.append("   1. VQC: Weak encoding (only ~10% of features discriminative)")
        results.append("   2. MLP: Cannot find weak signal in high-dimensional space")
        results.append("\n   RECOMMENDATION:")
        results.append("   • Feature selection: Use only the most discriminative features")
        results.append(f"   • Found {(prob_diff > 0.001).sum()} features with difference >0.001")
        results.append("   • Try training MLP on ONLY these discriminative features")
        results.append("   • OR: Increase VQC capacity to create stronger separation")
    
    # Save to file
    results_text = "\n".join(results)
    print(results_text)
    
    stats_path = os.path.join(save_dir, 'statistical_analysis.txt')
    with open(stats_path, 'w') as f:
        f.write(results_text)
    
    print(f"\n💾 Saved statistical analysis to {stats_path}")


def plot_distributions(features_dict, save_dir):
    """
    Create visualization plots of feature distributions.
    
    Args:
        features_dict: Dictionary of features per class
        save_dir: Directory to save plots
    """
    print(f"\n🎨 Generating visualization plots...")
    
    class0_probs = features_dict['class_0']['probabilities']
    class1_probs = features_dict['class_1']['probabilities']
    
    class0_amps = features_dict['class_0']['amplitudes']
    class1_amps = features_dict['class_1']['amplitudes']
    
    class0_phases = features_dict['class_0']['phases']
    class1_phases = features_dict['class_1']['phases']
    
    # === PLOT 1: PROBABILITY DISTRIBUTIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum Feature Distributions', fontsize=16, fontweight='bold')
    
    # Histogram of all probabilities
    ax = axes[0, 0]
    ax.hist(class0_probs.flatten(), bins=50, alpha=0.6, label='Class 0', density=True)
    ax.hist(class1_probs.flatten(), bins=50, alpha=0.6, label='Class 1', density=True)
    ax.set_xlabel('Probability |α|²')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of All Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean probability per image
    ax = axes[0, 1]
    class0_mean_per_image = class0_probs.mean(axis=1)
    class1_mean_per_image = class1_probs.mean(axis=1)
    
    # Check if data has enough range for histogram
    all_data = np.concatenate([class0_mean_per_image, class1_mean_per_image])
    data_range = all_data.max() - all_data.min()
    
    if data_range < 1e-6:  # Essentially no variation
        # Just show box plot instead
        ax.boxplot([class0_mean_per_image, class1_mean_per_image], 
                   labels=['Class 0', 'Class 1'])
        ax.set_ylabel('Mean Probability')
        ax.set_title('Average Probability Per Image\n(Distributions nearly identical)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # Adaptive binning based on data range
        n_bins = max(5, min(30, int(np.sqrt(len(all_data)))))
        try:
            ax.hist(class0_mean_per_image, bins=n_bins, alpha=0.6, label='Class 0', density=True)
            ax.hist(class1_mean_per_image, bins=n_bins, alpha=0.6, label='Class 1', density=True)
            ax.set_xlabel('Mean Probability Per Image')
            ax.set_ylabel('Density')
            ax.set_title('Average Probability Per Image')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except ValueError:
            # Fallback to box plot if histogram still fails
            ax.boxplot([class0_mean_per_image, class1_mean_per_image], 
                       labels=['Class 0', 'Class 1'])
            ax.set_ylabel('Mean Probability')
            ax.set_title('Average Probability Per Image\n(Range too small for histogram)')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Amplitude distributions
    ax = axes[1, 0]
    ax.hist(class0_amps.flatten(), bins=50, alpha=0.6, label='Class 0', density=True)
    ax.hist(class1_amps.flatten(), bins=50, alpha=0.6, label='Class 1', density=True)
    ax.set_xlabel('Amplitude |α|')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Amplitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase distributions (circular histogram)
    ax = axes[1, 1]
    ax.hist(class0_phases.flatten(), bins=50, alpha=0.6, label='Class 0', 
            density=True, range=(-np.pi, np.pi))
    ax.hist(class1_phases.flatten(), bins=50, alpha=0.6, label='Class 1', 
            density=True, range=(-np.pi, np.pi))
    ax.set_xlabel('Phase arg(α) [radians]')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-np.pi, np.pi)
    
    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'quantum_feature_distributions.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved {dist_path}")
    
    # === PLOT 2: PER-QUBIT ANALYSIS ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Qubit Analysis', fontsize=16, fontweight='bold')
    
    # Mean probability per qubit
    ax = axes[0, 0]
    class0_prob_mean = class0_probs.mean(axis=0)
    class1_prob_mean = class1_probs.mean(axis=0)
    x = np.arange(len(class0_prob_mean))
    ax.plot(x, class0_prob_mean, 'o-', alpha=0.7, label='Class 0', markersize=3)
    ax.plot(x, class1_prob_mean, 'o-', alpha=0.7, label='Class 1', markersize=3)
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('Mean Probability')
    ax.set_title('Mean Probability Per Qubit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Probability difference per qubit
    ax = axes[0, 1]
    prob_diff = np.abs(class0_prob_mean - class1_prob_mean)
    ax.bar(x, prob_diff, alpha=0.7, color='purple')
    ax.axhline(y=0.001, color='r', linestyle='--', label='0.001 threshold')
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('|Difference|')
    ax.set_title('Per-Qubit Probability Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std deviation per qubit
    ax = axes[1, 0]
    class0_prob_std = class0_probs.std(axis=0)
    class1_prob_std = class1_probs.std(axis=0)
    ax.plot(x, class0_prob_std, 'o-', alpha=0.7, label='Class 0', markersize=3)
    ax.plot(x, class1_prob_std, 'o-', alpha=0.7, label='Class 1', markersize=3)
    ax.set_xlabel('Qubit Index')
    ax.set_ylabel('Std Deviation')
    ax.set_title('Probability Std Dev Per Qubit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Most discriminative qubits
    ax = axes[1, 1]
    top_k = 20
    n_available = len(prob_diff)
    top_k_actual = min(top_k, n_available)  # Don't request more than we have
    
    if top_k_actual > 0:
        top_indices = np.argsort(prob_diff)[-top_k_actual:][::-1]
        ax.barh(np.arange(top_k_actual), prob_diff[top_indices], alpha=0.7, color='orange')
        ax.set_yticks(np.arange(top_k_actual))
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.set_xlabel('|Probability Difference|')
        ax.set_title(f'Top {top_k_actual} Most Discriminative Features')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No discriminative features found', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Most Discriminative Features')
    
    plt.tight_layout()
    perfeature_path = os.path.join(save_dir, 'per_feature_analysis.png')
    plt.savefig(perfeature_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved {perfeature_path}")
    
    # === PLOT 3: PCA PROJECTION ===
    print(f"   Computing PCA projection...")
    all_probs = np.vstack([class0_probs, class1_probs])
    all_labels = np.array([0]*len(class0_probs) + [1]*len(class1_probs))
    
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(all_probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold')
    
    # PCA
    ax = axes[0]
    scatter0 = ax.scatter(pca_features[all_labels==0, 0], 
                         pca_features[all_labels==0, 1],
                         alpha=0.6, label='Class 0', s=20)
    scatter1 = ax.scatter(pca_features[all_labels==1, 0], 
                         pca_features[all_labels==1, 1],
                         alpha=0.6, label='Class 1', s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_title('PCA Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # t-SNE (only if we have enough samples)
    ax = axes[1]
    if len(all_probs) > 50:
        print(f"   Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_probs)//4))
        tsne_features = tsne.fit_transform(all_probs)
        
        ax.scatter(tsne_features[all_labels==0, 0], 
                  tsne_features[all_labels==0, 1],
                  alpha=0.6, label='Class 0', s=20)
        ax.scatter(tsne_features[all_labels==1, 0], 
                  tsne_features[all_labels==1, 1],
                  alpha=0.6, label='Class 1', s=20)
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_title('t-SNE Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('t-SNE Projection')
    
    plt.tight_layout()
    pca_path = os.path.join(save_dir, 'dimensionality_reduction.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved {pca_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze quantum features from trained models")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint directory (containing best_model.pth and config.json)')
    parser.add_argument('--dataset', type=str, default='retinamnist',
                       choices=['retinamnist', 'breastmnist', 'pneumoniamnist'],
                       help='Dataset to use (default: retinamnist)')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples per class to analyze (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for data loading (default: 4)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " QUANTUM FEATURE ANALYSIS".center(78) + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Load model
    print(f"📥 Loading model from {args.model_path}...")
    model, config = load_model_from_checkpoint(args.model_path, device)
    
    # Determine dataset from config or argument
    dataset_name = config.get('dataset', args.dataset)
    print(f"\n📊 Loading {dataset_name.upper()} test dataset...")
    
    # Load test data
    _, _, test_loader, _ = get_medmnist_loaders(
        dataset_name=dataset_name,
        img_size=(256, 256),
        batch_size=args.batch_size,
        binary=True,
        balance_classes=True,
        n_train=None,
        n_val=None,
        n_test=None,
        num_workers=0
    )
    
    # Extract quantum features
    features_dict = extract_quantum_features(
        model, 
        test_loader, 
        device,
        max_samples_per_class=args.n_samples
    )
    
    if features_dict is None:
        print("❌ Failed to extract features!")
        return
    
    # Save features to CSV
    save_features_to_csv(features_dict, args.save_dir)
    
    # Compute statistics
    compute_statistics(features_dict, args.save_dir)
    
    # Generate plots
    plot_distributions(features_dict, args.save_dir)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " ANALYSIS COMPLETE!".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print(f"\n📁 All results saved to: {args.save_dir}\n")


if __name__ == "__main__":
    main()

# python /scratch90/chris/qc2025_final_proj/src/troubleshooting_scripts/amp_phases_analysis.py \
#     --model_path /scratch90/chris/qc2025_final_proj/results/results_retinamnist_n10_dual_lr/p2a_probabilities/run_1 \
#     --dataset retinamnist \
#     --save_dir /scratch90/chris/qc2025_final_proj/graphics_misc/amp_phases_analysis \
#     --n_samples 100