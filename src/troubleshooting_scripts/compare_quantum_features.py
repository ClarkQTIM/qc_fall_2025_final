"""
compare_quantum_features.py

Compare quantum feature separability between two models:
- Model 1 (P2): VQC → Features → MLP (FAILED to learn)
- Model 2 (P3): VQC → Direct Measurement (WORKED)

This answers: "Does the working model produce more separable quantum states?"

Usage:
    python compare_quantum_features.py \
        --model1_path /path/to/p2b/run_1 \
        --model2_path /path/to/p3/run_1 \
        --dataset retinamnist \
        --save_dir /path/to/comparison \
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

# Add parent directory to path to import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model import HybridModel
from src.loader import get_medmnist_loaders


def load_model_from_checkpoint(model_dir, device, model_name):
    """Load trained model from checkpoint directory."""
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n📋 {model_name} Configuration:")
    print(f"   Dataset: {config.get('dataset', 'retinamnist')}")
    print(f"   VQC Mode: {config['vqc_mode']}")
    print(f"   Feature Transform: {config.get('feature_transform', 'N/A')}")
    print(f"   N Qubits: {config['n_qubits']}")
    print(f"   Layers: {config['num_layers']}")
    
    # Recreate model
    model = HybridModel(
        num_classes=2,
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
        feature_transform=config.get('feature_transform', 'probabilities')
    ).to(device)
    
    # Load weights
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"best_model.pth not found in {model_dir}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"   ✅ Loaded weights")
    
    return model, config


def extract_quantum_features(model, dataloader, device, max_samples_per_class, model_name):
    """Extract raw quantum features from VQC."""
    print(f"\n🔬 Extracting quantum features from {model_name}...")
    
    features_dict = {
        'class_0': {'amplitudes': [], 'phases': [], 'probabilities': [], 'image_indices': []},
        'class_1': {'amplitudes': [], 'phases': [], 'probabilities': [], 'image_indices': []}
    }
    
    class_counts = {'class_0': 0, 'class_1': 0}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"{model_name}")):
            images = images.to(device)
            labels = labels.squeeze().cpu().numpy()
            
            cnn_features = model.feature_extractor(images)
            batch_size = cnn_features.shape[0]
            
            for i in range(batch_size):
                label = int(labels[i])
                class_key = f'class_{label}'
                
                if max_samples_per_class and class_counts[class_key] >= max_samples_per_class:
                    continue
                
                x = cnn_features[i].double()
                
                # For P3 (classifier mode), we need to get statevector BEFORE measurement
                # Create a temporary qnode that returns statevector instead of probabilities
                if model.vqc_mode == "classifier":
                    # We need to recreate the circuit to get statevector
                    import pennylane as qml
                    
                    # Create temporary device and qnode for statevector extraction
                    temp_dev = qml.device("default.qubit", wires=model.n_qubits)
                    
                    if model.vqc_type == "basic":
                        @qml.qnode(temp_dev, interface="torch")
                        def get_statevector(x_input, weights):
                            qml.AmplitudeEmbedding(x_input, wires=range(model.n_qubits), normalize=True)
                            for layer in range(model.num_layers):
                                for q in range(model.n_qubits):
                                    qml.RY(weights[layer, q], wires=q)
                                for q in range(model.n_qubits):
                                    qml.CNOT(wires=[q, (q + 1) % model.n_qubits])
                            return qml.state()
                        
                        quantum_output = get_statevector(x, model.vqc_weights)
                    
                    elif model.vqc_type == "expressive":
                        def all_to_all_isingzz(weights, n_qubits):
                            for i in range(n_qubits):
                                for j in range(i + 1, n_qubits):
                                    qml.IsingZZ(weights[i, j], wires=[i, j])
                        
                        @qml.qnode(temp_dev, interface="torch")
                        def get_statevector(x_input, sq_weights, ent_weights):
                            qml.AmplitudeEmbedding(x_input, wires=range(model.n_qubits), normalize=True)
                            for layer in range(model.num_layers):
                                for q in range(model.n_qubits):
                                    qml.U3(*sq_weights[layer, q], wires=q)
                                all_to_all_isingzz(ent_weights[layer], model.n_qubits)
                            return qml.state()
                        
                        quantum_output = get_statevector(x, model.single_qubit_weights, 
                                                        model.entangling_weights)
                    
                    statevector = quantum_output.cpu().numpy()
                
                else:
                    # P2: Already returns statevector
                    if model.vqc_type == "basic":
                        quantum_output = model.qnode(x, model.vqc_weights)
                    elif model.vqc_type == "expressive":
                        quantum_output = model.qnode(x, model.single_qubit_weights, 
                                                    model.entangling_weights)
                    
                    statevector = quantum_output.cpu().numpy()
                
                # Extract amplitudes and phases
                if hasattr(statevector, '__len__') and len(statevector) >= 512:
                    # Full statevector
                    amplitudes = np.abs(statevector[:512])
                    phases = np.angle(statevector[:512])
                    probabilities = amplitudes ** 2
                else:
                    print(f"   ⚠️  Unexpected statevector shape: {statevector.shape}")
                    continue
                
                features_dict[class_key]['amplitudes'].append(amplitudes)
                features_dict[class_key]['phases'].append(phases)
                features_dict[class_key]['probabilities'].append(probabilities)
                features_dict[class_key]['image_indices'].append(batch_idx * dataloader.batch_size + i)
                
                class_counts[class_key] += 1
                
                if max_samples_per_class and all(count >= max_samples_per_class for count in class_counts.values()):
                    break
            
            if max_samples_per_class and all(count >= max_samples_per_class for count in class_counts.values()):
                break
    
    # Convert to arrays
    for class_key in features_dict.keys():
        if len(features_dict[class_key]['amplitudes']) > 0:
            features_dict[class_key]['amplitudes'] = np.array(features_dict[class_key]['amplitudes'])
            features_dict[class_key]['phases'] = np.array(features_dict[class_key]['phases'])
            features_dict[class_key]['probabilities'] = np.array(features_dict[class_key]['probabilities'])
    
    print(f"   Class 0: {len(features_dict['class_0']['amplitudes'])} samples")
    print(f"   Class 1: {len(features_dict['class_1']['amplitudes'])} samples")
    
    return features_dict


def compute_separability_metrics(features_dict):
    """Compute key separability metrics."""
    class0_probs = features_dict['class_0']['probabilities']
    class1_probs = features_dict['class_1']['probabilities']
    
    if len(class0_probs) == 0 or len(class1_probs) == 0:
        return None
    
    # Mean difference per feature
    class0_prob_mean = class0_probs.mean(axis=0)
    class1_prob_mean = class1_probs.mean(axis=0)
    prob_diff = np.abs(class0_prob_mean - class1_prob_mean)
    
    # Distance metrics
    def compute_mean_pairwise_distance(data):
        n_samples = len(data)
        if n_samples < 2:
            return 0.0
        distances = []
        for i in range(min(n_samples, 100)):
            for j in range(i+1, min(n_samples, 100)):
                dist = np.linalg.norm(data[i] - data[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0.0
    
    intra_class0_dist = compute_mean_pairwise_distance(class0_probs)
    intra_class1_dist = compute_mean_pairwise_distance(class1_probs)
    
    inter_class_dists = []
    for i in range(min(len(class0_probs), 100)):
        for j in range(min(len(class1_probs), 100)):
            dist = np.linalg.norm(class0_probs[i] - class1_probs[j])
            inter_class_dists.append(dist)
    inter_class_dist = np.mean(inter_class_dists)
    
    mean_intra = (intra_class0_dist + intra_class1_dist) / 2
    separability_ratio = inter_class_dist / mean_intra if mean_intra > 0 else 0
    
    # Statistical tests
    class0_flat = class0_probs.flatten()
    class1_flat = class1_probs.flatten()
    t_stat, p_value = stats.ttest_ind(class0_flat, class1_flat)
    ks_stat, ks_pvalue = stats.ks_2samp(class0_flat, class1_flat)
    
    return {
        'mean_diff': prob_diff.mean(),
        'max_diff': prob_diff.max(),
        'n_discriminative': (prob_diff > 0.001).sum(),
        'total_features': len(prob_diff),
        'separability_ratio': separability_ratio,
        'intra_dist': mean_intra,
        'inter_dist': inter_class_dist,
        't_statistic': t_stat,
        'p_value': p_value,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'prob_diff': prob_diff  # For plotting
    }


def create_comparison_plots(features1, features2, metrics1, metrics2, model1_name, model2_name, save_dir):
    """Create side-by-side comparison plots."""
    print(f"\n🎨 Generating comparison plots...")
    
    if metrics1 is None or metrics2 is None:
        print("   ⚠️  Cannot create comparison plots (missing data)")
        return
    
    # === PLOT 1: SEPARABILITY METRICS COMPARISON ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Quantum Feature Separability: {model1_name} vs {model2_name}', 
                fontsize=16, fontweight='bold')
    
    # Bar chart: Key metrics
    ax = axes[0, 0]
    metrics_names = ['Max\nDifference', 'Mean\nDifference', 'Separability\nRatio']
    model1_vals = [metrics1['max_diff'], metrics1['mean_diff'], metrics1['separability_ratio']]
    model2_vals = [metrics2['max_diff'], metrics2['mean_diff'], metrics2['separability_ratio']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, model1_vals, width, label=model1_name, alpha=0.8)
    ax.bar(x + width/2, model2_vals, width, label=model2_name, alpha=0.8)
    ax.set_ylabel('Value')
    ax.set_title('Key Separability Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Discriminative features
    ax = axes[0, 1]
    categories = [model1_name, model2_name]
    discriminative = [metrics1['n_discriminative'], metrics2['n_discriminative']]
    non_discriminative = [
        metrics1['total_features'] - metrics1['n_discriminative'],
        metrics2['total_features'] - metrics2['n_discriminative']
    ]
    ax.bar(categories, discriminative, label='Discriminative (>0.001)', alpha=0.8, color='green')
    ax.bar(categories, non_discriminative, bottom=discriminative, 
           label='Non-discriminative', alpha=0.8, color='gray')
    ax.set_ylabel('Number of Features')
    ax.set_title('Discriminative vs Non-discriminative Features')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Distance comparison
    ax = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [metrics1['intra_dist'], metrics1['inter_dist']], 
           width, label=model1_name, alpha=0.8)
    ax.bar(x + width/2, [metrics2['intra_dist'], metrics2['inter_dist']], 
           width, label=model2_name, alpha=0.8)
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Intra-class vs Inter-class Distance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Intra-class', 'Inter-class'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Per-feature difference comparison (first 100 features)
    ax = axes[1, 1]
    n_show = min(100, len(metrics1['prob_diff']))
    x = np.arange(n_show)
    ax.plot(x, metrics1['prob_diff'][:n_show], alpha=0.7, label=model1_name, linewidth=1.5)
    ax.plot(x, metrics2['prob_diff'][:n_show], alpha=0.7, label=model2_name, linewidth=1.5)
    ax.axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='0.001 threshold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('|Probability Difference|')
    ax.set_title('Per-Feature Difference (first 100 features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved {comparison_path}")


def create_comparison_report(metrics1, metrics2, model1_name, model2_name, save_dir):
    """Create text report comparing the two models."""
    print(f"\n📊 Generating comparison report...")
    
    if metrics1 is None or metrics2 is None:
        print("   ⚠️  Cannot create report (missing data)")
        return
    
    report = []
    report.append("="*80)
    report.append("QUANTUM FEATURE SEPARABILITY COMPARISON")
    report.append("="*80)
    
    report.append(f"\n{model1_name}:")
    report.append(f"   Max difference:       {metrics1['max_diff']:.6f}")
    report.append(f"   Mean difference:      {metrics1['mean_diff']:.6f}")
    report.append(f"   Discriminative:       {metrics1['n_discriminative']}/{metrics1['total_features']} features ({100*metrics1['n_discriminative']/metrics1['total_features']:.1f}%)")
    report.append(f"   Separability ratio:   {metrics1['separability_ratio']:.4f}")
    report.append(f"   T-test p-value:       {metrics1['p_value']:.4e}")
    
    report.append(f"\n{model2_name}:")
    report.append(f"   Max difference:       {metrics2['max_diff']:.6f}")
    report.append(f"   Mean difference:      {metrics2['mean_diff']:.6f}")
    report.append(f"   Discriminative:       {metrics2['n_discriminative']}/{metrics2['total_features']} features ({100*metrics2['n_discriminative']/metrics2['total_features']:.1f}%)")
    report.append(f"   Separability ratio:   {metrics2['separability_ratio']:.4f}")
    report.append(f"   T-test p-value:       {metrics2['p_value']:.4e}")
    
    report.append("\n" + "="*80)
    report.append("COMPARISON")
    report.append("="*80)
    
    # Compare metrics
    diff_max = metrics2['max_diff'] - metrics1['max_diff']
    diff_sep = metrics2['separability_ratio'] - metrics1['separability_ratio']
    diff_disc = metrics2['n_discriminative'] - metrics1['n_discriminative']
    
    report.append(f"\nMax difference change:       {diff_max:+.6f} ({model2_name} vs {model1_name})")
    report.append(f"Separability ratio change:   {diff_sep:+.4f}")
    report.append(f"Discriminative features:     {diff_disc:+d}")
    
    report.append("\n" + "="*80)
    report.append("INTERPRETATION")
    report.append("="*80)
    
    if abs(diff_sep) < 0.1 and abs(diff_max) < 0.002:
        report.append("\n✅ BOTH models produce SIMILAR quantum state separability!")
        report.append("   → The VQC itself is not the differentiating factor")
        report.append("   → Problem is likely in how features are used downstream")
        report.append(f"\n   Since {model2_name} works but {model1_name} doesn't:")
        report.append("   • P3 (direct measurement) aggregates 512 features → 2 outputs")
        report.append("   • P2 (feature extraction) gives all 512/1024 features to MLP")
        report.append("   • MLP cannot find signal in high-dimensional weak features")
    elif metrics2['separability_ratio'] > metrics1['separability_ratio'] + 0.2:
        report.append(f"\n📈 {model2_name} produces MORE SEPARABLE states!")
        report.append(f"   Separability: {metrics2['separability_ratio']:.2f} vs {metrics1['separability_ratio']:.2f}")
        report.append(f"   → {model2_name}'s VQC learned to encode classes better")
    else:
        report.append(f"\n📉 {model1_name} produces MORE SEPARABLE states!")
        report.append(f"   Separability: {metrics1['separability_ratio']:.2f} vs {metrics2['separability_ratio']:.2f}")
        report.append(f"   → But {model1_name} still failed to classify correctly")
        report.append("   → MLP cannot learn from these quantum features")
    
    report_text = "\n".join(report)
    print(report_text)
    
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n💾 Saved {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantum features between two models")
    
    parser.add_argument('--model1_path', type=str, required=True,
                       help='Path to Model 1 (e.g., P2b that failed)')
    parser.add_argument('--model2_path', type=str, required=True,
                       help='Path to Model 2 (e.g., P3 that worked)')
    parser.add_argument('--model1_name', type=str, default='P2 (Failed)',
                       help='Display name for Model 1')
    parser.add_argument('--model2_name', type=str, default='P3 (Worked)',
                       help='Display name for Model 2')
    parser.add_argument('--dataset', type=str, default='retinamnist',
                       choices=['retinamnist', 'breastmnist', 'pneumoniamnist'],
                       help='Dataset to use')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save comparison results')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples per class')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " QUANTUM FEATURE COMPARISON".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load both models
    print(f"\n{'='*80}")
    print(f"LOADING MODELS")
    print(f"{'='*80}")
    
    model1, config1 = load_model_from_checkpoint(args.model1_path, device, args.model1_name)
    model2, config2 = load_model_from_checkpoint(args.model2_path, device, args.model2_name)
    
    # Load test data
    dataset_name = config1.get('dataset', args.dataset)
    print(f"\n📊 Loading {dataset_name.upper()} test dataset...")
    
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
    
    # Extract features from both models
    features1 = extract_quantum_features(model1, test_loader, device, 
                                        args.n_samples, args.model1_name)
    features2 = extract_quantum_features(model2, test_loader, device, 
                                        args.n_samples, args.model2_name)
    
    # Compute metrics
    print(f"\n📈 Computing separability metrics...")
    metrics1 = compute_separability_metrics(features1)
    metrics2 = compute_separability_metrics(features2)
    
    # Create comparison
    create_comparison_report(metrics1, metrics2, args.model1_name, args.model2_name, args.save_dir)
    create_comparison_plots(features1, features2, metrics1, metrics2, 
                           args.model1_name, args.model2_name, args.save_dir)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " COMPARISON COMPLETE!".center(78) + "║")
    print("╚" + "="*78 + "╝")
    print(f"\n📁 Results saved to: {args.save_dir}\n")


if __name__ == "__main__":
    main()

# python /scratch90/chris/qc2025_final_proj/src/troubleshooting_scripts/compare_quantum_features.py \
#     --model1_path /scratch90/chris/qc2025_final_proj/results/results_retinamnist_n10_dual_lr/p2b_complex_amplitudes/run_1 \
#     --model2_path /scratch90/chris/qc2025_final_proj/results/results_retinamnist_n10_dual_lr/p3_no_noise/run_1 \
#     --model1_name "P2b (Complex Amplitudes)" \
#     --model2_name "P3 (Direct Classification)" \
#     --dataset retinamnist \
#     --save_dir /scratch90/chris/qc2025_final_proj/graphics_misc/comparison_p2b_vs_p3 \
#     --n_samples 100