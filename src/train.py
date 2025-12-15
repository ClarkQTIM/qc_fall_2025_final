"""
train.py

Training script for hybrid CNN-VQC feature extraction experiments.
Supports binary and multiclass classification with appropriate metrics.
Supports multiple MedMNIST datasets: RetinaMNIST, BreastMNIST, PneumoniaMNIST.
NOW WITH DUAL LEARNING RATES for quantum and classical parameters!
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, 
    roc_auc_score, 
    accuracy_score,
    roc_curve,
    auc
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import HybridModel
from src.loader import get_medmnist_loaders


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.squeeze().to(device).long()  # Squeeze for multiclass
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.squeeze().to(device).long()
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def compute_binary_metrics(all_labels, all_predictions, all_probs):
    """
    Compute metrics for binary classification.
    
    Returns:
        dict with accuracy, auc
    """
    metrics = {}
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    metrics['accuracy'] = accuracy
    
    # AUC (using probability of positive class)
    auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    metrics['auc'] = auc_score
    
    return metrics


def compute_multiclass_metrics(all_labels, all_predictions, all_probs, num_classes):
    """
    Compute metrics for multiclass classification.
    
    Returns:
        dict with accuracy, linear_weighted_kappa, per_class_auc, average_auc
    """
    metrics = {}
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    metrics['accuracy'] = accuracy
    
    # Linear Weighted Kappa (for ordinal classification)
    kappa = cohen_kappa_score(all_labels, all_predictions, weights='linear')
    metrics['linear_weighted_kappa'] = kappa
    
    # Per-class AUC (One-vs-Rest)
    per_class_auc = {}
    auc_scores = []
    
    for class_idx in range(num_classes):
        try:
            # Binary labels: 1 if class_idx, 0 otherwise
            binary_labels = (all_labels == class_idx).astype(int)
            class_probs = all_probs[:, class_idx]
            
            # Only compute if we have both classes in test set
            if len(np.unique(binary_labels)) > 1:
                class_auc = roc_auc_score(binary_labels, class_probs)
                per_class_auc[f'class_{class_idx}_auc'] = class_auc
                auc_scores.append(class_auc)
            else:
                per_class_auc[f'class_{class_idx}_auc'] = None
        except Exception as e:
            print(f"⚠️ Could not compute AUC for class {class_idx}: {e}")
            per_class_auc[f'class_{class_idx}_auc'] = None
    
    # Average AUC (macro average)
    if auc_scores:
        metrics['average_auc'] = np.mean(auc_scores)
    else:
        metrics['average_auc'] = None
    
    # Add per-class AUCs to metrics
    metrics.update(per_class_auc)
    
    return metrics


def test_and_save_predictions(model, test_loader, device, save_path, binary=True):
    """
    Test the model and save predictions with probabilities.
    Computes appropriate metrics based on binary vs multiclass.
    
    Args:
        binary: If True, compute binary metrics. Otherwise multiclass.
    
    Returns:
        dict of metrics
    """
    model.eval()
    
    results = {
        'image_idx': [],
        'true_label': [],
        'predicted_label': [],
    }
    
    # Add columns for each class probability
    num_classes = model.num_classes
    for i in range(num_classes):
        results[f'prob_class_{i}'] = []
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            labels = labels.squeeze().to(device).long()
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Store results for CSV
            batch_size = images.size(0)
            for i in range(batch_size):
                global_idx = batch_idx * test_loader.batch_size + i
                results['image_idx'].append(global_idx)
                results['true_label'].append(labels[i].item())
                results['predicted_label'].append(predicted[i].item())
                
                # Store probabilities for each class
                for c in range(num_classes):
                    results[f'prob_class_{c}'].append(probs[i, c].item())
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.vstack(all_probs)
    
    # Compute appropriate metrics
    if binary:
        metrics = compute_binary_metrics(all_labels, all_predictions, all_probs)
        print(f"\n🎯 Binary Classification Results:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   AUC: {metrics['auc']:.4f}")
    else:
        metrics = compute_multiclass_metrics(all_labels, all_predictions, all_probs, num_classes)
        print(f"\n🎯 Multiclass Classification Results:")
        print(f"   Accuracy: {metrics['accuracy']:.2f}%")
        print(f"   Linear Weighted Kappa: {metrics['linear_weighted_kappa']:.4f}")
        print(f"   Average AUC: {metrics.get('average_auc', 'N/A'):.4f}" if metrics.get('average_auc') else "   Average AUC: N/A")
        print(f"   Per-class AUC:")
        for i in range(num_classes):
            auc_val = metrics.get(f'class_{i}_auc')
            if auc_val is not None:
                print(f"      Class {i}: {auc_val:.4f}")
            else:
                print(f"      Class {i}: N/A")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\n📊 Test predictions saved to: {save_path}")
    
    return metrics


def plot_losses(train_losses, val_losses, save_path):
    """Plot and save training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-VQC Model")
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='retinamnist',
                       choices=['retinamnist', 'breastmnist', 'pneumoniamnist'],
                       help='Which MedMNIST dataset to use (default: retinamnist)')
    parser.add_argument('--binary', action='store_true',
                        help='Use binary classification (0 vs 3+4 for RetinaMNIST). Default is multiclass.')
    parser.add_argument('--balance_classes', action='store_true',
                        help='Balance classes by undersampling')
    
    # Model configuration
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability in classifier (default: 0.5)')
    parser.add_argument('--use_quantum', action='store_true', 
                        help='Use quantum feature extraction')
    parser.add_argument('--n_qubits', type=int, default=9,
                        help='Number of qubits (default: 9 for 512-dim features)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of VQC layers')
    parser.add_argument('--vqc_type', type=str, default='basic',
                        choices=['basic', 'expressive'],
                        help='Type of VQC circuit')
    parser.add_argument('--feature_transform', type=str, default='probabilities',
                    choices=['probabilities', 'complex_amplitudes', 'pauli', 'density_matrix'],
                    help='Feature extraction mode (only for feature_extractor mode)')
    parser.add_argument('--vqc_mode', type=str, default='feature_extractor',
                        choices=['feature_extractor', 'classifier'],
                        help='VQC mode: feature_extractor or classifier (only used if use_quantum=True)')
    
    # Noise parameters
    parser.add_argument('--bit_flip_prob', type=float, default=0.0,
                        help='Bit flip noise probability')
    parser.add_argument('--amp_damp_prob', type=float, default=0.0,
                        help='Amplitude damping probability')
    parser.add_argument('--depol_prob', type=float, default=0.0,
                        help='Depolarizing noise probability')
    parser.add_argument('--shots', type=int, default=None,
                        help='Number of shots (None = statevector)')
    
    # Training parameters
    parser.add_argument('--n_train', type=int, default=1000,
                        help='Number of training samples (use None or -1 for all, especially for BreastMNIST)')
    parser.add_argument('--n_val', type=int, default=None,
                        help='Number of validation samples (None = use all)')
    parser.add_argument('--n_test', type=int, default=None,
                        help='Number of test samples (None = use all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=25,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (used for classical-only models)')
    parser.add_argument('--learning_rate_quantum', type=float, default=0.1,
                        help='Learning rate for quantum parameters (VQC weights)')
    parser.add_argument('--learning_rate_classical', type=float, default=0.001,
                        help='Learning rate for classical parameters (MLP weights)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--run_id', type=int, default=1,
                        help='Run ID for multiple runs')
    
    args = parser.parse_args()
    
    # Handle n_train=None or -1 for using all samples
    if args.n_train == -1:
        args.n_train = None
    
    # Special handling for BreastMNIST (very small dataset)
    if args.dataset == 'breastmnist' and args.n_train and args.n_train > 300:
        print(f"⚠️  WARNING: BreastMNIST only has ~294 balanced samples!")
        print(f"   Setting n_train=None to use all available samples.")
        args.n_train = None
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"💾 Configuration saved to: {config_path}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print(f"\n📊 Loading {args.dataset.upper()} dataset...")
    train_loader, val_loader, test_loader, num_classes = get_medmnist_loaders(
        dataset_name=args.dataset,
        img_size=(256, 256),  # Keep 256x256 to match original RetinaMNIST setup
        batch_size=args.batch_size,
        binary=args.binary,
        balance_classes=args.balance_classes,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        num_workers=4
    )

    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = HybridModel(
        num_classes=num_classes,
        use_quantum=args.use_quantum,
        vqc_mode=args.vqc_mode,
        n_qubits=args.n_qubits,
        num_layers=args.num_layers,
        vqc_type=args.vqc_type,
        shots=args.shots,
        bit_flip_prob=args.bit_flip_prob,
        amp_damp_prob=args.amp_damp_prob,
        depol_prob=args.depol_prob,
        dropout=args.dropout,
        feature_transform=args.feature_transform
    ).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer with DUAL LEARNING RATES
    if model.use_quantum and model.vqc_mode == "feature_extractor":
        # P2: Separate learning rates for quantum and classical parts
        print(f"\n⚙️  Using DUAL learning rates:")
        print(f"   Quantum (VQC): {args.learning_rate_quantum}")
        print(f"   Classical (MLP): {args.learning_rate_classical}")
        
        quantum_params = []
        classical_params = []
        
        # Collect quantum parameters
        if model.vqc_type == "basic":
            quantum_params.append(model.vqc_weights)
        else:  # expressive
            quantum_params.extend([model.single_qubit_weights, model.entangling_weights])
        
        # Collect classical parameters
        if model.classifier is not None:
            classical_params.extend(model.classifier.parameters())
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.Adam([
            {'params': quantum_params, 'lr': args.learning_rate_quantum},
            {'params': classical_params, 'lr': args.learning_rate_classical}
        ])
    
    else:
        # P1 or P3: Single learning rate
        if model.use_quantum:
            lr = args.learning_rate_quantum
            print(f"\n⚙️  Using quantum learning rate: {lr}")
        else:
            lr = args.learning_rate
            print(f"\n⚙️  Using classical learning rate: {lr}")
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.n_epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(args.n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update loss plot
        plot_path = os.path.join(args.save_dir, 'training_curves.png')
        plot_losses(train_losses, val_losses, plot_path)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print("🔥 New best validation loss! Model saved.")
        else:
            epochs_no_improve += 1
            print(f"🚫 No improvement for {epochs_no_improve} epoch(s)")
        
        if epochs_no_improve >= args.patience:
            print(f"\n⏹️  Early stopping triggered after {args.patience} epochs with no improvement.")
            break
    
    # Load best model
    print(f"\n📥 Loading best model...")
    model.load_state_dict(best_model_state)
    
    # Save best model
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    torch.save(best_model_state, model_path)
    print(f"💾 Best model saved to: {model_path}")
    
    # Test and save predictions
    print(f"\n🧪 Testing model...")
    predictions_path = os.path.join(args.save_dir, 'test_predictions.csv')
    test_metrics = test_and_save_predictions(
        model, test_loader, device, predictions_path, binary=args.binary
    )
    
    # Save final metrics
    final_metrics = {
        'dataset': args.dataset,
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }
    final_metrics.update(test_metrics)
    
    metrics_path = os.path.join(args.save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"\n📊 Metrics saved to: {metrics_path}")
    
    print(f"\n✅ Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()