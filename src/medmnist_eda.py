"""
explore_medmnist_datasets.py

Explore BreastMNIST and PneumoniaMNIST datasets to understand their properties
before integrating into the training pipeline.

Usage:
    python medmnist_eda.py --save_dir /path/to/output
"""

import numpy as np
import matplotlib.pyplot as plt
from medmnist import BreastMNIST, PneumoniaMNIST, RetinaMNIST
import torch
from torch.utils.data import DataLoader
import argparse
import os

def explore_dataset(dataset_class, dataset_name, save_dir='.'):
    """
    Explore a MedMNIST dataset and print comprehensive statistics.
    
    Args:
        dataset_class: The MedMNIST dataset class (e.g., BreastMNIST)
        dataset_name: Name for display (e.g., "BreastMNIST")
        save_dir: Directory to save output images
    """
    print("\n" + "="*80)
    print(f"EXPLORING {dataset_name}")
    print("="*80)
    
    # Load all splits
    train_data = dataset_class(split='train', download=True)
    val_data = dataset_class(split='val', download=True)
    test_data = dataset_class(split='test', download=True)
    
    print(f"\n📊 DATASET SPLITS:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")
    print(f"   Total: {len(train_data) + len(val_data) + len(test_data)} samples")
    
    # Get a sample to check dimensions
    sample_img, sample_label = train_data[0]
    sample_img = np.array(sample_img)
    
    print(f"\n🖼️  IMAGE PROPERTIES:")
    print(f"   Shape: {sample_img.shape}")
    print(f"   Dtype: {sample_img.dtype}")
    print(f"   Range: [{sample_img.min()}, {sample_img.max()}]")
    
    if len(sample_img.shape) == 3:
        channels = sample_img.shape[2]
        print(f"   Channels: {channels}")
        if channels == 1:
            print(f"   Type: Grayscale")
        elif channels == 3:
            print(f"   Type: RGB")
        else:
            print(f"   Type: {channels}-channel")
    else:
        print(f"   Type: 2D Grayscale")
    
    print(f"   Spatial: {sample_img.shape[0]}x{sample_img.shape[1]}")
    
    # Label analysis for each split
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        labels = []
        for i in range(len(split_data)):
            _, label = split_data[i]
            labels.append(label)
        
        labels = np.array(labels).flatten()
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print(f"\n🏷️  {split_name.upper()} LABELS:")
        print(f"   Unique classes: {unique_labels.tolist()}")
        print(f"   Class counts:")
        for label, count in zip(unique_labels, counts):
            percentage = 100 * count / len(labels)
            print(f"      Class {label}: {count:5d} samples ({percentage:5.2f}%)")
        
        # Check balance
        if len(unique_labels) == 2:
            imbalance_ratio = max(counts) / min(counts)
            print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 2:
                print(f"   ⚠️  WARNING: Significant class imbalance!")
            else:
                print(f"   ✅ Reasonably balanced")
    
    # Visualize samples - ONE EXAMPLE PER CLASS
    print(f"\n🎨 GENERATING SAMPLE VISUALIZATION...")
    
    # Find one example of each class
    class_examples = {}
    for idx in range(len(train_data)):
        img, label = train_data[idx]
        label_val = int(label) if isinstance(label, (int, np.integer)) else int(label[0])
        
        if label_val not in class_examples:
            class_examples[label_val] = (np.array(img), label_val)
        
        # Stop once we have examples of all classes
        if len(class_examples) >= 2:
            break
    
    # Create figure with samples and stats
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
    
    fig.suptitle(f'{dataset_name} - Dataset Overview', fontsize=16, fontweight='bold')
    
    # Plot class examples
    for i, (class_label, (img, _)) in enumerate(sorted(class_examples.items())):
        ax = fig.add_subplot(gs[0, i])
        
        # Handle grayscale vs RGB
        if len(img.shape) == 3 and img.shape[2] == 1:
            img_display = img.squeeze()
            ax.imshow(img_display, cmap='gray')
        elif len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        ax.set_title(f'Class {class_label}', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    # Add statistics text box at bottom
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    
    # Get train statistics
    train_labels = []
    for i in range(len(train_data)):
        _, label = train_data[i]
        train_labels.append(label)
    train_labels = np.array(train_labels).flatten()
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Create stats text
    stats_text = f"Image Size: {sample_img.shape[0]}×{sample_img.shape[1]}"
    if len(sample_img.shape) == 3:
        channels = sample_img.shape[2]
        if channels == 1:
            stats_text += " (Grayscale)"
        elif channels == 3:
            stats_text += " (RGB)"
    else:
        stats_text += " (Grayscale)"
    
    stats_text += f"\n\nTraining Set:\n"
    for label, count in zip(unique_labels, counts):
        percentage = 100 * count / len(train_labels)
        stats_text += f"  Class {label}: {count:,} samples ({percentage:.1f}%)\n"
    
    stats_text += f"\nTotal Train: {len(train_data):,} samples"
    stats_text += f"  |  Val: {len(val_data):,}  |  Test: {len(test_data):,}"
    
    ax_text.text(0.5, 0.5, stats_text, 
                ha='center', va='center',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{dataset_name.lower()}_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()
    
    # Additional metadata if available
    print(f"\n📋 DATASET METADATA:")
    if hasattr(train_data, 'info'):
        info = train_data.info
        print(f"   Description: {info.get('description', 'N/A')}")
        print(f"   Task: {info.get('task', 'N/A')}")
        print(f"   Modality: {info.get('modality', 'N/A')}")
        print(f"   License: {info.get('license', 'N/A')}")
    
    return train_data, val_data, test_data


def compare_datasets(breast_train, pneumonia_train):
    """
    Compare the two datasets side by side.
    """
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    # Get samples
    breast_img, breast_label = breast_train[0]
    pneumonia_img, pneumonia_label = pneumonia_train[0]
    
    breast_img = np.array(breast_img)
    pneumonia_img = np.array(pneumonia_img)
    
    print(f"\n🔍 SHAPE COMPARISON:")
    print(f"   BreastMNIST:     {breast_img.shape}")
    print(f"   PneumoniaMNIST:  {pneumonia_img.shape}")
    
    print(f"\n📏 SIZE COMPARISON:")
    print(f"   BreastMNIST:     {len(breast_train)} train samples")
    print(f"   PneumoniaMNIST:  {len(pneumonia_train)} train samples")
    print(f"   Ratio:           {len(pneumonia_train)/len(breast_train):.1f}x larger")
    
    # Check if preprocessing needed
    print(f"\n🔧 PREPROCESSING REQUIREMENTS:")
    
    if breast_img.shape != pneumonia_img.shape:
        print(f"   ⚠️  Different shapes - both will be resized to 224x224 for ResNet")
    
    if len(breast_img.shape) == 2 or (len(breast_img.shape) == 3 and breast_img.shape[2] == 1):
        print(f"   ⚠️  BreastMNIST is grayscale - will need RGB conversion")
    
    if len(pneumonia_img.shape) == 2 or (len(pneumonia_img.shape) == 3 and pneumonia_img.shape[2] == 1):
        print(f"   ⚠️  PneumoniaMNIST is grayscale - will need RGB conversion")


def test_dataloader_compatibility(dataset, dataset_name):
    """
    Test if dataset works with PyTorch DataLoader.
    """
    print(f"\n🧪 TESTING DATALOADER COMPATIBILITY ({dataset_name})...")
    
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_imgs, batch_labels = next(iter(loader))
        
        print(f"   ✅ DataLoader works!")
        print(f"   Batch images shape: {batch_imgs.shape}")
        print(f"   Batch labels shape: {batch_labels.shape}")
        print(f"   Batch images dtype: {batch_imgs.dtype}")
        print(f"   Batch labels dtype: {batch_labels.dtype}")
        
    except Exception as e:
        print(f"   ❌ DataLoader failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore MedMNIST datasets")
    parser.add_argument('--save_dir', type=str, default='.',
                       help='Directory to save output images (default: current directory)')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "MedMNIST DATASET EXPLORER" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print(f"\n📁 Saving outputs to: {args.save_dir}\n")
    
    # Explore RetinaMNIST
    retina_train, retina_val, retina_test = explore_dataset(RetinaMNIST, "RetinaMNIST", args.save_dir)
    test_dataloader_compatibility(retina_train, "RetinaMNIST")
    
    # Explore BreastMNIST
    breast_train, breast_val, breast_test = explore_dataset(BreastMNIST, "BreastMNIST", args.save_dir)
    test_dataloader_compatibility(breast_train, "BreastMNIST")
    
    # Explore PneumoniaMNIST
    pneumonia_train, pneumonia_val, pneumonia_test = explore_dataset(PneumoniaMNIST, "PneumoniaMNIST", args.save_dir)
    test_dataloader_compatibility(pneumonia_train, "PneumoniaMNIST")
    
    # Compare (show all three)
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    print(f"\n📏 SIZE COMPARISON:")
    print(f"   RetinaMNIST:     {len(retina_train):,} train samples")
    print(f"   BreastMNIST:     {len(breast_train):,} train samples")
    print(f"   PneumoniaMNIST:  {len(pneumonia_train):,} train samples")
    
    # Get sample images
    retina_img, _ = retina_train[0]
    breast_img, _ = breast_train[0]
    pneumonia_img, _ = pneumonia_train[0]
    
    retina_img = np.array(retina_img)
    breast_img = np.array(breast_img)
    pneumonia_img = np.array(pneumonia_img)
    
    print(f"\n🔍 SHAPE COMPARISON:")
    print(f"   RetinaMNIST:     {retina_img.shape}")
    print(f"   BreastMNIST:     {breast_img.shape}")
    print(f"   PneumoniaMNIST:  {pneumonia_img.shape}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n✅ ALL THREE DATASETS ARE BINARY CLASSIFICATION TASKS")
    print(f"\n📝 DATASET PROPERTIES:")
    print(f"   - RetinaMNIST:     RGB fundus images, multiclass (5) → binary (0 vs 3+4)")
    print(f"   - BreastMNIST:     Grayscale ultrasound, small dataset (546 train)")
    print(f"   - PneumoniaMNIST:  Grayscale X-ray, large dataset (4,708 train)")
    print(f"\n💡 EXPECTED RESULTS:")
    print(f"   - RetinaMNIST:     Baseline performance")
    print(f"   - BreastMNIST:     Smaller → May need more regularization / higher variance")
    print(f"   - PneumoniaMNIST:  Larger → Should be more stable, better generalization")
    
    print("\n" + "="*80)
    print("✅ EXPLORATION COMPLETE!")
    print(f"Generated: {os.path.join(args.save_dir, 'retinamnist_overview.png')}")
    print(f"Generated: {os.path.join(args.save_dir, 'breastmnist_overview.png')}")
    print(f"Generated: {os.path.join(args.save_dir, 'pneumoniamnist_overview.png')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

# python /scratch90/chris/qc2025_final_proj/src/medmnist_eda.py --save_dir /scratch90/chris/qc2025_final_proj/graphics_misc