"""
loader.py

Unified loader for MedMNIST datasets.
Supports:
- RetinaMNIST: Binary (0 vs 3+4) or Multiclass (5 classes)
- BreastMNIST: Binary (normal/benign vs malignant)
- PneumoniaMNIST: Binary (normal vs pneumonia)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import RetinaMNIST, BreastMNIST, PneumoniaMNIST
import numpy as np


class BinarySubset(Dataset):
    """
    Binary subset wrapper for any MedMNIST dataset.
    """
    def __init__(self, base_dataset, indices, relabeled_labels):
        self.dataset = base_dataset
        self.indices = indices
        self.labels = torch.tensor(relabeled_labels, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        return img, self.labels[idx]


def get_medmnist_loaders(
    dataset_name='retinamnist',
    img_size=(224, 224),
    batch_size=4,
    binary=True,
    balance_classes=True,
    n_train=None,
    n_val=None, 
    n_test=None,
    num_workers=4
):
    """
    Load MedMNIST dataset for binary or multiclass classification.
    
    Args:
        dataset_name (str): 'retinamnist', 'breastmnist', or 'pneumoniamnist'
        img_size (tuple): Image size for resizing
        batch_size (int): Batch size
        binary (bool): If True, binary classification. If False, multiclass (RetinaMNIST only).
        balance_classes (bool): Undersample to balance classes
        n_train, n_val, n_test (int or None): Optional subsample counts per split
        num_workers (int): DataLoader workers
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    
    # Validate dataset choice
    valid_datasets = ['retinamnist', 'breastmnist', 'pneumoniamnist']
    if dataset_name not in valid_datasets:
        raise ValueError(f"dataset_name must be one of {valid_datasets}, got '{dataset_name}'")
    
    # Force binary for BreastMNIST and PneumoniaMNIST
    if dataset_name in ['breastmnist', 'pneumoniamnist'] and not binary:
        print(f"⚠️  {dataset_name} only supports binary classification. Setting binary=True.")
        binary = True
    
    print(f"\n{'='*80}")
    print(f"LOADING {dataset_name.upper()} DATASET")
    print(f"{'='*80}\n")
    
    # Define transforms
    # RetinaMNIST is RGB (28x28x3), BreastMNIST and PneumoniaMNIST are grayscale (28x28)
    if dataset_name == 'retinamnist':
        # RetinaMNIST is already RGB
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # BreastMNIST and PneumoniaMNIST are grayscale - convert to RGB
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],     # Simpler normalization for medical
                               std=[0.5, 0.5, 0.5])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        ])
    
    # Load datasets based on dataset_name
    if dataset_name == 'retinamnist':
        DatasetClass = RetinaMNIST
    elif dataset_name == 'breastmnist':
        DatasetClass = BreastMNIST
    elif dataset_name == 'pneumoniamnist':
        DatasetClass = PneumoniaMNIST
    
    train_dataset = DatasetClass(split='train', transform=train_transform, download=True)
    val_dataset = DatasetClass(split='val', transform=val_test_transform, download=True)
    test_dataset = DatasetClass(split='test', transform=val_test_transform, download=True)
    
    print(f"✅ Downloaded {dataset_name}")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    def filter_and_subsample(dataset, split_name, n_samples=None):
        """Filter to binary if needed, balance, and subsample."""
        labels_np = dataset.labels.squeeze()
        all_indices = np.arange(len(labels_np))
        
        if binary:
            if dataset_name == 'retinamnist':
                # RetinaMNIST: Binary classification (class 0 vs classes 3+4)
                keep_mask = (labels_np == 0) | (labels_np == 3) | (labels_np == 4)
                filtered_indices = all_indices[keep_mask]
                filtered_labels = labels_np[filtered_indices]
                relabeled = np.copy(filtered_labels)
                relabeled[(relabeled == 3) | (relabeled == 4)] = 1
                num_classes = 2
                class_names = ['Class 0', 'Classes 3+4']
            
            elif dataset_name == 'breastmnist':
                # BreastMNIST: Already binary (0=normal/benign, 1=malignant)
                filtered_indices = all_indices
                relabeled = labels_np
                num_classes = 2
                class_names = ['Normal/Benign', 'Malignant']
            
            elif dataset_name == 'pneumoniamnist':
                # PneumoniaMNIST: Already binary (0=normal, 1=pneumonia)
                filtered_indices = all_indices
                relabeled = labels_np
                num_classes = 2
                class_names = ['Normal', 'Pneumonia']
        
        else:
            # Multiclass mode (only RetinaMNIST supports this)
            if dataset_name != 'retinamnist':
                raise ValueError(f"Multiclass mode only supported for RetinaMNIST")
            filtered_indices = all_indices
            relabeled = labels_np
            num_classes = 5
            class_names = [f'Class {i}' for i in range(5)]
        
        def summarize_class_counts(labels):
            unique, counts = np.unique(labels, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}
        
        class_dist = summarize_class_counts(relabeled)
        print(f"\n📊 {split_name.upper()} - {len(relabeled)} samples")
        print(f"   Class distribution: {class_dist}")
        
        # Balance classes if requested
        if balance_classes:
            class_indices = {c: filtered_indices[relabeled == c] for c in np.unique(relabeled)}
            min_len = min(len(idx) for idx in class_indices.values())
            balanced_indices = np.concatenate([idx[:min_len] for idx in class_indices.values()])
            relabeled_balanced = np.concatenate([
                relabeled[relabeled == c][:min_len] for c in np.unique(relabeled)
            ])
            
            balanced_dist = summarize_class_counts(relabeled_balanced)
            print(f"   ✅ Balanced: {balanced_dist}")
            filtered_indices = balanced_indices
            relabeled = relabeled_balanced
        
        # Subsample if requested
        if n_samples and len(filtered_indices) > n_samples:
            shuffle_idx = np.random.permutation(len(filtered_indices))[:n_samples]
            filtered_indices = filtered_indices[shuffle_idx]
            relabeled = relabeled[shuffle_idx]
            
            subsample_dist = summarize_class_counts(relabeled)
            print(f"   ✅ Subsampled to {len(filtered_indices)}: {subsample_dist}")
        
        if binary:
            return BinarySubset(dataset, filtered_indices, relabeled), num_classes
        else:
            # For multiclass, create subset with indices
            subset = torch.utils.data.Subset(dataset, filtered_indices)
            subset.labels = relabeled
            return subset, num_classes
    
    # Build loaders
    train_subset, num_classes = filter_and_subsample(train_dataset, "train", n_samples=n_train)
    val_subset, _ = filter_and_subsample(val_dataset, "val", n_samples=n_val)
    test_subset, _ = filter_and_subsample(test_dataset, "test", n_samples=n_test)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    print(f"\n{'='*80}")
    print(f"FINAL LOADERS SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Mode: {'Binary' if binary else 'Multiclass'}")
    print(f"Balanced: {balance_classes}")
    print(f"Train: {len(train_subset)} samples")
    print(f"Val:   {len(val_subset)} samples")
    print(f"Test:  {len(test_subset)} samples")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, test_loader, num_classes


def test_all_datasets():
    """Test the data loader with all 3 datasets."""
    
    datasets_to_test = [
        ('retinamnist', True, 'RetinaMNIST Binary (0 vs 3+4)'),
        ('breastmnist', True, 'BreastMNIST Binary'),
        ('pneumoniamnist', True, 'PneumoniaMNIST Binary')
    ]
    
    for dataset_name, binary, description in datasets_to_test:
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + f" TESTING: {description}".center(78) + "║")
        print("╚" + "="*78 + "╝\n")
        
        try:
            train_loader, val_loader, test_loader, num_classes = get_medmnist_loaders(
                dataset_name=dataset_name,
                img_size=(224, 224),
                batch_size=4,
                binary=binary,
                balance_classes=True,
                n_train=100 if dataset_name != 'breastmnist' else None,  # Use all for tiny dataset
                num_workers=0
            )
            
            print(f"\n🔍 Examining a batch from train_loader:")
            images, labels = next(iter(train_loader))
            print(f"   Batch shape: {images.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels: {labels.tolist()}")
            print(f"   Unique labels: {torch.unique(labels).tolist()}")
            print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   ✅ SUCCESS!")
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Test RetinaMNIST multiclass
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " TESTING: RetinaMNIST Multiclass (5 classes)".center(78) + "║")
    print("╚" + "="*78 + "╝\n")
    
    try:
        train_loader, val_loader, test_loader, num_classes = get_medmnist_loaders(
            dataset_name='retinamnist',
            img_size=(224, 224),
            batch_size=4,
            binary=False,
            balance_classes=True,
            n_train=100,
            num_workers=0
        )
        
        print(f"\n🔍 Examining a batch from train_loader:")
        images, labels = next(iter(train_loader))
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels: {labels.tolist()}")
        print(f"   Unique labels: {torch.unique(labels).tolist()}")
        print(f"   ✅ SUCCESS!")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " ALL TESTS COMPLETE!".center(78) + "║")
    print("╚" + "="*78 + "╝\n")


if __name__ == "__main__":
    test_all_datasets()