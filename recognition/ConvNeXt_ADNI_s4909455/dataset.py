"""Dataset and dataloader utilities for the ADNI 2-class (AD vs NC) task.

This module provides:
- `ADNIDataset`: image-level dataset with optional subject-aware filtering
- `create_subject_split`: subject-level split to avoid leakage across splits
- `build_transform`: train/eval torchvision transforms
- `build_loader`: ready-to-use train/val/test dataloaders

Notes
- Subject IDs are inferred from the filename prefix before the first underscore.
- Train/val splits are created at the subject level to prevent leakage.
"""

import os
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import numpy as np


class ADNIDataset(Dataset):
    """ADNI dataset wrapper over folder-structured jpeg images.

    Args:
        root_dir: Root directory containing `train/` and `test/` subfolders.
        split: One of {'train', 'val', 'test'}. For 'val', pass `subject_ids`.
        transform: Optional torchvision transform to apply to each image.
        subject_ids: Optional set of subject IDs to include (subject-level split).

    The dataset records samples as (image_path, label_index) pairs where
    label_index is 0 for 'NC' and 1 for 'AD'.
    """
    def __init__(self, root_dir, split='train', transform=None, subject_ids=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'NC': 0, 'AD': 1}
        
        if split == 'test':
            self._load_test_data()
        else:
            # For train/val we expect caller to pre-select subject IDs to
            # avoid subject leakage across splits.
            self._load_train_val_data(subject_ids)
    
    def _load_test_data(self):
        """Populate `self.samples` from the test directory."""
        test_dir = self.root_dir / 'test'
        for class_name in ['AD', 'NC']:
            class_dir = test_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.jpeg')):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def _load_train_val_data(self, subject_ids):
        """Populate `self.samples` from train directory, optionally filtering by subject.

        Only images whose subject prefix exists in `subject_ids` (if provided)
        are included.
        """
        train_dir = self.root_dir / 'train'
        for class_name in ['AD', 'NC']:
            class_dir = train_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.jpeg')):
                    subject_id = self._extract_subject_id(img_path.name)
                    if subject_ids is None or subject_id in subject_ids:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    @staticmethod
    def _extract_subject_id(filename):
        """Return subject identifier parsed from filename prefix before first underscore."""
        return filename.split('_')[0]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load image and return `(image_tensor, label_index)`.

        Images are converted to RGB before transforms for consistency.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_subject_split(root_dir, train_ratio=0.9, seed=42):
    """Create subject-level train/val split to prevent leakage.

    Returns two sets of subject IDs: (train_subjects, val_subjects).
    The split is stratified per class by splitting each class' subjects
    independently then combining.
    """
    train_dir = Path(root_dir) / 'train'
    subject_ids_per_class = defaultdict(set)
    
    for class_name in ['AD', 'NC']:
        class_dir = train_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*.jpeg'):
                subject_id = ADNIDataset._extract_subject_id(img_path.name)
                subject_ids_per_class[class_name].add(subject_id)
    
    train_subjects = []
    val_subjects = []
    
    np.random.seed(seed)
    for class_name, subject_ids in subject_ids_per_class.items():
        subject_list = sorted(list(subject_ids))
        np.random.shuffle(subject_list)
        
        n_train = int(len(subject_list) * train_ratio)
        train_subjects.extend(subject_list[:n_train])
        val_subjects.extend(subject_list[n_train:])
    
    return set(train_subjects), set(val_subjects)


def build_transform(is_train, input_size=224):
    """Return torchvision transform pipeline for train or eval.

    Train augments use RandAugment and RandomErasing for regularization;
    eval uses deterministic resize + normalization only.
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=9, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def build_loader(root_dir, batch_size=32, num_workers=4, input_size=224, seed=42):
    """Create train/val/test dataloaders with subject-level split.

    The validation set is derived from the training directory by holding out
    subjects; the test set is loaded from the `test/` directory.
    """
    train_subjects, val_subjects = create_subject_split(root_dir, train_ratio=0.9, seed=seed)
    
    train_transform = build_transform(is_train=True, input_size=input_size)
    val_transform = build_transform(is_train=False, input_size=input_size)
    
    train_dataset = ADNIDataset(root_dir, split='train', transform=train_transform, subject_ids=train_subjects)
    val_dataset = ADNIDataset(root_dir, split='train', transform=val_transform, subject_ids=val_subjects)
    test_dataset = ADNIDataset(root_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader