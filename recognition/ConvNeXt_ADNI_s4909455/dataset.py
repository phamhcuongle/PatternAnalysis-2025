from pathlib import Path
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def _get_subject_id(fname: str) -> str:
    return Path(fname).stem.split('_')[0]


class ADNIDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_subject_splits(root: str, train_dir_name='train', val_ratio=0.15, seed=42):
    root = Path(root)
    train_root = root / 'AD_NC' / train_dir_name
    classes = ['AD', 'NC']

    subjects_per_class = {}
    files_per_subject = {}

    for cls in classes:
        cls_dir = train_root / cls
        if not cls_dir.exists():
            continue
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        subj_map = {}
        for f in files:
            sid = _get_subject_id(f.name)
            subj_map.setdefault(sid, []).append(f)
        subjects_per_class[cls] = list(subj_map.keys())
        files_per_subject.update({(cls, sid): subj_map[sid] for sid in subj_map})

    random.seed(seed)
    train_items = []
    val_items = []
    for cls in classes:
        sids = subjects_per_class.get(cls, [])
        random.shuffle(sids)
        n_val = max(1, int(len(sids) * val_ratio))
        val_sids = set(sids[:n_val])
        label = 0 if cls == 'AD' else 1
        for sid in sids:
            for p in files_per_subject[(cls, sid)]:
                if sid in val_sids:
                    val_items.append((p, label))
                else:
                    train_items.append((p, label))

    return train_items, val_items


def build_dataloaders(root: str, batch_size: int = 64, image_size: int = 224, val_ratio: float = 0.15,
                      num_workers: int = 4, seed: int = 42):
    train_items, val_items = build_subject_splits(root, val_ratio=val_ratio, seed=seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.RandAugment(num_ops=2, magnitude=9)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_ds = ADNIDataset(train_items, transform=train_transform)
    val_ds = ADNIDataset(val_items, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./ADNI')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    
    if not Path(args.root).exists():
        print(f"Error: Data directory not found at {args.root}")
    else:
        tr, va = build_dataloaders(args.root, batch_size=args.batch_size, image_size=224, num_workers=args.num_workers)
        print(f'train batches: {len(tr)}, val batches: {len(va)}')