import os
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from timm.utils import ModelEma
from tqdm.notebook import tqdm

from modules import convnext_small
from dataset import build_loader


class MixupCutmix:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, num_classes=2):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.enabled = True
    
    def set_enabled(self, enabled):
        self.enabled = enabled
    
    def mixup(self, images, labels):
        """Apply MixUp augmentation"""
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        
        return mixed_images, mixed_labels
    
    def cutmix(self, images, labels):
        """Apply CutMix augmentation"""
        batch_size = images.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        index = torch.randperm(batch_size).to(images.device)
        
        _, _, h, w = images.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        labels_a = labels
        labels_b = labels[index]
        mixed_labels = lam * labels_a + (1 - lam) * labels_b
        
        return mixed_images, mixed_labels
    
    def __call__(self, images, labels):
        if not self.enabled or np.random.rand() > self.prob:
            return images, labels
        
        if np.random.rand() < self.switch_prob:
            return self.cutmix(images, labels)
        else:
            return self.mixup(images, labels)


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, mixup_cutmix=None, ema=None, label_smoothing=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).float()
        original_labels = labels.clone()
        
        if mixup_cutmix and mixup_cutmix.enabled:
            images, labels_for_loss = mixup_cutmix(images, labels)
        else:
            labels_for_loss = labels * (1 - label_smoothing) + label_smoothing / 2
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels_for_loss)
        loss.backward()
        optimizer.step()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(original_labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc


def validate(model, val_loader, criterion, device, threshold=0.5, return_probs=False):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if return_probs:
        return avg_loss, all_probs, all_labels
    
    all_preds = (all_probs > threshold).astype(float)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, f1, acc, cm


def find_optimal_threshold(probs, labels, metric='f1'):
    thresholds = np.arange(0.1, 0.95, 0.05)
    scores = []
    
    for thresh in thresholds:
        preds = (probs > thresh).astype(float)
        
        if metric == 'f1':
            score = f1_score(labels, preds)
        elif metric == 'balanced_acc':
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        elif metric == 'youden':
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1  # Youden's J statistic
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, list(zip(thresholds, scores))


def test(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return f1, acc, cm, all_probs, all_preds, all_labels


def plot_metrics(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs, 
                 test_f1s, test_accs, test_epochs, save_path):
    """Plot training, validation and test metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_f1s, 'b-', label='Train F1', linewidth=2)
    axes[1].plot(epochs, val_f1s, 'r-', label='Val F1', linewidth=2)
    if test_f1s:
        axes[1].plot(test_epochs, test_f1s, 'go-', label='Test F1', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[2].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    if test_accs:
        axes[2].plot(test_epochs, test_accs, 'go-', label='Test Acc', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Accuracy', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = build_loader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        seed=args.seed
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Determine if using pretrained head
    use_pretrained_head = args.pretrained_path and os.path.exists(args.pretrained_path)
    
    model = convnext_small(
        num_classes=1,
        drop_path_rate=args.drop_path_rate,
        layer_scale_init_value=args.layer_scale_init_value,
        use_pretrained_head=use_pretrained_head
    )
    
    # Load pretrained weights from ImageNet-22k
    if use_pretrained_head:
        print(f'Loading pretrained weights from {args.pretrained_path}')
        print(f'Using two-stage head: pretrained ImageNet-22k head (21841 classes) + adapter layer (21841→1)')
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
        
        # Get current model state dict
        model_dict = model.state_dict()
        
        # Filter out layers that don't match
        pretrained_dict_filtered = {}
        skipped_layers = []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict_filtered[k] = v
                else:
                    skipped_layers.append(f'{k}: pretrained {v.shape} vs model {model_dict[k].shape}')
            else:
                skipped_layers.append(f'{k} (not in model)')
        
        # Load the filtered pretrained weights
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        
        print(f'  Loaded {len(pretrained_dict_filtered)}/{len(pretrained_dict)} layers from pretrained checkpoint')
        
        # Count randomly initialized layers
        random_init_layers = len(model_dict) - len(pretrained_dict_filtered)
        if random_init_layers > 0:
            print(f'  Randomly initialized: {random_init_layers} layers')
            print(f'    - head_norm (LayerNorm for stability): head_norm.weight, head_norm.bias')
            print(f'    - adapter (21841→1): adapter.weight, adapter.bias')
        
        if skipped_layers and len(skipped_layers) <= 5:
            for skip_msg in skipped_layers:
                print(f'  Skipped: {skip_msg}')
    else:
        print('Training from scratch (no pretrained weights)')
    
    # Initialize the adapter bias to encourage balanced predictions
    if hasattr(model, 'adapter') and model.adapter is not None:
        nn.init.constant_(model.adapter.bias, 0.0)
    
    model = model.to(device)
    
    ema = None
    if args.use_ema:
        ema = ModelEma(model, decay=args.ema_decay, device=device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr
    )
    
    mixup_cutmix = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_cutmix = MixupCutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=1.0,
            switch_prob=0.5
        )
        print(f'Using Mixup (alpha={args.mixup_alpha}) and Cutmix (alpha={args.cutmix_alpha})')
    else:
        print('Mixup/Cutmix disabled (set --mixup_alpha and --cutmix_alpha to enable)')
    
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    test_f1s, test_accs, test_epochs = [], [], []
    best_val_f1 = 0.0
    best_val_acc = 0.0
    
    print(f'\nStarting training for {args.epochs} epochs...\n')
    
    epoch_pbar = tqdm(range(args.epochs), desc='Training Progress')
    for epoch in epoch_pbar:
        start_time = time.time()
        
        lr = scheduler.step(epoch)
        
        if mixup_cutmix is not None:
            if epoch < 100:
                mixup_cutmix.set_enabled(True)
            elif epoch < 150:
                progress = (epoch - 100) / 50
                mixup_cutmix.prob = 1.0 - progress
                mixup_cutmix.set_enabled(True)
            else:
                mixup_cutmix.set_enabled(False)
        
        train_loss, train_f1, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mixup_cutmix, ema, args.label_smoothing
        )
        
        val_model = ema.ema if ema is not None else model
        val_loss, val_f1, val_acc, val_cm = validate(val_model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        epoch_pbar.set_postfix({
            'Train_F1': f'{train_f1:.4f}',
            'Val_F1': f'{val_f1:.4f}',
            'Best_F1': f'{best_val_f1:.4f}',
            'Val_Acc': f'{val_acc:.4f}',
            'Best_Acc': f'{best_val_acc:.4f}'
        })
        
        print(f'Epoch [{epoch+1}/{args.epochs}] ({epoch_time:.2f}s) LR: {lr:.6f}')
        print(f'  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}')
        
        if val_f1 < 0.1 and epoch > args.warmup_epochs:
            print(f'  WARNING: Validation F1 very low ({val_f1:.4f}) - possible model collapse!')
        if epoch > 0 and val_f1 < val_f1s[-1] - 0.3:
            print(f'  WARNING: Validation F1 dropped significantly from {val_f1s[-1]:.4f} to {val_f1:.4f}')
        
        if val_f1 == best_val_f1:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }
            if ema is not None:
                save_dict['ema_state_dict'] = ema.ema.state_dict()
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model_f1.pth'))
            print(f'  --> New best F1 model saved! (F1: {val_f1:.4f})')
        
        if val_acc == best_val_acc:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }
            if ema is not None:
                save_dict['ema_state_dict'] = ema.ema.state_dict()
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model_acc.pth'))
            print(f'  --> New best Accuracy model saved! (Acc: {val_acc:.4f})')
        
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }
            if ema is not None:
                save_dict['ema_state_dict'] = ema.ema.state_dict()
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            print(f'\n  Evaluating best accuracy model on test set...')
            checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_acc.pth'))
            test_model = convnext_small(
                num_classes=1,
                drop_path_rate=args.drop_path_rate,
                layer_scale_init_value=args.layer_scale_init_value,
                use_pretrained_head=use_pretrained_head
            )
            if ema is not None and 'ema_state_dict' in checkpoint:
                test_model.load_state_dict(checkpoint['ema_state_dict'])
            else:
                test_model.load_state_dict(checkpoint['model_state_dict'])
            test_model = test_model.to(device)
            
            # Find optimal threshold on validation set (optimizing for F1)
            _, val_probs_temp, val_labels_temp = validate(test_model, val_loader, criterion, device, return_probs=True)
            opt_thresh, _, _ = find_optimal_threshold(val_probs_temp, val_labels_temp, metric='f1')
            
            # Evaluate on test set with optimal threshold
            test_f1, test_acc, test_cm, _, _, _ = test(test_model, test_loader, device, threshold=opt_thresh)
            test_f1s.append(test_f1)
            test_accs.append(test_acc)
            test_epochs.append(epoch + 1)
            
            print(f'  Test (thresh={opt_thresh:.3f}) - F1: {test_f1:.4f}, Acc: {test_acc:.4f}')
            
            plot_save_path = os.path.join(args.output_dir, f'training_{epoch+1}.png')
            plot_metrics(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs,
                        test_f1s, test_accs, test_epochs, plot_save_path)
            print(f'  Training plot saved to {plot_save_path}')
        
        print()
    
    print('Training completed!')
    print(f'Best validation F1: {best_val_f1:.4f}')
    print(f'Best validation Accuracy: {best_val_acc:.4f}')
    
    final_plot_path = os.path.join(args.output_dir, 'training_final.png')
    plot_metrics(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs,
                test_f1s, test_accs, test_epochs, final_plot_path)
    print(f'Final training metrics plot saved to {final_plot_path}')
    
    print('\nLoading best accuracy model for final evaluation...')
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_acc.pth'))
    if ema is not None and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print('\nOptimizing classification threshold on validation set (using F1 metric)...')
    _, val_probs, val_labels = validate(model, val_loader, criterion, device, return_probs=True)
    optimal_threshold, optimal_f1, threshold_scores = find_optimal_threshold(val_probs, val_labels, metric='f1')
    
    print(f'Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})')
    print(f'Default threshold (0.5) F1: {f1_score(val_labels, (val_probs > 0.5).astype(float)):.4f}')
    
    print(f'\nFinal evaluation on test set (threshold={optimal_threshold:.3f})...')
    test_f1, test_acc, test_cm, test_probs, test_preds, test_labels = test(model, test_loader, device, threshold=optimal_threshold)
    
    test_f1_default, test_acc_default, test_cm_default, _, test_preds_default, _ = test(model, test_loader, device, threshold=0.5)
    
    print(f'\n' + '='*70)
    print('TEST RESULTS WITH OPTIMIZED THRESHOLD')
    print('='*70)
    print(f'Threshold: {optimal_threshold:.3f}')
    print(f'F1 Score:  {test_f1:.4f}')
    print(f'Accuracy:  {test_acc:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'  {test_cm}')
    print(f'  TN: {test_cm[0, 0]}, FP: {test_cm[0, 1]}')
    print(f'  FN: {test_cm[1, 0]}, TP: {test_cm[1, 1]}')
    
    if test_cm[1, 1] + test_cm[1, 0] > 0:
        sensitivity = test_cm[1, 1] / (test_cm[1, 1] + test_cm[1, 0])
        print(f'\nSensitivity (Recall): {sensitivity:.4f}')
    
    if test_cm[0, 0] + test_cm[0, 1] > 0:
        specificity = test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1])
        print(f'Specificity:          {specificity:.4f}')
    
    print(f'\n' + '='*70)
    print('TEST RESULTS WITH DEFAULT THRESHOLD (0.5)')
    print('='*70)
    print(f'F1 Score:  {test_f1_default:.4f}')
    print(f'Accuracy:  {test_acc_default:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'  {test_cm_default}')
    print('='*70)
    
    results = {
        'optimal_threshold': float(optimal_threshold),
        'test_f1_optimized': float(test_f1),
        'test_acc_optimized': float(test_acc),
        'test_cm_optimized': test_cm.tolist(),
        'test_f1_default': float(test_f1_default),
        'test_acc_default': float(test_acc_default),
        'test_cm_default': test_cm_default.tolist(),
        'best_val_f1': float(best_val_f1),
        'best_val_acc': float(best_val_acc),
        'test_f1s_over_time': [float(x) for x in test_f1s],
        'test_accs_over_time': [float(x) for x in test_accs],
        'test_epochs': [int(x) for x in test_epochs],
        'threshold_scores': [(float(t), float(s)) for t, s in threshold_scores]
    }
    
    np.save(os.path.join(args.output_dir, 'test_results.npy'), results)
    print(f'\nTest results saved to {args.output_dir}/test_results.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt-S training for ADNI dataset')
    
    parser.add_argument('--data_path', type=str, default='ADNI/AD_NC',
                        help='Path to ADNI dataset')
    parser.add_argument('--output_dir', type=str, default='ADNI_outputs',
                        help='Path to save outputs')
    parser.add_argument('--pretrained_path', type=str, default='convnext_small_22k_224.pth',
                        help='Path to pretrained weights (ImageNet-22k). Set to empty string to train from scratch.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='AdamW betas')
    parser.add_argument('--warmup_epochs', type=int, default=4,
                        help='Warmup epochs')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--drop_path_rate', type=float, default=0.0,
                        help='Drop path rate')
    parser.add_argument('--layer_scale_init_value', type=float, default=1e-6,
                        help='Layer scale init value')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                        help='Mixup alpha (set to 0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='Cutmix alpha (set to 0 to disable)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Save checkpoint frequency')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)