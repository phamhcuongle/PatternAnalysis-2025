import argparse
import math
import os
from pathlib import Path
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import convnext_b
from dataset import build_dataloaders
from predict import evaluate, load_model


def get_layerwise_param_groups(model, base_lr, weight_decay, layer_decay=0.8):
    param_groups = {}
    num_stages = 4

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_lr = base_lr
        if 'stages.' in name:
            try:
                idx = int(name.split('stages.')[1].split('.')[0])
                scale = layer_decay ** (num_stages - 1 - idx)
                group_lr = base_lr * scale
            except Exception:
                group_lr = base_lr
        elif 'downsample_layers.' in name:
            try:
                idx = int(name.split('downsample_layers.')[1].split('.')[0])
                scale = layer_decay ** (num_stages - 1 - idx)
                group_lr = base_lr * scale
            except Exception:
                group_lr = base_lr

        key = (group_lr, weight_decay)
        if key not in param_groups:
            param_groups[key] = {'params': [], 'lr': group_lr, 'weight_decay': weight_decay}
        param_groups[key]['params'].append(param)

    return list(param_groups.values())


def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()

def _rand_bbox(H, W, lam):
    cut_rat = math.sqrt(max(0.0, 1. - lam))
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)


def apply_mixup_cutmix_same_class(images, labels, alpha=0.4, device='cpu', use_cutmix=False):
    B, C, H, W = images.shape
    labels = labels.to(device)

    pair_idx = torch.empty(B, dtype=torch.long, device=device)
    for i in range(B):
        same = (labels == labels[i]).nonzero(as_tuple=False).squeeze()
        if same.numel() == 0:
            pair_idx[i] = i
        else:
            j = same[torch.randint(0, same.numel(), (1,)).item()]
            pair_idx[i] = j

    images2 = images[pair_idx]
    labels_a = labels
    labels_b = labels[pair_idx]

    lam = np.random.beta(alpha, alpha, size=B).astype(np.float32)
    lams = torch.from_numpy(lam).to(device)

    mixed_images = images.clone()
    if use_cutmix:
        for i in range(B):
            lam_i = float(lams[i].item())
            bbx1, bby1, bbx2, bby2 = _rand_bbox(H, W, lam_i)
            mixed_images[i, :, bby1:bby2, bbx1:bbx2] = images2[i, :, bby1:bby2, bbx1:bbx2]
            area = (bbx2 - bbx1) * (bby2 - bby1)
            lams[i] = 1.0 - float(area) / float(H * W)
    else:
        lams_img = lams.view(B, 1, 1, 1)
        mixed_images = images * lams_img + images2 * (1.0 - lams_img)

    return mixed_images, labels_a, labels_b, lams


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, max_mix_prob=0.5, mix_decay_epochs=50, mix_alpha=0.4):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    pbar = tqdm(loader, desc='train', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # schedule mix probability: linear decay from max_mix_prob -> 0 over mix_decay_epochs
        if epoch <= mix_decay_epochs:
            mix_prob = max_mix_prob * (1.0 - (epoch - 1) / float(max(1, mix_decay_epochs)))
        else:
            mix_prob = 0.0

        do_mix = random.random() < mix_prob
        if do_mix:
            use_cutmix = random.random() < 0.5
            mixed_images, labels_a, labels_b, lams = apply_mixup_cutmix_same_class(
                images, labels, alpha=mix_alpha, device=device, use_cutmix=use_cutmix)
            out = model(mixed_images)
            loss_a = loss_fn(out, labels_a)
            loss_b = loss_fn(out, labels_b)
            if loss_a.ndim == 0:
                loss_a = loss_a.unsqueeze(0)
                loss_b = loss_b.unsqueeze(0)
            loss = (lams * loss_a + (1.0 - lams) * loss_b).mean()
        else:
            out = model(images)
            l = loss_fn(out, labels)
            loss = l.mean() if l.ndim > 0 else l

        loss.backward()
        optimizer.step()
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(out, labels) * bs
        n += bs
        pbar.set_postfix(loss=running_loss / n, acc=running_acc / n, mix_prob=f"{mix_prob:.3f}")
    return running_loss / n, running_acc / n


def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='val', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss_t = loss_fn(out, labels)
            loss_val = loss_t.mean().item() if isinstance(loss_t, torch.Tensor) else float(loss_t)
            bs = images.size(0)
            running_loss += loss_val * bs
            running_acc += accuracy(out, labels) * bs
            n += bs
    return running_loss / n, running_acc / n


def plot_stats(stats, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(stats['train_loss']) + 1))
    plt.figure()
    plt.plot(epochs, stats['train_loss'], label='train_loss')
    plt.plot(epochs, stats['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(out_dir / 'loss.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, stats['train_acc'], label='train_acc')
    plt.plot(epochs, stats['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(out_dir / 'acc.png')
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = build_dataloaders(args.data_root, batch_size=args.batch_size,
                                                image_size=args.img_size, val_ratio=0.15,
                                                num_workers=args.num_workers)

    model = convnext_b(num_classes=2, in_chans=3, pretrained_path=args.pretrained, device=device).to(device)

    param_groups = get_layerwise_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)

    stats = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_acc = 0.0
    epochs_since_best = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                                epoch, max_mix_prob=args.mix_max_prob,
                                                mix_decay_epochs=args.mix_decay_epochs,
                                                mix_alpha=args.mix_alpha)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        stats['train_acc'].append(train_acc)
        stats['val_acc'].append(val_acc)

        # get learning rates for all param groups and print epoch summary
        lrs = [pg.get('lr', None) for pg in optimizer.param_groups]
        lr_str = ','.join([f"{lr:.3e}" for lr in lrs])
        print(f"Epoch {epoch}/{args.epochs}  lr={lr_str}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={time.time()-start:.1f}s")

        # save best
        if val_acc > best_val_acc:
            print(f"  ** New best validation accuracy: {val_acc:.4f}! Saving model to {out_dir / 'best.pth'} **")
            best_val_acc = val_acc
            epochs_since_best = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, out_dir / 'best.pth')
        else:
            epochs_since_best += 1

        # early stopping check
        if args.early_stop > 0 and epochs_since_best >= args.early_stop:
            print(f"Early stopping: no improvement for {epochs_since_best} epochs (early_stop={args.early_stop})")
            break

        # periodic save
        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, out_dir / f'epoch_{epoch}.pth')

        # run evaluation on test set using best weights every eval_every epochs
        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            best_ckpt = out_dir / 'best.pth'
            if best_ckpt.exists():
                print(f"Evaluating best checkpoint at epoch {epoch} on test set...")
                try:
                    test_model, test_device = load_model(str(best_ckpt), device=device)
                    # Pass output_dir so plot is saved to Drive
                    evaluate(test_model, test_device, args.data_root, args.output_dir,
                             img_size=args.img_size, batch_size=args.batch_size)
                except Exception as e:
                    print(f"Test evaluation failed: {e}")

    plot_stats(stats, out_dir)
    # final save
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict()}, out_dir / 'final.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='ADNI')
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=2e-8)
    parser.add_argument('--layer-decay', type=float, default=0.8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mix-alpha', type=float, default=0.4, help='alpha parameter for Beta distribution used in mixup/cutmix')
    parser.add_argument('--mix-max-prob', type=float, default=0, help='starting probability of applying mixup/cutmix (linearly decays)')
    parser.add_argument('--mix-decay-epochs', type=int, default=100, help='number of epochs over which to decay mix probability to 0')
    parser.add_argument('--eval-every', type=int, default=10, help='run test evaluation with best.pth every N epochs (0 to disable)')
    parser.add_argument('--early-stop', type=int, default=75, help='stop training if no val acc improvement for this many epochs (0 to disable)')
    args = parser.parse_args()
    main(args)