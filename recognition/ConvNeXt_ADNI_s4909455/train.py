import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import convnext_b
from dataset import build_dataloaders


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


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    pbar = tqdm(loader, desc='train', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(out, labels) * bs
        n += bs
        pbar.set_postfix(loss=running_loss / n, acc=running_acc / n)
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
            loss = loss_fn(out, labels)
            bs = images.size(0)
            running_loss += loss.item() * bs
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

    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = convnext_b(num_classes=2, in_chans=3, pretrained_path=args.pretrained, device=device).to(device)

    param_groups = get_layerwise_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    stats = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting training for {args.epochs} epochs... Saving outputs to {out_dir}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        stats['train_acc'].append(train_acc)
        stats['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={time.time()-start:.1f}s")

        if val_acc > best_val_acc:
            print(f"  New best val_acc: {val_acc:.4f} (was {best_val_acc:.4f}). Saving model...")
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, out_dir / 'best.pth')

        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, out_dir / f'epoch_{epoch}.pth')

    plot_stats(stats, out_dir)
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict()}, out_dir / 'final.pth')
    print(f"Training complete. Final model saved to {out_dir / 'final.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./ADNI')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=2e-8)
    parser.add_argument('--layer-decay', type=float, default=0.8)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    args = parser.parse_args()
    main(args)
