import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm

from modules import convnext_small
from dataset import ADNIDataset, build_transform


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine which state dict to use
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print('Loaded EMA model weights')
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print('Loaded model weights')
    else:
        state_dict = checkpoint
        print('Loaded model weights (legacy format)')
    
    # Detect if model uses two-stage head (has head_norm)
    use_pretrained_head = any('head_norm' in k for k in state_dict.keys())
    
    if use_pretrained_head:
        print('Detected two-stage head architecture (with head_norm)')
    else:
        print('Detected single-stage head architecture')
    
    model = convnext_small(num_classes=1, use_pretrained_head=use_pretrained_head)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model


def predict_dataset(model, data_loader, device, threshold=0.5):
    """Make predictions on a dataset
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        threshold: Classification threshold (default: 0.5)
    """
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'Predicting (threshold={threshold:.3f})')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if len(all_images) < 20:
                all_images.extend(images.cpu())
            
            current_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images[:20]


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'ROC curve saved to {save_path}')


def plot_prediction_examples(images, labels, preds, probs, save_path):
    """Plot example predictions"""
    class_names = ['NC (Normal)', 'AD (Alzheimer\'s)']
    n_samples = min(20, len(images))
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for i in range(n_samples):
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        true_label = int(labels[i])
        pred_label = int(preds[i])
        confidence = probs[i] if pred_label == 1 else (1 - probs[i])
        
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {class_names[true_label]}\n'
        title += f'Pred: {class_names[pred_label]}\n'
        title += f'Conf: {confidence:.3f}'
        
        axes[i].set_title(title, fontsize=8, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Prediction examples saved to {save_path}')


def plot_probability_distribution(labels, probs, save_path):
    """Plot probability distribution for each class"""
    nc_probs = probs[labels == 0]
    ad_probs = probs[labels == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(nc_probs, bins=30, alpha=0.5, label='NC (Normal)', color='blue', density=True)
    plt.hist(ad_probs, bins=30, alpha=0.5, label='AD (Alzheimer\'s)', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Predicted Probability (AD)')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Probability distribution saved to {save_path}')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)
    
    transform = build_transform(is_train=False, input_size=args.input_size)
    
    eval_dataset = ADNIDataset(args.data_path, split=args.split, transform=transform)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Evaluating on: {args.split.upper()} dataset')
    print(f'Dataset size: {len(eval_dataset)}')
    print(f'Classification threshold: {args.threshold}\n')
    
    print('Making predictions...')
    preds, labels, probs, sample_images = predict_dataset(model, eval_loader, device, threshold=args.threshold)
    
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    
    print('\n' + '='*60)
    print(f'{args.split.upper()} RESULTS (Threshold={args.threshold:.3f})')
    print('='*60)
    print(f'F1 Score:  {f1:.4f}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'                  Predicted')
    print(f'                  NC    AD')
    print(f'True    NC     {cm[0, 0]:5d} {cm[0, 1]:5d}')
    print(f'        AD     {cm[1, 0]:5d} {cm[1, 1]:5d}')
    print(f'\nTrue Negatives:  {cm[0, 0]}')
    print(f'False Positives: {cm[0, 1]}')
    print(f'False Negatives: {cm[1, 0]}')
    print(f'True Positives:  {cm[1, 1]}')
    
    if cm[1, 1] + cm[1, 0] > 0:
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        print(f'\nSensitivity (Recall): {sensitivity:.4f}')
    
    if cm[0, 0] + cm[0, 1] > 0:
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        print(f'Specificity:          {specificity:.4f}')
    
    if cm[1, 1] + cm[0, 1] > 0:
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        print(f'Precision:            {precision:.4f}')
    
    print('='*60 + '\n')
    
    # Create split-specific output directory
    output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_roc_curve(labels, probs, os.path.join(output_dir, 'roc_curve.png'))
    plot_probability_distribution(labels, probs, os.path.join(output_dir, 'probability_distribution.png'))
    
    if len(sample_images) > 0:
        sample_labels = labels[:len(sample_images)]
        sample_preds = preds[:len(sample_images)]
        sample_probs = probs[:len(sample_images)]
        plot_prediction_examples(sample_images, sample_labels, sample_preds, sample_probs,
                                os.path.join(output_dir, 'prediction_examples.png'))
    
    print(f'\nAll visualizations saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt-S prediction for ADNI dataset')
    
    parser.add_argument('--checkpoint', type=str, default='ADNI_outputs/best_model_acc.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='ADNI/AD_NC',
                        help='Path to ADNI dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which dataset split to evaluate (default: test)')
    parser.add_argument('--output_dir', type=str, default='ADNI_outputs/predictions',
                        help='Path to save prediction outputs')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)