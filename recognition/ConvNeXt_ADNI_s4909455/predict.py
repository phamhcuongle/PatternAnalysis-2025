from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

from modules import convnext_b


def load_model(checkpoint_path: str, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None

    ck = torch.load(checkpoint_path, map_location=device)
    model = convnext_b(num_classes=2, pretrained_path=None, device=device)
    model.load_state_dict(ck['model_state'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and set to eval mode.")
    return model, device


def evaluate(model, device, test_root: str, save_plot_path: str, img_size=224, batch_size=64):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dir = Path(test_root) / 'AD_NC' / 'test'
    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return

    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Found {len(dataset)} images in test set. Starting evaluation...")

    ys = []
    yps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            yps.extend(preds.tolist())
            ys.extend(y.numpy().tolist())

    ys = np.array(ys)
    yps = np.array(yps)
    acc = (ys == yps).mean()
    print(f'Test accuracy: {acc:.4f}  N={len(ys)}')

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(ys, yps, target_names=dataset.classes))
        print('confusion_matrix:')
        print(confusion_matrix(ys, yps))
    except Exception as e:
        print(f"Could not print classification report: {e}")
        pass

    print(f"Saving prediction samples to '{save_plot_path}'...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    shown = 0
    for i in range(min(50, len(dataset))):
        img, label = dataset[i]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            p = F.softmax(out, dim=1)[0].cpu().numpy()
            pred = p.argmax()
        ax = axes.flat[shown]
        # unnormalize for display
        img_np = img.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f'{dataset.classes[pred]} / {dataset.classes[label]}')
        ax.axis('off')
        shown += 1
        if shown >= 10:
            break
    plt.tight_layout()

    plt.savefig(save_plot_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./ADNI')
    parser.add_argument('--checkpoint', type=str, default='./outputs/best.pth')
    parser.add_argument('--save-plot', type=str, default='./pred_samples.png')
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)
    if model:
        evaluate(model, device, args.data_root, args.save_plot)