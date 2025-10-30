### Alzheimer's Disease Classification with ConvNeXt (ADNI)

Binary AD vs NC MRI slice classification using a lightweight ConvNeXt implementation, subject-wise split, strong augmentations, and optional ImageNet-22k head adaptation.

### Table of Contents

- **Goal**
- **Model Architecture**
  - ConvNeXt blocks and stages
  - Two-stage head (optional ImageNet-22k → adapter)
  - Regularization (LayerScale, DropPath, EMA)
- **Dataset**
  - Expected structure
  - Subject-wise split
  - Transforms and augmentation
- **Training**
- **Results**
- **Usage**
  - Install dependencies
  - Train
  - Predict/evaluate
- **References**

### Goal

Classify brain MR images from the ADNI dataset into two classes: Alzheimer's Disease (AD) and Normal Control (NC). The pipeline targets robust generalization through subject-wise splits and carefully chosen augmentations/regularization.

### Model Architecture

- **Backbone: ConvNeXt (small)** implemented in `modules.py`.
  - Block: 7×7 depthwise conv → LayerNorm → 4× expansion MLP (Linear → GELU → Linear) → residual with optional DropPath and per‑channel LayerScale.
  - Stem: 4×4 Conv2d (stride 4), LayerNorm.
  - 4 stages with downsampling (2×2 Conv2d, stride 2) between stages; default depths `[3, 3, 27, 3]`, dims `[96, 192, 384, 768]`.
  - Global average pooling → LayerNorm.
- **Head(s):**
  - Single‑stage (default): `Linear(dims[-1] → 1)` for binary logits.
  - Two‑stage (optional): `Linear(dims[-1] → 21841)` (pretrained ImageNet‑22k head) → `LayerNorm` → `Linear(21841 → 1)` adapter. Enabled automatically if a compatible pretrained checkpoint is provided at training, or detected when loading a checkpoint at prediction.
- **Regularization & stabilization:**
  - LayerScale (init 1e‑6), optional DropPath.
  - Exponential Moving Average (EMA) of weights (enabled by default in `train.py`).

### Dataset

- **Expected directory layout** (set root with `--data_path`, defaults to `ADNI/AD_NC`):

```
ADNI/AD_NC
├── train
│   ├── AD
│   │   ├── <subjectId_xxx>.jpeg
│   └── NC
│       ├── <subjectId_xxx>.jpeg
└── test
    ├── AD
    └── NC
```

- **Subject-wise split (train/val):**
  - Derived from filenames: subject id is the prefix before the first underscore (see `ADNIDataset._extract_subject_id`).
  - `create_subject_split(..., train_ratio=0.9)` builds a 90/10 split by subject per class to prevent leakage across splits.

- **Transforms (224×224):**
  - Train: `Resize` → `RandomHorizontalFlip` → `RandAugment(num_ops=9, magnitude=9)` → `ToTensor` → `Normalize(ImageNet mean/std)` → `RandomErasing(p=0.25)`.
  - Val/Test: `Resize` → `ToTensor` → `Normalize`.

### Training

Training script: `train.py`.

- Loss: `BCEWithLogitsLoss` on a single logit output.
- Optimizer: AdamW; default `lr=5e-4`, `weight_decay=0.1`, `betas=(0.9, 0.999)`.
- LR schedule: linear warmup (`warmup_epochs=4`) → cosine decay (`WarmupCosineScheduler`).
- MixUp/CutMix: enabled early, linearly decayed, disabled after epoch 150 by default.
- Label smoothing: 0.1 for vanilla batches (no mixing).
- EMA: `timm.utils.ModelEma` (default enabled) with decay `0.9999`.
- Checkpoints: saves best by F1 (`best_model_f1.pth`), best by Accuracy (`best_model_acc.pth`), periodic `checkpoint_epoch_*.pth`.
- Test during training: periodically loads the current best‑acc model, finds an optimal decision threshold on val (F1), evaluates test set, and plots training curves.

Key defaults (see `argparse` in `train.py`):

- `--epochs 200`  `--batch_size 128`  `--input_size 224`  `--num_workers 8`  `--seed 42`
- `--drop_path_rate 0.0`  `--layer_scale_init_value 1e-6`
- `--use_ema` (default True)  `--ema_decay 0.9999`
- Optional pretrained ImageNet‑22k head: provide `--pretrained_path convnext_small_22k_224.pth` (two‑stage head + adapter will be used automatically).

### Results

From `predict.py` on the provided checkpoint and test split (`results_predict.txt`):

- Threshold: 0.01
- F1: 0.7755
- Accuracy: 0.8069
- Confusion matrix (rows: true NC/AD; cols: pred NC/AD):
  - [[4260, 280], [1458, 3002]]
- Sensitivity (AD recall): 0.6731
- Specificity (NC): 0.9383
- Precision (AD): 0.9147

The prediction script also saves: confusion matrix, ROC curve (with AUC), probability distribution, and example predictions.

### Usage

#### Install dependencies

This project uses PyTorch, torchvision, timm, and common ML/plotting libs.

```bash
pip install torch torchvision timm scikit-learn seaborn matplotlib tqdm pillow
```

#### Train

```bash
cd recognition/ConvNeXt_ADNI_s4909455
python train.py \
  --data_path ADNI/AD_NC \
  --output_dir ADNI_outputs \
  --pretrained_path convnext_small_22k_224.pth   # optional; leave empty to train from scratch
```

Common flags:

- `--epochs 200` `--batch_size 128` `--lr 5e-4` `--num_workers 8`
- `--drop_path_rate 0.0` `--layer_scale_init_value 1e-6`
- `--save_freq 20`

Artifacts are written to `--output_dir` (checkpoints, plots, `test_results.npy`).

#### Predict / Evaluate

```bash
python predict.py \
  --checkpoint ADNI_outputs/best_model_acc.pth \
  --data_path ADNI/AD_NC \
  --split test \
  --output_dir ADNI_outputs/predictions \
  --threshold 0.01 \
  --batch_size 32
```

Outputs (per split) are saved under `--output_dir/<split>/`:

- `confusion_matrix.png`, `roc_curve.png`, `probability_distribution.png`, `prediction_examples.png`.

### References

- Liu, Z., Mao, H., Wu, C.‑Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s (ConvNeXt). CVPR. `https://arxiv.org/abs/2201.03545`

