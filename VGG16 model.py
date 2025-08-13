# vgg_baseline.py
import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transforms(img_size=224, aug=True):
    if aug:
        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


class RGBArrayDataset(Dataset):
    """Dataset for DataFrames with an 'rgb_array' column (HxWx3 uint8)."""
    def __init__(self, df: pd.DataFrame, label_col: str, transform):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        self.images = self.df["rgb_array"].tolist()
        self.labels = torch.tensor(self.df[self.label_col].values, dtype=torch.long)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        arr = self.images[idx]
        img_t = self.transform(arr)
        y = self.labels[idx]
        return img_t, y


class PathDataset(Dataset):
    """Dataset for DataFrames with an 'image_path' column."""
    def __init__(self, df: pd.DataFrame, label_col: str, transform):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.transform = transform
        self.paths = self.df["image_path"].tolist()
        self.labels = torch.tensor(self.df[self.label_col].values, dtype=torch.long)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        img_t = self.transform(arr)
        y = self.labels[idx]
        return img_t, y



# Model: VGG16 baseline (vision-only)
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes: int, unfreeze_from_idx: int = 17):
        """
        unfreeze_from_idx: conv layer index in features to start fine-tuning
        (e.g., 17 ≈ conv4_*, 23+ ≈ conv5_*). Lower = more layers trainable.
        """
        super().__init__()
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = m.features  # conv stack

        # Freeze all, then unfreeze last blocks
        for p in self.features.parameters():
            p.requires_grad = False
        for name, p in self.features.named_parameters():
            # 'name' looks like '0.weight', '10.bias', etc.
            layer_idx = name.split(".")[0]
            if layer_idx.isdigit() and int(layer_idx) >= unfreeze_from_idx:
                p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)  # Bx512
        return self.fc(x)



# Train / Eval
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for imgs, ys in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(imgs)
        loss = criterion(logits, ys)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * ys.size(0)
        total_correct += (logits.argmax(1) == ys).sum().item()
        total_count += ys.size(0)

    return total_loss / total_count, total_correct / total_count


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    # -------- Load DataFrame --------
    data_path = Path(args.data)
    if data_path.suffix.lower() in [".pkl", ".pickle"]:
        df = pd.read_pickle(data_path)
    elif data_path.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(data_path)
    else:
        # assume CSV by default
        df = pd.read_csv(data_path)

    # Normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # -------- Filter labels to binary GT --------
    if "GT" not in df.columns:
        raise ValueError("DataFrame must contain a 'GT' column with 'True'/'False' labels.")
    df = df[df["GT"].astype(str).isin(["True", "False"])].copy()

    # Ensure we have either rgb_array or image_path
    has_rgb = "rgb_array" in df.columns
    has_path = "image_path" in df.columns
    if not has_rgb and not has_path:
        raise ValueError("DataFrame must contain 'rgb_array' or 'image_path' column.")

    # Drop NAs for critical fields
    need_cols = ["GT"]
    if has_rgb:  need_cols.append("rgb_array")
    if has_path: need_cols.append("image_path")
    df = df.dropna(subset=need_cols).reset_index(drop=True)

    # Label encode GT
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["GT"].astype(str))
    num_classes = df["label_id"].nunique()
    if num_classes < 2:
        raise ValueError("Need at least two classes after filtering 'GT'.")

    # Train/val split (stratified)
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, stratify=df["label_id"], random_state=args.seed
    )

    # Transforms
    train_tf, val_tf = make_transforms(img_size=args.img_size, aug=not args.no_aug)

    # Datasets & Loaders
    if has_rgb:
        train_ds = RGBArrayDataset(train_df, "label_id", train_tf)
        val_ds   = RGBArrayDataset(val_df,   "label_id", val_tf)
    else:
        # verify files exist
        missing = [p for p in val_df["image_path"].tolist() if not os.path.exists(p)]
        if missing:
            print(f"Warning: {len(missing)} missing files in val set (showing first 3): {missing[:3]}")
        train_ds = PathDataset(train_df, "label_id", train_tf)
        val_ds   = PathDataset(val_df,   "label_id", val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch, 64), shuffle=False, num_workers=2, pin_memory=True)

    # Class weights for imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = VGG16Classifier(num_classes=num_classes, unfreeze_from_idx=args.unfreeze_from).to(device)

    # Optimizer & Scheduler
    # Slightly lower LR for fine-tuning convs; classifier head will adapt quickly.
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params     = [p for p in model.fc.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train
    best_val = 0.0
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = str(Path(args.outdir) / "vgg_baseline_best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.3f} | "
              f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model": model.state_dict(),
                "label_encoder": le,
                "args": vars(args)
            }, ckpt_path)

    print("Best Val Acc:", best_val)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG16 vision-only baseline trainer")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to DataFrame file (csv/parquet/pickle) with 'GT' and 'rgb_array' or 'image_path'")
    parser.add_argument("--outdir", type=str, default="./checkpoints", help="Output dir for checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-aug", action="store_true", help="Disable training augmentations")

    parser.add_argument("--unfreeze-from", type=int, default=17,
                        help="VGG16 features layer index to start fine-tuning (e.g., 17≈conv4_*, 23≈conv5_*)")

    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="LR for unfrozen VGG conv layers")
    parser.add_argument("--lr-head",     type=float, default=1e-4, help="LR for classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma",     type=float, default=0.5)

    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()
    main(args)
