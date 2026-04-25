"""
train.py — Yieldy EfficientNet-B0 Training Script
===================================================
Dataset layout expected (two separate source folders merged at load time):

    Talong_DataSets/
        Original Dataset/           ← real photos, all 11 classes
            Eggplant Fruit Creaking/
            Eggplant Healthy Fruit/
            Eggplant Healthy Leaf/
            Eggplant Insect Pest Disease/
            Eggplant Leaf Spot Disease/
            Eggplant Mosaic Virus Disease/
            Eggplant Phomopsis Blight/
            Eggplant Shoot and Fruit Borer/
            Eggplant Small Leaf Disease/
            Eggplant Wet Rot/
            Eggplant Wilt Disease/

        Augmented Dataset/          ← augmented photos, 6 leaf classes only
            Eggplant Healthy Leaf/
            Eggplant Insect Pest Disease/
            Eggplant Leaf Spot Disease/
            Eggplant Mosaic Virus Disease/
            Eggplant Small Leaf Disease/
            Eggplant Wilt Disease/

Both folders are loaded and merged into one combined dataset before splitting.
The class label is taken directly from each subfolder name.

Usage:
    python train.py

Output:
    yieldy_model.pth             — best checkpoint (place next to app.py)
    yieldy_model_class_map.json  — index <-> class name mapping for verification
"""

import os
import copy
import time
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Both source folders are merged at load time into one combined dataset.
# Add or remove paths here if your folder structure ever changes.
DATASET_DIRS = [
    r"C:\Users\Asus\Documents\Talong_DataSets\Original Dataset",
    r"C:\Users\Asus\Documents\Talong_DataSets\Augmented Dataset",
]

OUTPUT_PATH = "yieldy_model.pth"
NUM_CLASSES = 11
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-4    # classifier head learning rate
LR_BACKBONE = 1e-5    # pretrained backbone learning rate
VAL_SPLIT   = 0.15
SEED        = 42
NUM_WORKERS = 0       # keep 0 on Windows to avoid multiprocessing pickle errors

# Class names must match the subfolder names in your dataset EXACTLY.
# These also match the DISEASE_INFO keys in app.py.
CLASS_NAMES = [
    # ── Leaf classes ───────────────────────────────────────────────────────────
    "Eggplant Healthy Leaf",
    "Eggplant Insect Pest Disease",
    "Eggplant Leaf Spot Disease",
    "Eggplant Mosaic Virus Disease",
    "Eggplant Small Leaf Disease",
    "Eggplant Wilt Disease",
    # ── Fruit classes ──────────────────────────────────────────────────────────
    "Eggplant Healthy Fruit",
    "Eggplant Fruit Creaking",
    "Eggplant Phomopsis Blight",
    "Eggplant Shoot and Fruit Borer",
    "Eggplant Wet Rot",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM FLAT-FOLDER DATASET
# ══════════════════════════════════════════════════════════════════════════════

class EggplantDataset(Dataset):
    """
    Loads images from one or more subfolder-per-class directories and merges
    all samples into a single dataset. Accepts either a single path string or
    a list of path strings.

    Each directory is scanned independently:
        SomeFolder/
            Eggplant Healthy Leaf/
                image1.jpg  ...
            Eggplant Mosaic Virus Disease/
                image1.jpg  ...

    The class label is taken directly from the subfolder name.
    Falls back to flat-file parsing (_NNN suffix stripping) if no subfolders
    matching a known class are found inside a directory.
    """

    def __init__(self, directories, class_to_idx: dict, transform=None):
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self.samples      = []   # list of (filepath, class_index)
        self.skipped      = []   # files whose label didn't match any class

        # Accept either a single path string or a list of paths
        if isinstance(directories, (str, Path)):
            directories = [directories]

        for directory in directories:
            root = Path(directory)
            dir_samples = []

            # ── Subfolder layout ──────────────────────────────────────────────
            for subfolder in sorted(root.iterdir()):
                if not subfolder.is_dir():
                    continue
                label = subfolder.name.strip()
                if label not in class_to_idx:
                    continue
                class_idx = class_to_idx[label]
                for fpath in sorted(subfolder.iterdir()):
                    if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    dir_samples.append((str(fpath), class_idx))

            # ── Flat-file fallback ────────────────────────────────────────────
            if not dir_samples:
                for fpath in sorted(root.iterdir()):
                    if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    label = self._parse_label(fpath.stem)
                    if label is None or label not in class_to_idx:
                        self.skipped.append(fpath.name)
                        continue
                    dir_samples.append((str(fpath), class_to_idx[label]))

            self.samples.extend(dir_samples)
            print(f"   📁 {root.name:<25} → {len(dir_samples):>5} images loaded")

    @staticmethod
    def _parse_label(stem: str):
        """Strip trailing '_NNN' from filename stem."""
        idx = stem.rfind("_")
        return stem[:idx].strip() if idx != -1 else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        fpath, label = self.samples[index]
        image = Image.open(fpath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def make_loaders(dataset_dirs):
    # Build two Dataset objects over the same files — different transforms
    print("\n📂 Scanning source folders:")
    train_dataset = EggplantDataset(dataset_dirs, CLASS_TO_IDX, transform=train_transforms)
    print()
    val_dataset   = EggplantDataset(dataset_dirs, CLASS_TO_IDX, transform=val_transforms)

    # Report skipped files
    if train_dataset.skipped:
        print(f"\n⚠️  Skipped {len(train_dataset.skipped)} unrecognised file(s):")
        for name in train_dataset.skipped[:10]:
            print(f"     {name}")
        if len(train_dataset.skipped) > 10:
            print(f"     … and {len(train_dataset.skipped) - 10} more.")

    n_total = len(train_dataset)
    if n_total == 0:
        raise RuntimeError(
            "No images were loaded.\n"
            "  1. Confirm all paths in DATASET_DIRS point to folders containing class subfolders.\n"
            "  2. Confirm subfolder names match CLASS_NAMES exactly.\n"
            "  3. Confirm image files are .jpg / .jpeg / .png / .bmp / .webp"
        )

    # Per-class counts (full dataset)
    counts = Counter(label for _, label in train_dataset.samples)
    print("\n📂 Images found per class:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"   [{idx}] {name:<42} {counts.get(idx, 0):>5} images")

    # Split by index (same indices applied to both dataset objects)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(SEED)
    shuffled  = torch.randperm(n_total, generator=generator).tolist()
    train_idx, val_idx = shuffled[:n_train], shuffled[n_train:]

    # ── Option 1: Weighted Sampler ────────────────────────────────────────────
    # Compute per-class counts only within the training split, then assign each
    # sample a weight = 1 / class_count so every class is sampled equally.
    train_labels  = [train_dataset.samples[i][1] for i in train_idx]
    train_counts  = Counter(train_labels)
    sample_weights = [
        1.0 / train_counts[train_dataset.samples[i][1]]
        for i in train_idx
    ]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_idx),
        replacement = True,   # required for oversampling minority classes
    )
    print(f"\  Weighted sampler active — minority classes will be upsampled.")

    # ── Option 2: Class weights for weighted loss (computed here, used in main)
    # Weight for each class = total_train / (num_classes * class_count)
    # Minority classes get higher weights, majority classes get lower weights.
    class_weights = torch.zeros(NUM_CLASSES)
    for class_idx, cnt in train_counts.items():
        class_weights[class_idx] = n_train / (NUM_CLASSES * cnt)
    print(f"  Loss weights per class:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"     [{idx}] {name:<42} weight={class_weights[idx]:.4f}")
    # ─────────────────────────────────────────────────────────────────────────

    # NOTE: sampler and shuffle are mutually exclusive — sampler handles ordering
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"\n   Train: {n_train}  |  Val: {n_val}  |  Total: {n_total}\n")
    return train_loader, val_loader, class_weights


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device: torch.device):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * images.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        total_loss    += criterion(out, labels).item() * images.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    if device.type == "cpu":
        print("   (No GPU detected — training on CPU will be slower)\n")

    for path in DATASET_DIRS:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Dataset directory not found:\n  {path}\n"
                "Check that the path is correct and the folder exists."
            )

    train_loader, val_loader, class_weights = make_loaders(DATASET_DIRS)

    model     = build_model(device)

    # ── Option 2: Weighted Loss ───────────────────────────────────────────────
    # Move class_weights to the same device as the model so PyTorch doesn't
    # raise a device mismatch error during the backward pass.
    criterion = nn.CrossEntropyLoss(
        weight        = class_weights.to(device),
        label_smoothing = 0.1,
    )
    # ─────────────────────────────────────────────────────────────────────────

    backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier"     in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params,   "lr": LR_BACKBONE},
        {"params": classifier_params, "lr": LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    best_state   = None

    print("=" * 68)
    print(f"  EfficientNet-B0  ·  {NUM_CLASSES} classes  ·  {NUM_EPOCHS} epochs")
    print("=" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            flag = "  <- best ✅"

        print(
            f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc*100:.1f}% | "
            f"val loss {val_loss:.4f}  acc {val_acc*100:.1f}% | "
            f"{time.time()-t0:.0f}s{flag}"
        )

    # Save best weights
    torch.save(best_state, OUTPUT_PATH)
    print(f"\n✅ Best model saved  -> {OUTPUT_PATH}")
    print(f"   Best val accuracy : {best_val_acc * 100:.2f}%")

    # Save class index map
    map_path = OUTPUT_PATH.replace(".pth", "_class_map.json")
    with open(map_path, "w") as f:
        json.dump({
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": {str(v): k for k, v in CLASS_TO_IDX.items()},
        }, f, indent=2)
    print(f"   Class map saved   -> {map_path}")
    print("\n▶  Place yieldy_model.pth next to app.py, then run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()