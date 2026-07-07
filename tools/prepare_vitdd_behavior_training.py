from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"
DOCS_DIR = ROOT / "docs"


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def build_notebook():
    cells = [
        markdown_cell(
            """# ViT-DD-style Driver Behavior Recognition Training

This notebook fine-tunes a Vision Transformer behavior classifier for the MDDAD project.

Expected dataset layout:

```text
dataset_root/
  train/c0/*.jpg
  train/c1/*.jpg
  ...
  valid/c0/*.jpg
  test/c0/*.jpg
```

If the dataset only has `c0..c9` folders at the root, the notebook creates a train/valid split automatically.

Class mapping follows the common distracted-driver format:

`c0 safe driving, c1 texting right, c2 talking phone right, c3 texting left, c4 talking phone left, c5 operating radio, c6 drinking, c7 reaching behind, c8 hair and makeup, c9 talking to passenger`.
"""
        ),
        code_cell(
            """import os
import sys
import zipfile
from pathlib import Path

try:
    import timm
except Exception:
    !{sys.executable} -m pip install -q timm scikit-learn
"""
        ),
        code_cell(
            """import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_COUNT = torch.cuda.device_count()
print('Device:', DEVICE, 'GPU count:', GPU_COUNT)
"""
        ),
        code_cell(
            """# Training config
MODEL_NAME = 'vit_base_patch16_224'
IMAGE_SIZE = 224
EPOCHS = 25
BATCH_SIZE = 32
LR = 3e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 2
PATIENCE = 6
PRETRAINED_BACKBONE = True  # set False if your defense requires training from scratch

CLASS_NAMES = [
    'safe driving',
    'texting right',
    'talking phone right',
    'texting left',
    'talking phone left',
    'operating radio',
    'drinking',
    'reaching behind',
    'hair and makeup',
    'talking to passenger',
]
NUM_CLASSES = len(CLASS_NAMES)
"""
        ),
        code_cell(
            """def find_dataset_root():
    input_root = Path('/kaggle/input')
    candidates = []
    for path in input_root.rglob('*'):
        if not path.is_dir():
            continue
        names = {p.name.lower() for p in path.iterdir() if p.is_dir()}
        if {'train', 'valid'}.issubset(names) or {'train', 'val'}.issubset(names):
            candidates.append(path)
        elif all(f'c{i}' in names for i in range(10)):
            candidates.append(path)
    if not candidates:
        zips = list(input_root.rglob('*.zip'))
        if zips:
            extract_dir = Path('/kaggle/working/extracted_behavior_dataset')
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zips[0], 'r') as zf:
                zf.extractall(extract_dir)
            for path in extract_dir.rglob('*'):
                if path.is_dir():
                    names = {p.name.lower() for p in path.iterdir() if p.is_dir()}
                    if {'train', 'valid'}.issubset(names) or {'train', 'val'}.issubset(names) or all(f'c{i}' in names for i in range(10)):
                        candidates.append(path)
    if not candidates:
        raise FileNotFoundError('Could not find behavior dataset. Add a Kaggle dataset with c0..c9 class folders.')
    return sorted(candidates, key=lambda p: len(str(p)))[0]

DATA_ROOT = find_dataset_root()
print('Dataset root:', DATA_ROOT)
print('Children:', [p.name for p in DATA_ROOT.iterdir() if p.is_dir()][:20])
"""
        ),
        code_cell(
            """train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def split_paths(root):
    names = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    train_dir = names.get('train')
    valid_dir = names.get('valid') or names.get('val')
    test_dir = names.get('test')
    return train_dir, valid_dir, test_dir

train_dir, valid_dir, test_dir = split_paths(DATA_ROOT)

if train_dir and valid_dir:
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    valid_ds = datasets.ImageFolder(valid_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms) if test_dir else None
else:
    full_ds = datasets.ImageFolder(DATA_ROOT, transform=train_tfms)
    valid_size = max(1, int(len(full_ds) * 0.15))
    train_size = len(full_ds) - valid_size
    train_ds, valid_ds = random_split(full_ds, [train_size, valid_size], generator=torch.Generator().manual_seed(SEED))
    valid_ds.dataset.transform = eval_tfms
    test_ds = None

print('Class mapping:', getattr(train_ds, 'class_to_idx', getattr(train_ds.dataset, 'class_to_idx', None)))
print('Train/Valid/Test:', len(train_ds), len(valid_ds), len(test_ds) if test_ds else 0)

pin = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin) if test_ds else None
"""
        ),
        code_cell(
            """model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED_BACKBONE, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
if GPU_COUNT > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
"""
        ),
        code_cell(
            """def run_epoch(loader, training):
    model.train(training)
    losses = []
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        losses.append(float(loss.detach().cpu()))

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'accuracy': float(accuracy_score(all_targets, all_preds)) if all_targets else 0.0,
        'f1_macro': float(f1_score(all_targets, all_preds, average='macro')) if all_targets else 0.0,
        'preds': all_preds,
        'targets': all_targets,
    }

best_f1 = -1.0
best_state = None
history = []
bad_epochs = 0
out_dir = Path('/kaggle/working/vitdd_behavior_export')
out_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_metrics = run_epoch(train_loader, training=True)
    valid_metrics = run_epoch(valid_loader, training=False)
    scheduler.step()

    row = {
        'epoch': epoch,
        'train_loss': train_metrics['loss'],
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['f1_macro'],
        'valid_loss': valid_metrics['loss'],
        'valid_acc': valid_metrics['accuracy'],
        'valid_f1': valid_metrics['f1_macro'],
        'seconds': round(time.time() - start, 2),
    }
    history.append(row)
    print(row)

    if valid_metrics['f1_macro'] > best_f1:
        best_f1 = valid_metrics['f1_macro']
        best_state = copy.deepcopy(model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
        torch.save({
            'state_dict': best_state,
            'model_name': MODEL_NAME,
            'image_size': IMAGE_SIZE,
            'class_names': CLASS_NAMES,
            'valid_f1_macro': best_f1,
            'valid_accuracy': valid_metrics['accuracy'],
        }, out_dir / 'vitdd_behavior.pt')
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print('Early stopping')
            break

with open(out_dir / 'training_history.json', 'w', encoding='utf-8') as f:
    json.dump(history, f, indent=2)
print('Best valid F1:', best_f1)
"""
        ),
        code_cell(
            """checkpoint = torch.load(out_dir / 'vitdd_behavior.pt', map_location=DEVICE)
state = checkpoint['state_dict']
eval_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
eval_model.load_state_dict(state, strict=False)
eval_model.eval()

if test_loader:
    model_to_restore = model
    model = eval_model
    test_metrics = run_epoch(test_loader, training=False)
    model = model_to_restore
    print('Test accuracy:', test_metrics['accuracy'])
    print('Test F1 macro:', test_metrics['f1_macro'])
    print(classification_report(test_metrics['targets'], test_metrics['preds'], target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(test_metrics['targets'], test_metrics['preds'])
    np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
else:
    print('No test split found; skipped test evaluation.')
"""
        ),
        code_cell(
            """readme = f'''# ViT-DD Behavior Model Export

Files:
- vitdd_behavior.pt: copy this to your MDDAD models folder.
- training_history.json: epoch metrics.
- confusion_matrix.csv: only present if a test split existed.

MDDAD config:
VITDD_MODEL_PATH=D:\\\\Work\\\\HocKy3\\\\ComputerVision\\\\MDDAD\\\\models\\\\vitdd_behavior.pt
VITDD_MODEL_NAME={MODEL_NAME}
VITDD_IMAGE_SIZE={IMAGE_SIZE}
VITDD_SAMPLE_FPS=3

Validation best F1 macro: {best_f1}
'''
(out_dir / 'README.md').write_text(readme, encoding='utf-8')

zip_path = Path('/kaggle/working/vitdd_behavior_training_results.zip')
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', out_dir)
print('Export zip:', zip_path)
print('Copy vitdd_behavior.pt into your project models folder.')
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    notebook_path = KAGGLE_DIR / "vitdd_driver_behavior_kaggle_train.ipynb"
    notebook_path.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")

    readme = """# ViT-DD Behavior Training

Notebook: `kaggle/vitdd_driver_behavior_kaggle_train.ipynb`

Use this for the driver-behavior branch of the MDDAD multimodal system.

## Dataset format

Prepare a Kaggle dataset using one of these layouts:

```text
train/c0/*.jpg
train/c1/*.jpg
...
valid/c0/*.jpg
test/c0/*.jpg
```

or:

```text
c0/*.jpg
c1/*.jpg
...
c9/*.jpg
```

The second layout is auto-split into train/valid inside the notebook.

## Output

After training, download:

```text
/kaggle/working/vitdd_behavior_training_results.zip
```

Copy `vitdd_behavior.pt` to:

```text
D:\\Work\\HocKy3\\ComputerVision\\MDDAD\\models\\vitdd_behavior.pt
```

Then open `/behavior-test` in the Flask app and test an image or video.
"""
    (KAGGLE_DIR / "README_KAGGLE_VITDD_BEHAVIOR_TRAIN.md").write_text(readme, encoding="utf-8")

    plan = """# ViT-DD Behavior Recognition Integration

## Purpose

This adds the second branch of the multimodal system:

- Phone branch: object detection, outputs phone bounding boxes and confidence.
- Behavior branch: ViT-DD-style driver behavior recognition, outputs behavior class probabilities.

## App integration

- Route: `/behavior-test`
- Model path: `models/vitdd_behavior.pt`
- Config keys:
  - `VITDD_MODEL_PATH`
  - `VITDD_MODEL_NAME`
  - `VITDD_IMAGE_SIZE`
  - `VITDD_SAMPLE_FPS`
  - `VITDD_LABELS`

## Model

The local adapter uses a Vision Transformer classifier through `timm`, matching the practical ViT-DD integration path. The original ViT-DD paper/source is used as the reference method, while this adapter keeps the Flask app stable and easy to test.

## Not yet connected

The behavior branch is not yet fused into realtime monitoring or uploaded-video Full AI reports. It is isolated first so the model can be trained, validated, and demonstrated safely.
"""
    (DOCS_DIR / "vitdd_behavior_integration.md").write_text(plan, encoding="utf-8")

    print(f"Wrote {notebook_path}")
    print(f"Wrote {KAGGLE_DIR / 'README_KAGGLE_VITDD_BEHAVIOR_TRAIN.md'}")
    print(f"Wrote {DOCS_DIR / 'vitdd_behavior_integration.md'}")


if __name__ == "__main__":
    main()
