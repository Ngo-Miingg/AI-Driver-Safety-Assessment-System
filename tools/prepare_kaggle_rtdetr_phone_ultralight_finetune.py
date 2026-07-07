from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "dataset" / "roboflow_mobilephone_wusj2_v7" / "extracted"
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
KAGGLE_DATASET_DIR = PROJECT_ROOT / "dataset" / "kaggle_upload"
KAGGLE_DATASET_ZIP = KAGGLE_DATASET_DIR / "mobilephone_wusj2_v7_rtdetr_ultralight_finetune_kaggle.zip"
NOTEBOOK_PATH = KAGGLE_DIR / "phone_detection_rtdetr_ultralight_finetune_kaggle.ipynb"
README_PATH = KAGGLE_DIR / "README_KAGGLE_RTDETR_PHONE_ULTRALIGHT_FINETUNE.md"

RANDOM_SEED = 42
TRAIN_IMAGE_LIMIT = 1500
VALID_IMAGE_LIMIT = 326
TEST_IMAGE_LIMIT = 185


FIXED_DATA_YAML = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Mobile-phone']

roboflow:
  workspace: m17865515473-163-com
  project: mobilephone-wusj2
  version: 7
  license: CC BY 4.0
  url: https://universe.roboflow.com/m17865515473-163-com/mobilephone-wusj2/dataset/7
  note: ultralight RT-DETR fine-tuning subset
"""


def main() -> int:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DATASET}")

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    write_dataset_zip(KAGGLE_DATASET_ZIP)
    write_notebook(NOTEBOOK_PATH)
    write_readme(README_PATH)

    print(f"Ultralight fine-tune notebook: {NOTEBOOK_PATH}")
    print(f"Ultralight dataset zip: {KAGGLE_DATASET_ZIP}")
    print(f"README: {README_PATH}")
    return 0


def write_dataset_zip(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", FIXED_DATA_YAML)
        for split, limit in (
            ("train", TRAIN_IMAGE_LIMIT),
            ("valid", VALID_IMAGE_LIMIT),
            ("test", TEST_IMAGE_LIMIT),
        ):
            image_dir = SOURCE_DATASET / split / "images"
            label_dir = SOURCE_DATASET / split / "labels"
            for image_path in select_images(image_dir, label_dir, limit):
                label_path = label_dir / f"{image_path.stem}.txt"
                zf.write(image_path, f"{split}/images/{image_path.name}")
                zf.write(label_path, f"{split}/labels/{label_path.name}")

        for readme_name in ("README.dataset.txt", "README.roboflow.txt"):
            readme_path = SOURCE_DATASET / readme_name
            if readme_path.exists():
                zf.write(readme_path, readme_name)


def select_images(image_dir: Path, label_dir: Path, limit: int) -> list[Path]:
    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        and (label_dir / f"{p.stem}.txt").exists()
    )
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)
    return sorted(images[: min(limit, len(images))])


def write_notebook(output_path: Path) -> None:
    cells = [
        md(
            "# Ultralight RT-DETR Phone Fine-Tuning\n\n"
            "This notebook is lighter than the practical-final notebook.\n\n"
            "Use it when RT-DETR-L fine-tuning is still too slow, but you still want an RT-DETR result for comparison/demo.\n\n"
            "Training profile:\n"
            "- Model: `rtdetr-l.pt`\n"
            "- Train subset: 1500 images\n"
            "- Validation/Test: full splits\n"
            "- Image size: 416\n"
            "- Epochs: 5\n"
            "- Freeze layers: 20\n"
            "- Batch retry: 24, 16, 12, 8, 4\n\n"
            "Output model:\n"
            "- `phone_rtdetr_l_ultralight_best.pt`\n"
        ),
        code(
            "import os\n"
            "import sys\n"
            "import shutil\n"
            "import subprocess\n"
            "import zipfile\n"
            "from pathlib import Path\n\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n\n"
            "def ensure_package(package_name, import_name=None):\n"
            "    import importlib.util\n"
            "    import_name = import_name or package_name\n"
            "    if importlib.util.find_spec(import_name) is None:\n"
            "        print(f'Installing {package_name} without dependency upgrades...')\n"
            "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', package_name])\n"
            "    else:\n"
            "        print(f'{package_name} is already available')\n\n"
            "ensure_package('ultralytics')\n\n"
            "import torch\n"
            "import yaml\n"
            "from ultralytics import RTDETR\n\n"
            "print('Python:', sys.version)\n"
            "print('Torch:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('GPU count:', torch.cuda.device_count())\n"
            "for i in range(torch.cuda.device_count()):\n"
            "    print(f'GPU {i}:', torch.cuda.get_device_name(i))\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('CUDA is not available. Enable GPU in Kaggle Settings, restart session, then run again.')\n"
        ),
        md("## 1. Locate Dataset"),
        code(
            "INPUT_ROOT = Path('/kaggle/input')\n"
            "WORKING_DIR = Path('/kaggle/working')\n"
            "RUNS_DIR = WORKING_DIR / 'runs'\n"
            "RUN_NAME = 'phone_rtdetr_l_ultralight_finetune_416'\n\n"
            "yaml_candidates = sorted(INPUT_ROOT.rglob('data.yaml'))\n"
            "if not yaml_candidates:\n"
            "    zip_candidates = sorted(INPUT_ROOT.rglob('*.zip'))\n"
            "    if zip_candidates:\n"
            "        extract_root = WORKING_DIR / 'extracted_dataset'\n"
            "        if extract_root.exists():\n"
            "            shutil.rmtree(extract_root)\n"
            "        extract_root.mkdir(parents=True, exist_ok=True)\n"
            "        print('Extracting:', zip_candidates[0])\n"
            "        with zipfile.ZipFile(zip_candidates[0]) as zf:\n"
            "            zf.extractall(extract_root)\n"
            "        yaml_candidates = sorted(extract_root.rglob('data.yaml'))\n\n"
            "if not yaml_candidates:\n"
            "    raise FileNotFoundError('No data.yaml found. Add the ultralight dataset zip to this notebook input.')\n\n"
            "source_yaml = yaml_candidates[0]\n"
            "dataset_root = source_yaml.parent\n"
            "print('Dataset root:', dataset_root)\n\n"
            "with open(source_yaml, 'r', encoding='utf-8') as f:\n"
            "    data_cfg = yaml.safe_load(f) or {}\n\n"
            "data_cfg['path'] = str(dataset_root)\n"
            "data_cfg['train'] = 'train/images'\n"
            "data_cfg['val'] = 'valid/images'\n"
            "data_cfg['test'] = 'test/images'\n"
            "data_cfg['nc'] = 1\n"
            "data_cfg['names'] = ['Mobile-phone']\n\n"
            "work_yaml = WORKING_DIR / 'mobilephone_rtdetr_ultralight_data.yaml'\n"
            "with open(work_yaml, 'w', encoding='utf-8') as f:\n"
            "    yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)\n"
            "print(work_yaml.read_text())\n"
        ),
        md("## 2. Dataset Sanity Check"),
        code(
            "IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}\n\n"
            "def count_split(split):\n"
            "    image_dir = dataset_root / split / 'images'\n"
            "    label_dir = dataset_root / split / 'labels'\n"
            "    images = [p for p in image_dir.glob('*') if p.suffix.lower() in IMG_EXTS]\n"
            "    labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []\n"
            "    image_stems = {p.stem for p in images}\n"
            "    label_stems = {p.stem for p in labels}\n"
            "    bad = 0\n"
            "    boxes = 0\n"
            "    classes = set()\n"
            "    for label in labels:\n"
            "        for line in label.read_text(encoding='utf-8', errors='replace').splitlines():\n"
            "            if not line.strip():\n"
            "                continue\n"
            "            parts = line.split()\n"
            "            if len(parts) != 5:\n"
            "                bad += 1\n"
            "                continue\n"
            "            try:\n"
            "                cls = int(float(parts[0]))\n"
            "                vals = list(map(float, parts[1:5]))\n"
            "                classes.add(cls)\n"
            "                boxes += 1\n"
            "                if cls != 0 or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:\n"
            "                    bad += 1\n"
            "            except Exception:\n"
            "                bad += 1\n"
            "    return {\n"
            "        'images': len(images), 'labels': len(labels),\n"
            "        'missing_labels': len(image_stems - label_stems),\n"
            "        'extra_labels': len(label_stems - image_stems),\n"
            "        'boxes': boxes, 'classes': sorted(classes), 'bad': bad,\n"
            "    }\n\n"
            "stats = {split: count_split(split) for split in ['train', 'valid', 'test']}\n"
            "print(stats)\n"
            "for split, row in stats.items():\n"
            "    assert row['images'] > 0 and row['labels'] > 0, split\n"
            "    assert row['missing_labels'] == 0 and row['extra_labels'] == 0, row\n"
            "    assert row['bad'] == 0, row\n"
            "    assert row['classes'] == [0], row\n"
        ),
        md("## 3. Ultralight Fine-Tune"),
        code(
            "MODEL_NAME = 'rtdetr-l.pt'\n"
            "EPOCHS = 5\n"
            "IMGSZ = 416\n"
            "BATCH_CANDIDATES = [24, 16, 12, 8, 4]\n"
            "PATIENCE = 3\n"
            "FREEZE_LAYERS = 20\n"
            "DEVICE = 0\n\n"
            "def train_once(batch):\n"
            "    model = RTDETR(MODEL_NAME)\n"
            "    return model.train(\n"
            "        data=str(work_yaml),\n"
            "        epochs=EPOCHS,\n"
            "        imgsz=IMGSZ,\n"
            "        batch=batch,\n"
            "        device=DEVICE,\n"
            "        workers=4,\n"
            "        patience=PATIENCE,\n"
            "        project=str(RUNS_DIR),\n"
            "        name=RUN_NAME,\n"
            "        exist_ok=True,\n"
            "        pretrained=True,\n"
            "        optimizer='AdamW',\n"
            "        lr0=0.0001,\n"
            "        weight_decay=0.0005,\n"
            "        cos_lr=True,\n"
            "        amp=True,\n"
            "        cache=False,\n"
            "        deterministic=False,\n"
            "        mosaic=0.0,\n"
            "        scale=0.15,\n"
            "        freeze=FREEZE_LAYERS,\n"
            "        plots=True,\n"
            "    )\n\n"
            "last_error = None\n"
            "for batch in BATCH_CANDIDATES:\n"
            "    try:\n"
            "        BATCH = batch\n"
            "        print(f'Ultralight RT-DETR fine-tune: imgsz={IMGSZ}, batch={BATCH}, epochs={EPOCHS}, freeze={FREEZE_LAYERS}')\n"
            "        train_results = train_once(BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'CUDA OOM at batch={batch}; retrying smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('Training failed for all batch sizes.') from last_error\n"
        ),
        md("## 4. Validate, Predict, and Export"),
        code(
            "run_dir = RUNS_DIR / 'detect' / RUN_NAME\n"
            "best_pt = run_dir / 'weights' / 'best.pt'\n"
            "last_pt = run_dir / 'weights' / 'last.pt'\n"
            "assert best_pt.exists(), f'best.pt not found: {best_pt}'\n\n"
            "best_model = RTDETR(str(best_pt))\n"
            "metrics = best_model.val(\n"
            "    data=str(work_yaml), split='test', imgsz=IMGSZ, batch=BATCH, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name=RUN_NAME + '_test_eval', exist_ok=True, plots=True,\n"
            ")\n"
            "print(metrics)\n\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:32]\n"
            "best_model.predict(\n"
            "    source=[str(p) for p in sample_images], conf=0.25, imgsz=IMGSZ, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name='rtdetr_ultralight_predict_samples', exist_ok=True, save=True,\n"
            ")\n\n"
            "package_dir = WORKING_DIR / 'phone_rtdetr_ultralight_export'\n"
            "if package_dir.exists():\n"
            "    shutil.rmtree(package_dir)\n"
            "package_dir.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(best_pt, package_dir / 'phone_rtdetr_l_ultralight_best.pt')\n"
            "shutil.copy2(best_pt, package_dir / 'phone_rtdetr_l_best.pt')\n"
            "if last_pt.exists():\n"
            "    shutil.copy2(last_pt, package_dir / 'phone_rtdetr_l_ultralight_last.pt')\n"
            "shutil.copy2(work_yaml, package_dir / 'mobilephone_rtdetr_ultralight_data.yaml')\n\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, package_dir / filename)\n\n"
            "zip_path = shutil.make_archive(str(WORKING_DIR / 'phone_rtdetr_ultralight_finetune_results'), 'zip', package_dir)\n"
            "print('Download:', zip_path)\n"
        ),
        md(
            "## 5. Use in Local App\n\n"
            "Download `phone_rtdetr_ultralight_finetune_results.zip`, extract `phone_rtdetr_l_best.pt`, and copy it to:\n\n"
            "`D:\\Work\\HocKy3\\ComputerVision\\MDDAD\\models\\phone_rtdetr_l.pt`\n"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output_path: Path) -> None:
    output_path.write_text(
        "# Ultralight RT-DETR Phone Fine-Tuning\n\n"
        "This notebook is lighter than the practical-final RT-DETR notebook.\n\n"
        "## Files\n\n"
        "- Notebook: `phone_detection_rtdetr_ultralight_finetune_kaggle.ipynb`\n"
        "- Dataset: `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_ultralight_finetune_kaggle.zip`\n\n"
        "## Profile\n\n"
        "- Model: `rtdetr-l.pt`\n"
        "- Train images: 1500\n"
        "- Valid images: 326\n"
        "- Test images: 185\n"
        "- Image size: 416\n"
        "- Epochs: 5\n"
        "- Freeze layers: 20\n"
        "- Batch retry: 24, 16, 12, 8, 4\n\n"
        "This is a quick RT-DETR fine-tuning run for comparison/demo, not the strongest final model.\n",
        encoding="utf-8",
    )


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
