from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
YOLO_NOTEBOOK = KAGGLE_DIR / "phone_detection_yolo26n_light_kaggle_train.ipynb"
RTDETR_NOTEBOOK = KAGGLE_DIR / "phone_detection_rtdetr_l_light_kaggle_train.ipynb"
README_PATH = KAGGLE_DIR / "README_KAGGLE_LIGHT_MODEL_COMPARISON.md"


def main() -> int:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    write_yolo_notebook(YOLO_NOTEBOOK)
    write_rtdetr_notebook(RTDETR_NOTEBOOK)
    write_readme(README_PATH)
    print(f"YOLO26n notebook: {YOLO_NOTEBOOK}")
    print(f"RT-DETR-L notebook: {RTDETR_NOTEBOOK}")
    print(f"README: {README_PATH}")
    return 0


def common_setup_cells(title: str, model_note: str) -> list[dict]:
    return [
        md(
            f"# {title}\n\n"
            f"{model_note}\n\n"
            "Use the same dataset for both comparison notebooks:\n\n"
            "`mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`\n\n"
            "This dataset uses YOLO detection labels, which Ultralytics YOLO and RT-DETR both read directly."
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
            "import yaml\n\n"
            "print('Python:', sys.version)\n"
            "print('Torch:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('GPU count:', torch.cuda.device_count())\n"
            "for i in range(torch.cuda.device_count()):\n"
            "    print(f'GPU {i}:', torch.cuda.get_device_name(i))\n"
        ),
        md("## 1. Locate dataset and normalize `data.yaml`"),
        code(
            "INPUT_ROOT = Path('/kaggle/input')\n"
            "WORKING_DIR = Path('/kaggle/working')\n"
            "RUNS_DIR = WORKING_DIR / 'runs'\n\n"
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
            "    raise FileNotFoundError('No data.yaml found. Add the practical-final dataset zip to this notebook input.')\n\n"
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
            "work_yaml = WORKING_DIR / 'mobilephone_light_compare_data.yaml'\n"
            "with open(work_yaml, 'w', encoding='utf-8') as f:\n"
            "    yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)\n"
            "print(work_yaml.read_text())\n"
        ),
        md("## 2. Dataset sanity check"),
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
            "        'images': len(images),\n"
            "        'labels': len(labels),\n"
            "        'missing_labels': len(image_stems - label_stems),\n"
            "        'extra_labels': len(label_stems - image_stems),\n"
            "        'boxes': boxes,\n"
            "        'classes': sorted(classes),\n"
            "        'bad': bad,\n"
            "    }\n\n"
            "stats = {split: count_split(split) for split in ['train', 'valid', 'test']}\n"
            "print(stats)\n"
            "for split, row in stats.items():\n"
            "    assert row['images'] > 0 and row['labels'] > 0, split\n"
            "    assert row['missing_labels'] == 0 and row['extra_labels'] == 0, row\n"
            "    assert row['bad'] == 0, row\n"
            "    assert row['classes'] == [0], row\n"
        ),
    ]


def write_yolo_notebook(output_path: Path) -> None:
    cells = common_setup_cells(
        "YOLO26n Lightweight Phone Detector Baseline",
        "Trains the latest/lightest Ultralytics YOLO family model available in current docs: `yolo26n.pt`. "
        "If the Kaggle Ultralytics package does not support YOLO26 yet, the notebook falls back to `yolo11n.pt` and reports it clearly.",
    )
    cells += [
        md("## 3. Train YOLO lightweight baseline"),
        code(
            "from ultralytics import YOLO\n\n"
            "PREFERRED_MODEL = 'yolo26n.pt'\n"
            "FALLBACK_MODEL = 'yolo11n.pt'\n"
            "RUN_NAME = 'phone_yolo26n_light_compare'\n"
            "EPOCHS = 12\n"
            "IMGSZ = 512\n"
            "BATCH_CANDIDATES = [64, 48, 32, 16]\n"
            "PATIENCE = 5\n"
            "DEVICE = 0 if torch.cuda.is_available() else 'cpu'\n\n"
            "def load_model():\n"
            "    try:\n"
            "        print('Loading preferred model:', PREFERRED_MODEL)\n"
            "        return YOLO(PREFERRED_MODEL), PREFERRED_MODEL\n"
            "    except Exception as exc:\n"
            "        print('Preferred model unavailable, falling back. Error:', repr(exc))\n"
            "        print('Loading fallback model:', FALLBACK_MODEL)\n"
            "        return YOLO(FALLBACK_MODEL), FALLBACK_MODEL\n\n"
            "def train_once(batch):\n"
            "    model, used_model_name = load_model()\n"
            "    results = model.train(\n"
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
            "        optimizer='auto',\n"
            "        cos_lr=True,\n"
            "        amp=True,\n"
            "        cache=False,\n"
            "        plots=True,\n"
            "    )\n"
            "    return model, used_model_name, results\n\n"
            "last_error = None\n"
            "for batch in BATCH_CANDIDATES:\n"
            "    try:\n"
            "        BATCH = batch\n"
            "        print(f'Training YOLO lightweight baseline: imgsz={IMGSZ}, batch={BATCH}, epochs={EPOCHS}, device={DEVICE}')\n"
            "        model, used_model_name, train_results = train_once(BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'CUDA OOM at batch={batch}; retrying smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('YOLO lightweight training failed for all batch sizes.') from last_error\n\n"
            "print('Used model:', used_model_name)\n"
        ),
        md("## 4. Validate, predict samples, and package"),
        code(
            "run_dir = RUNS_DIR / 'detect' / RUN_NAME\n"
            "best_pt = run_dir / 'weights' / 'best.pt'\n"
            "last_pt = run_dir / 'weights' / 'last.pt'\n"
            "assert best_pt.exists(), f'best.pt not found: {best_pt}'\n\n"
            "best_model = YOLO(str(best_pt))\n"
            "metrics = best_model.val(\n"
            "    data=str(work_yaml),\n"
            "    split='test',\n"
            "    imgsz=IMGSZ,\n"
            "    batch=BATCH,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name=RUN_NAME + '_test_eval',\n"
            "    exist_ok=True,\n"
            "    plots=True,\n"
            ")\n"
            "print(metrics)\n\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:32]\n"
            "best_model.predict(\n"
            "    source=[str(p) for p in sample_images],\n"
            "    conf=0.25,\n"
            "    imgsz=IMGSZ,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name='yolo_light_predict_samples',\n"
            "    exist_ok=True,\n"
            "    save=True,\n"
            ")\n\n"
            "package_dir = WORKING_DIR / 'phone_yolo_light_export'\n"
            "if package_dir.exists():\n"
            "    shutil.rmtree(package_dir)\n"
            "package_dir.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(best_pt, package_dir / 'phone_yolo_light_best.pt')\n"
            "if last_pt.exists():\n"
            "    shutil.copy2(last_pt, package_dir / 'phone_yolo_light_last.pt')\n"
            "shutil.copy2(work_yaml, package_dir / 'mobilephone_light_compare_data.yaml')\n"
            "(package_dir / 'MODEL_USED.txt').write_text(used_model_name, encoding='utf-8')\n\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, package_dir / filename)\n\n"
            "zip_path = shutil.make_archive(str(WORKING_DIR / 'phone_yolo_light_training_results'), 'zip', package_dir)\n"
            "print('Download:', zip_path)\n"
        ),
    ]
    write_notebook(output_path, cells)


def write_rtdetr_notebook(output_path: Path) -> None:
    cells = common_setup_cells(
        "RT-DETR-L Lightweight Phone Detector Baseline",
        "Trains the lightest official Ultralytics RT-DETR pretrained option available in current docs: `rtdetr-l.pt`. "
        "Ultralytics RT-DETR also provides `rtdetr-x.pt`, but X is larger/slower.",
    )
    cells += [
        md("## 3. Train RT-DETR light baseline"),
        code(
            "from ultralytics import RTDETR\n\n"
            "MODEL_NAME = 'rtdetr-l.pt'\n"
            "RUN_NAME = 'phone_rtdetr_l_light_compare'\n"
            "EPOCHS = 8\n"
            "IMGSZ = 512\n"
            "BATCH_CANDIDATES = [16, 12, 8, 4]\n"
            "PATIENCE = 4\n"
            "FREEZE_LAYERS = 10\n"
            "DEVICE = 0 if torch.cuda.is_available() else 'cpu'\n\n"
            "def train_once(batch):\n"
            "    model = RTDETR(MODEL_NAME)\n"
            "    results = model.train(\n"
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
            "        scale=0.20,\n"
            "        freeze=FREEZE_LAYERS,\n"
            "        plots=True,\n"
            "    )\n"
            "    return model, results\n\n"
            "last_error = None\n"
            "for batch in BATCH_CANDIDATES:\n"
            "    try:\n"
            "        BATCH = batch\n"
            "        print(f'Training RT-DETR light baseline: imgsz={IMGSZ}, batch={BATCH}, epochs={EPOCHS}, freeze={FREEZE_LAYERS}, device={DEVICE}')\n"
            "        model, train_results = train_once(BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'CUDA OOM at batch={batch}; retrying smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('RT-DETR training failed for all batch sizes.') from last_error\n"
        ),
        md("## 4. Validate, predict samples, and package"),
        code(
            "run_dir = RUNS_DIR / 'detect' / RUN_NAME\n"
            "best_pt = run_dir / 'weights' / 'best.pt'\n"
            "last_pt = run_dir / 'weights' / 'last.pt'\n"
            "assert best_pt.exists(), f'best.pt not found: {best_pt}'\n\n"
            "best_model = RTDETR(str(best_pt))\n"
            "metrics = best_model.val(\n"
            "    data=str(work_yaml),\n"
            "    split='test',\n"
            "    imgsz=IMGSZ,\n"
            "    batch=BATCH,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name=RUN_NAME + '_test_eval',\n"
            "    exist_ok=True,\n"
            "    plots=True,\n"
            ")\n"
            "print(metrics)\n\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:32]\n"
            "best_model.predict(\n"
            "    source=[str(p) for p in sample_images],\n"
            "    conf=0.25,\n"
            "    imgsz=IMGSZ,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name='rtdetr_light_predict_samples',\n"
            "    exist_ok=True,\n"
            "    save=True,\n"
            ")\n\n"
            "package_dir = WORKING_DIR / 'phone_rtdetr_light_export'\n"
            "if package_dir.exists():\n"
            "    shutil.rmtree(package_dir)\n"
            "package_dir.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(best_pt, package_dir / 'phone_rtdetr_l_best.pt')\n"
            "if last_pt.exists():\n"
            "    shutil.copy2(last_pt, package_dir / 'phone_rtdetr_l_last.pt')\n"
            "shutil.copy2(work_yaml, package_dir / 'mobilephone_light_compare_data.yaml')\n"
            "(package_dir / 'MODEL_USED.txt').write_text(MODEL_NAME, encoding='utf-8')\n\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, package_dir / filename)\n\n"
            "zip_path = shutil.make_archive(str(WORKING_DIR / 'phone_rtdetr_light_training_results'), 'zip', package_dir)\n"
            "print('Download:', zip_path)\n"
        ),
    ]
    write_notebook(output_path, cells)


def write_notebook(output_path: Path, cells: list[dict]) -> None:
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
        "# Lightweight Model Comparison for Phone Detection\n\n"
        "Use these notebooks before deciding the final fine-tuning direction.\n\n"
        "## Shared dataset\n\n"
        "Use the same practical-final dataset for both notebooks:\n\n"
        "`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`\n\n"
        "It contains 3000 train images, 326 validation images, and 185 test images.\n\n"
        "## Notebook A: YOLO latest/lightest\n\n"
        "`phone_detection_yolo26n_light_kaggle_train.ipynb`\n\n"
        "- Preferred model: `yolo26n.pt`\n"
        "- Fallback if unsupported by Kaggle package: `yolo11n.pt`\n"
        "- Epochs: 12\n"
        "- Image size: 512\n"
        "- Batch retry: 64, 48, 32, 16\n\n"
        "## Notebook B: RT-DETR lightest official Ultralytics option\n\n"
        "`phone_detection_rtdetr_l_light_kaggle_train.ipynb`\n\n"
        "- Model: `rtdetr-l.pt`\n"
        "- Epochs: 8\n"
        "- Image size: 512\n"
        "- Batch retry: 16, 12, 8, 4\n"
        "- Freeze layers: 10\n\n"
        "## Compare using\n\n"
        "- mAP50\n"
        "- mAP50-95\n"
        "- Precision\n"
        "- Recall\n"
        "- Training time\n"
        "- Inference speed\n"
        "- Sample prediction images\n\n",
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
