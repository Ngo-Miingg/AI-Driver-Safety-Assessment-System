from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
NOTEBOOK_PATH = KAGGLE_DIR / "phone_detection_combined_light_comparison_kaggle.ipynb"
README_PATH = KAGGLE_DIR / "README_KAGGLE_COMBINED_LIGHT_COMPARISON.md"


def main() -> int:
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_PATH)
    write_readme(README_PATH)
    print(f"Combined comparison notebook: {NOTEBOOK_PATH}")
    print(f"README: {README_PATH}")
    return 0


def write_notebook(output_path: Path) -> None:
    cells = [
        md(
            "# Combined Phone Detector Comparison: YOLO vs RT-DETR\n\n"
            "This notebook runs two lightweight phone detector training jobs sequentially, suitable for leaving overnight.\n\n"
            "Shared dataset:\n"
            "- `mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`\n"
            "- 3000 train images, 326 validation images, 185 test images\n\n"
            "Outputs are separated into two large folders:\n"
            "- `/kaggle/working/compare_outputs/yolo_light/`\n"
            "- `/kaggle/working/compare_outputs/rtdetr_light/`\n\n"
            "Training profile is intentionally moderate, not maximum-heavy."
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
            "from ultralytics import YOLO, RTDETR\n\n"
            "print('Python:', sys.version)\n"
            "print('Torch:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('GPU count:', torch.cuda.device_count())\n"
            "for i in range(torch.cuda.device_count()):\n"
            "    print(f'GPU {i}:', torch.cuda.get_device_name(i))\n"
            "if not torch.cuda.is_available():\n"
            "    raise RuntimeError('CUDA is not available. In Kaggle, open Settings, set Accelerator to GPU, then restart the session and run again. Do not train these models on CPU.')\n"
        ),
        md("## 1. Locate Dataset"),
        code(
            "INPUT_ROOT = Path('/kaggle/input')\n"
            "WORKING_DIR = Path('/kaggle/working')\n"
            "RUNS_DIR = WORKING_DIR / 'runs'\n"
            "OUTPUT_ROOT = WORKING_DIR / 'compare_outputs'\n"
            "YOLO_OUT = OUTPUT_ROOT / 'yolo_light'\n"
            "RTDETR_OUT = OUTPUT_ROOT / 'rtdetr_light'\n"
            "OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "YOLO_OUT.mkdir(parents=True, exist_ok=True)\n"
            "RTDETR_OUT.mkdir(parents=True, exist_ok=True)\n\n"
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
            "work_yaml = WORKING_DIR / 'mobilephone_combined_compare_data.yaml'\n"
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
        md("## 3. Train YOLO Lightweight Model"),
        code(
            "YOLO_PREFERRED_MODEL = 'yolo26n.pt'\n"
            "YOLO_FALLBACK_MODEL = 'yolo11n.pt'\n"
            "YOLO_RUN_NAME = 'compare_yolo_light'\n"
            "YOLO_EPOCHS = 8\n"
            "YOLO_IMGSZ = 512\n"
            "YOLO_BATCH_CANDIDATES = [64, 48, 32, 16]\n"
            "YOLO_PATIENCE = 4\n"
            "DEVICE = 0\n\n"
            "def load_yolo_model():\n"
            "    try:\n"
            "        print('Loading preferred YOLO:', YOLO_PREFERRED_MODEL)\n"
            "        return YOLO(YOLO_PREFERRED_MODEL), YOLO_PREFERRED_MODEL\n"
            "    except Exception as exc:\n"
            "        print('YOLO preferred model unavailable, fallback used. Error:', repr(exc))\n"
            "        return YOLO(YOLO_FALLBACK_MODEL), YOLO_FALLBACK_MODEL\n\n"
            "def train_yolo_once(batch):\n"
            "    model, used = load_yolo_model()\n"
            "    result = model.train(\n"
            "        data=str(work_yaml), epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=batch,\n"
            "        device=DEVICE, workers=4, patience=YOLO_PATIENCE,\n"
            "        project=str(RUNS_DIR), name=YOLO_RUN_NAME, exist_ok=True,\n"
            "        pretrained=True, optimizer='auto', cos_lr=True, amp=True, cache=False, plots=True,\n"
            "    )\n"
            "    return model, used, result\n\n"
            "last_error = None\n"
            "for batch in YOLO_BATCH_CANDIDATES:\n"
            "    try:\n"
            "        YOLO_BATCH = batch\n"
            "        print(f'Training YOLO: batch={YOLO_BATCH}, epochs={YOLO_EPOCHS}, imgsz={YOLO_IMGSZ}')\n"
            "        yolo_model, yolo_used_model, yolo_train_results = train_yolo_once(YOLO_BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'YOLO OOM at batch={batch}; retrying smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('YOLO training failed for all batch sizes.') from last_error\n\n"
            "print('YOLO used model:', yolo_used_model)\n"
        ),
        md("## 4. Validate and Export YOLO"),
        code(
            "def resolve_run_dir(runs_dir, run_name):\n"
            "    candidates = [\n"
            "        runs_dir / run_name,\n"
            "        runs_dir / 'detect' / run_name,\n"
            "    ]\n"
            "    for candidate in candidates:\n"
            "        if (candidate / 'weights' / 'best.pt').exists():\n"
            "            return candidate\n"
            "    matches = sorted(runs_dir.rglob(f'{run_name}/weights/best.pt'))\n"
            "    if matches:\n"
            "        return matches[0].parents[1]\n"
            "    print('Available run folders:')\n"
            "    for path in sorted(runs_dir.rglob('*')):\n"
            "        if path.is_dir() and path.name == 'weights':\n"
            "            print(path.parent)\n"
            "    raise FileNotFoundError(f'Could not locate best.pt for run {run_name}')\n\n"
            "yolo_run_dir = resolve_run_dir(RUNS_DIR, YOLO_RUN_NAME)\n"
            "yolo_best_pt = yolo_run_dir / 'weights' / 'best.pt'\n"
            "yolo_last_pt = yolo_run_dir / 'weights' / 'last.pt'\n"
            "assert yolo_best_pt.exists(), f'YOLO best.pt not found: {yolo_best_pt}'\n\n"
            "yolo_best_model = YOLO(str(yolo_best_pt))\n"
            "yolo_metrics = yolo_best_model.val(\n"
            "    data=str(work_yaml), split='test', imgsz=YOLO_IMGSZ, batch=YOLO_BATCH, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name=YOLO_RUN_NAME + '_test_eval', exist_ok=True, plots=True,\n"
            ")\n"
            "print(yolo_metrics)\n\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:32]\n"
            "yolo_best_model.predict(\n"
            "    source=[str(p) for p in sample_images], conf=0.25, imgsz=YOLO_IMGSZ, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name='compare_yolo_predict_samples', exist_ok=True, save=True,\n"
            ")\n\n"
            "shutil.copy2(yolo_best_pt, YOLO_OUT / 'phone_yolo_light_best.pt')\n"
            "if yolo_last_pt.exists():\n"
            "    shutil.copy2(yolo_last_pt, YOLO_OUT / 'phone_yolo_light_last.pt')\n"
            "shutil.copy2(work_yaml, YOLO_OUT / 'mobilephone_compare_data.yaml')\n"
            "(YOLO_OUT / 'MODEL_USED.txt').write_text(yolo_used_model, encoding='utf-8')\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = yolo_run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, YOLO_OUT / filename)\n"
        ),
        md("## 5. Train RT-DETR Lightweight Model"),
        code(
            "RTDETR_MODEL_NAME = 'rtdetr-l.pt'\n"
            "RTDETR_RUN_NAME = 'compare_rtdetr_light'\n"
            "RTDETR_EPOCHS = 5\n"
            "RTDETR_IMGSZ = 512\n"
            "RTDETR_BATCH_CANDIDATES = [16, 12, 8, 4]\n"
            "RTDETR_PATIENCE = 3\n"
            "RTDETR_FREEZE_LAYERS = 10\n\n"
            "def train_rtdetr_once(batch):\n"
            "    model = RTDETR(RTDETR_MODEL_NAME)\n"
            "    result = model.train(\n"
            "        data=str(work_yaml), epochs=RTDETR_EPOCHS, imgsz=RTDETR_IMGSZ, batch=batch,\n"
            "        device=DEVICE, workers=4, patience=RTDETR_PATIENCE,\n"
            "        project=str(RUNS_DIR), name=RTDETR_RUN_NAME, exist_ok=True,\n"
            "        pretrained=True, optimizer='AdamW', lr0=0.0001, weight_decay=0.0005,\n"
            "        cos_lr=True, amp=True, cache=False, deterministic=False,\n"
            "        mosaic=0.0, scale=0.20, freeze=RTDETR_FREEZE_LAYERS, plots=True,\n"
            "    )\n"
            "    return model, result\n\n"
            "last_error = None\n"
            "for batch in RTDETR_BATCH_CANDIDATES:\n"
            "    try:\n"
            "        RTDETR_BATCH = batch\n"
            "        print(f'Training RT-DETR: batch={RTDETR_BATCH}, epochs={RTDETR_EPOCHS}, imgsz={RTDETR_IMGSZ}')\n"
            "        rtdetr_model, rtdetr_train_results = train_rtdetr_once(RTDETR_BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'RT-DETR OOM at batch={batch}; retrying smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('RT-DETR training failed for all batch sizes.') from last_error\n"
        ),
        md("## 6. Validate and Export RT-DETR"),
        code(
            "rtdetr_run_dir = resolve_run_dir(RUNS_DIR, RTDETR_RUN_NAME)\n"
            "rtdetr_best_pt = rtdetr_run_dir / 'weights' / 'best.pt'\n"
            "rtdetr_last_pt = rtdetr_run_dir / 'weights' / 'last.pt'\n"
            "assert rtdetr_best_pt.exists(), f'RT-DETR best.pt not found: {rtdetr_best_pt}'\n\n"
            "rtdetr_best_model = RTDETR(str(rtdetr_best_pt))\n"
            "rtdetr_metrics = rtdetr_best_model.val(\n"
            "    data=str(work_yaml), split='test', imgsz=RTDETR_IMGSZ, batch=RTDETR_BATCH, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name=RTDETR_RUN_NAME + '_test_eval', exist_ok=True, plots=True,\n"
            ")\n"
            "print(rtdetr_metrics)\n\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:32]\n"
            "rtdetr_best_model.predict(\n"
            "    source=[str(p) for p in sample_images], conf=0.25, imgsz=RTDETR_IMGSZ, device=DEVICE,\n"
            "    project=str(RUNS_DIR), name='compare_rtdetr_predict_samples', exist_ok=True, save=True,\n"
            ")\n\n"
            "shutil.copy2(rtdetr_best_pt, RTDETR_OUT / 'phone_rtdetr_l_best.pt')\n"
            "if rtdetr_last_pt.exists():\n"
            "    shutil.copy2(rtdetr_last_pt, RTDETR_OUT / 'phone_rtdetr_l_last.pt')\n"
            "shutil.copy2(work_yaml, RTDETR_OUT / 'mobilephone_compare_data.yaml')\n"
            "(RTDETR_OUT / 'MODEL_USED.txt').write_text(RTDETR_MODEL_NAME, encoding='utf-8')\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = rtdetr_run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, RTDETR_OUT / filename)\n"
        ),
        md("## 7. Package Both Outputs"),
        code(
            "summary = {\n"
            "    'dataset': str(dataset_root),\n"
            "    'yolo_model_used': yolo_used_model,\n"
            "    'yolo_epochs': YOLO_EPOCHS,\n"
            "    'yolo_imgsz': YOLO_IMGSZ,\n"
            "    'rtdetr_model_used': RTDETR_MODEL_NAME,\n"
            "    'rtdetr_epochs': RTDETR_EPOCHS,\n"
            "    'rtdetr_imgsz': RTDETR_IMGSZ,\n"
            "}\n"
            "(OUTPUT_ROOT / 'comparison_summary.json').write_text(__import__('json').dumps(summary, indent=2), encoding='utf-8')\n"
            "zip_path = shutil.make_archive(str(WORKING_DIR / 'phone_detector_combined_light_comparison_results'), 'zip', OUTPUT_ROOT)\n"
            "print('Output root:', OUTPUT_ROOT)\n"
            "print('Download:', zip_path)\n"
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
        "# Combined Lightweight YOLO vs RT-DETR Comparison\n\n"
        "Use this single notebook when you want to start a run and leave it unattended.\n\n"
        "Notebook:\n\n"
        "`phone_detection_combined_light_comparison_kaggle.ipynb`\n\n"
        "Dataset to upload:\n\n"
        "`dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_practical_final_kaggle.zip`\n\n"
        "Outputs on Kaggle:\n\n"
        "- `/kaggle/working/compare_outputs/yolo_light/`\n"
        "- `/kaggle/working/compare_outputs/rtdetr_light/`\n"
        "- `/kaggle/working/phone_detector_combined_light_comparison_results.zip`\n\n"
        "Moderate configuration:\n\n"
        "- YOLO: `yolo26n.pt`, fallback `yolo11n.pt`, 8 epochs, imgsz 512\n"
        "- RT-DETR: `rtdetr-l.pt`, 5 epochs, imgsz 512, freeze 10 layers\n\n",
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
