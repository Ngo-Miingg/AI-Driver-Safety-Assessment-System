from __future__ import annotations

import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "dataset" / "roboflow_mobilephone_wusj2_v7" / "extracted"
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
KAGGLE_DATASET_DIR = PROJECT_ROOT / "dataset" / "kaggle_upload"
KAGGLE_DATASET_ZIP = KAGGLE_DATASET_DIR / "mobilephone_wusj2_v7_rtdetr_kaggle.zip"
NOTEBOOK_PATH = KAGGLE_DIR / "phone_detection_rtdetr_kaggle_train.ipynb"
README_PATH = KAGGLE_DIR / "README_KAGGLE_RTDETR_PHONE_TRAIN.md"


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
"""


def main() -> int:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DATASET}")

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    validate_source_dataset()
    write_dataset_zip(KAGGLE_DATASET_ZIP)
    write_notebook(NOTEBOOK_PATH)
    write_readme(README_PATH)

    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Kaggle dataset zip: {KAGGLE_DATASET_ZIP}")
    print(f"README: {README_PATH}")
    return 0


def validate_source_dataset() -> None:
    for split in ("train", "valid", "test"):
        image_dir = SOURCE_DATASET / split / "images"
        label_dir = SOURCE_DATASET / split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing {split}/images or {split}/labels in {SOURCE_DATASET}")
        image_count = sum(1 for p in image_dir.iterdir() if p.is_file())
        label_count = sum(1 for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        if image_count == 0 or label_count == 0:
            raise ValueError(f"Empty split {split}: images={image_count}, labels={label_count}")


def write_dataset_zip(output_path: Path) -> None:
    allowed_roots = {"train", "valid", "test"}
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", FIXED_DATA_YAML)
        for child in sorted(SOURCE_DATASET.iterdir()):
            if child.name not in allowed_roots:
                continue
            for file_path in sorted(child.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(SOURCE_DATASET).as_posix())
        for readme_name in ("README.dataset.txt", "README.roboflow.txt"):
            readme_path = SOURCE_DATASET / readme_name
            if readme_path.exists():
                zf.write(readme_path, readme_name)


def write_notebook(output_path: Path) -> None:
    cells = [
        md(
            "# RT-DETR Phone Detection Training on Kaggle\n\n"
            "This notebook fine-tunes `rtdetr-l.pt` for one-class phone detection.\n\n"
            "Recommended Kaggle settings:\n"
            "- Accelerator: GPU T4 x2\n"
            "- Internet: ON for the first run, so `ultralytics` can be installed if needed\n"
            "- Dataset: add `mobilephone_wusj2_v7_rtdetr_kaggle.zip` as a Kaggle Dataset\n\n"
            "Main output:\n"
            "- `phone_rtdetr_l_best.pt`\n"
            "- `phone_rtdetr_l_last.pt`\n"
            "- `phone_rtdetr_training_results.zip`\n"
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
        ),
        md("## 1. Locate and normalize the dataset"),
        code(
            "INPUT_ROOT = Path('/kaggle/input')\n"
            "WORKING_DIR = Path('/kaggle/working')\n"
            "RUNS_DIR = WORKING_DIR / 'runs'\n"
            "RUN_NAME = 'phone_rtdetr_l_t4_512'\n\n"
            "yaml_candidates = sorted(INPUT_ROOT.rglob('data.yaml'))\n"
            "if not yaml_candidates:\n"
            "    zip_candidates = sorted(INPUT_ROOT.rglob('*.zip'))\n"
            "    if zip_candidates:\n"
            "        extract_root = WORKING_DIR / 'extracted_dataset'\n"
            "        if extract_root.exists():\n"
            "            shutil.rmtree(extract_root)\n"
            "        extract_root.mkdir(parents=True, exist_ok=True)\n"
            "        print('No extracted data.yaml found. Extracting:', zip_candidates[0])\n"
            "        with zipfile.ZipFile(zip_candidates[0]) as zf:\n"
            "            zf.extractall(extract_root)\n"
            "        yaml_candidates = sorted(extract_root.rglob('data.yaml'))\n\n"
            "if not yaml_candidates:\n"
            "    raise FileNotFoundError('No data.yaml found under /kaggle/input or inside uploaded zip dataset.')\n\n"
            "source_yaml = yaml_candidates[0]\n"
            "dataset_root = source_yaml.parent\n"
            "print('Using YAML:', source_yaml)\n"
            "print('Dataset root:', dataset_root)\n\n"
            "with open(source_yaml, 'r', encoding='utf-8') as f:\n"
            "    data_cfg = yaml.safe_load(f) or {}\n\n"
            "data_cfg['path'] = str(dataset_root)\n"
            "data_cfg['train'] = 'train/images'\n"
            "data_cfg['val'] = 'valid/images'\n"
            "data_cfg['test'] = 'test/images'\n"
            "data_cfg['nc'] = 1\n"
            "data_cfg['names'] = ['Mobile-phone']\n\n"
            "work_yaml = WORKING_DIR / 'mobilephone_rtdetr_data.yaml'\n"
            "with open(work_yaml, 'w', encoding='utf-8') as f:\n"
            "    yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)\n\n"
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
            "    boxes = 0\n"
            "    bad_lines = 0\n"
            "    class_ids = set()\n"
            "    for label in labels:\n"
            "        text = label.read_text(encoding='utf-8', errors='replace').strip()\n"
            "        for line in text.splitlines():\n"
            "            parts = line.split()\n"
            "            if len(parts) != 5:\n"
            "                bad_lines += 1\n"
            "                continue\n"
            "            try:\n"
            "                cls = int(float(parts[0]))\n"
            "                vals = list(map(float, parts[1:5]))\n"
            "                class_ids.add(cls)\n"
            "                if cls != 0 or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:\n"
            "                    bad_lines += 1\n"
            "                boxes += 1\n"
            "            except Exception:\n"
            "                bad_lines += 1\n"
            "    return {\n"
            "        'images': len(images),\n"
            "        'labels': len(labels),\n"
            "        'missing_labels': len(image_stems - label_stems),\n"
            "        'extra_labels': len(label_stems - image_stems),\n"
            "        'boxes': boxes,\n"
            "        'bad_label_lines': bad_lines,\n"
            "        'class_ids': sorted(class_ids),\n"
            "    }\n\n"
            "stats = {split: count_split(split) for split in ['train', 'valid', 'test']}\n"
            "stats\n"
        ),
        code(
            "for split, row in stats.items():\n"
            "    assert row['images'] > 0, f'{split} has no images'\n"
            "    assert row['labels'] > 0, f'{split} has no labels'\n"
            "    assert row['missing_labels'] == 0, f'{split} has missing labels'\n"
            "    assert row['bad_label_lines'] == 0, f'{split} has invalid label rows'\n"
            "    assert row['class_ids'] == [0], f'{split} has unexpected classes: {row[\"class_ids\"]}'\n"
            "print('Dataset sanity check passed.')\n"
        ),
        md(
            "## 3. Train RT-DETR-L\n\n"
            "`rtdetr-l.pt` is the recommended starting point for this project because it is stronger than a nano detector "
            "while still designed for real-time detection. If T4x2 training fails, the code falls back to single GPU "
            "with a smaller batch."
        ),
        code(
            "MODEL_NAME = 'rtdetr-l.pt'\n"
            "EPOCHS = 60\n"
            "IMGSZ = 512\n"
            "BATCH_CANDIDATES = [4, 2, 1]\n"
            "PATIENCE = 15\n\n"
            "# RT-DETR-L is memory-heavy on T4. Single GPU is more stable than Kaggle DDP here.\n"
            "DEVICE = 0 if torch.cuda.is_available() else 'cpu'\n\n"
            "print('Training device:', DEVICE)\n\n"
            "def train_once(device, batch):\n"
            "    model = RTDETR(MODEL_NAME)\n"
            "    return model.train(\n"
            "        data=str(work_yaml),\n"
            "        epochs=EPOCHS,\n"
            "        imgsz=IMGSZ,\n"
            "        batch=batch,\n"
            "        device=device,\n"
            "        workers=2,\n"
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
            "        scale=0.3,\n"
            "        plots=True,\n"
            "    )\n\n"
            "last_error = None\n"
            "for batch in BATCH_CANDIDATES:\n"
            "    try:\n"
            "        BATCH = batch\n"
            "        print(f'Trying training with batch={BATCH}, imgsz={IMGSZ}, device={DEVICE}')\n"
            "        train_results = train_once(DEVICE, BATCH)\n"
            "        break\n"
            "    except RuntimeError as exc:\n"
            "        last_error = exc\n"
            "        if 'out of memory' not in str(exc).lower():\n"
            "            raise\n"
            "        print(f'CUDA OOM with batch={batch}. Retrying with a smaller batch...')\n"
            "        if torch.cuda.is_available():\n"
            "            torch.cuda.empty_cache()\n"
            "else:\n"
            "    raise RuntimeError('RT-DETR-L training failed for all batch sizes.') from last_error\n"
        ),
        md("## 4. Validate on the test split"),
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
            "print(metrics)\n"
        ),
        md("## 5. Predict samples and package outputs"),
        code(
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:24]\n"
            "assert sample_images, 'No test images found for sample prediction'\n\n"
            "best_model.predict(\n"
            "    source=[str(p) for p in sample_images],\n"
            "    conf=0.25,\n"
            "    imgsz=IMGSZ,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name='rtdetr_predict_samples',\n"
            "    exist_ok=True,\n"
            "    save=True,\n"
            ")\n\n"
            "package_dir = WORKING_DIR / 'phone_rtdetr_export'\n"
            "if package_dir.exists():\n"
            "    shutil.rmtree(package_dir)\n"
            "package_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "shutil.copy2(best_pt, package_dir / 'phone_rtdetr_l_best.pt')\n"
            "if last_pt.exists():\n"
            "    shutil.copy2(last_pt, package_dir / 'phone_rtdetr_l_last.pt')\n"
            "shutil.copy2(work_yaml, package_dir / 'mobilephone_rtdetr_data.yaml')\n\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, package_dir / filename)\n\n"
            "zip_base = WORKING_DIR / 'phone_rtdetr_training_results'\n"
            "zip_path = shutil.make_archive(str(zip_base), 'zip', package_dir)\n"
            "print('Export folder:', package_dir)\n"
            "print('Download zip:', zip_path)\n"
        ),
        md(
            "## 6. Use the trained model in the local project\n\n"
            "After downloading `phone_rtdetr_training_results.zip`, extract `phone_rtdetr_l_best.pt` and copy it to:\n\n"
            "`D:\\Work\\HocKy3\\ComputerVision\\MDDAD\\models\\phone_rtdetr_l.pt`\n\n"
            "Then test it on `/phone-test` by selecting the RT-DETR fine-tuned model or by entering the custom model path."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(output_path: Path) -> None:
    output_path.write_text(
        "# Kaggle RT-DETR Phone Training\n\n"
        "Use this package to fine-tune RT-DETR-L for phone detection.\n\n"
        "## Files\n\n"
        "- `phone_detection_rtdetr_kaggle_train.ipynb`: Kaggle notebook.\n"
        "- `dataset/kaggle_upload/mobilephone_wusj2_v7_rtdetr_kaggle.zip`: dataset upload file.\n\n"
        "## Kaggle steps\n\n"
        "1. Upload `mobilephone_wusj2_v7_rtdetr_kaggle.zip` as a Kaggle Dataset.\n"
        "2. Create a Kaggle Notebook and import `phone_detection_rtdetr_kaggle_train.ipynb`.\n"
        "3. Add the uploaded dataset to the notebook input.\n"
        "4. Set Accelerator to `GPU T4 x2` and enable Internet for first install.\n"
        "5. Run all cells.\n"
        "6. Download `/kaggle/working/phone_rtdetr_training_results.zip`.\n"
        "7. Copy `phone_rtdetr_l_best.pt` to `models/phone_rtdetr_l.pt` in the local project.\n\n"
        "## Recommended model\n\n"
        "`rtdetr-l.pt` is used because it balances real-time inference and detection quality better than `rtdetr-x.pt` for this project.\n\n",
        encoding="utf-8",
    )


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


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
