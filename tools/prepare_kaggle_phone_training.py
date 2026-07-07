from __future__ import annotations

import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = PROJECT_ROOT / "dataset" / "roboflow_mobilephone_wusj2_v7" / "extracted"
KAGGLE_DIR = PROJECT_ROOT / "kaggle"
KAGGLE_DATASET_DIR = PROJECT_ROOT / "dataset" / "kaggle_upload"
KAGGLE_DATASET_ZIP = KAGGLE_DATASET_DIR / "mobilephone_wusj2_v7_yolo11_kaggle.zip"
NOTEBOOK_PATH = KAGGLE_DIR / "phone_detection_yolo11_kaggle_train.ipynb"


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

    write_notebook(NOTEBOOK_PATH)
    write_dataset_zip(KAGGLE_DATASET_ZIP)

    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Kaggle dataset zip: {KAGGLE_DATASET_ZIP}")
    return 0


def write_dataset_zip(output_path: Path) -> None:
    allowed_roots = {"train", "valid", "test"}
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
            "# Phone Detection Training - YOLO11 on Kaggle T4x2\n\n"
            "Notebook này dùng để train model phát hiện điện thoại cho dự án **AI Driver Safety Assessment System**.\n\n"
            "**Thiết lập Kaggle khuyến nghị:**\n"
            "- Accelerator: GPU T4 x2\n"
            "- Internet: bật ở lần chạy đầu để cài `ultralytics` nếu Kaggle image chưa có\n"
            "- Dataset: upload file `mobilephone_wusj2_v7_yolo11_kaggle.zip` dưới dạng Kaggle Dataset\n"
            "- Output chính: `best.pt`, `last.pt`, metrics và file zip tổng hợp để tải về\n"
        ),
        code(
            "import os\n"
            "import sys\n"
            "import shutil\n"
            "import subprocess\n"
            "import zipfile\n"
            "from pathlib import Path\n\n"
            "def ensure_package(package_name, import_name=None):\n"
            "    import importlib.util\n"
            "    import_name = import_name or package_name\n"
            "    if importlib.util.find_spec(import_name) is None:\n"
            "        print(f'Installing {package_name}...')\n"
            "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])\n"
            "    else:\n"
            "        print(f'{package_name} is already available')\n\n"
            "ensure_package('ultralytics')\n\n"
            "import torch\n"
            "import yaml\n"
            "from ultralytics import YOLO\n\n"
            "print('Python:', sys.version)\n"
            "print('Torch:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('GPU count:', torch.cuda.device_count())\n"
            "for i in range(torch.cuda.device_count()):\n"
            "    print(f'GPU {i}:', torch.cuda.get_device_name(i))\n"
        ),
        md(
            "## 1. Tìm dataset và chuẩn hóa `data.yaml`\n\n"
            "Cell này tự tìm `data.yaml` trong `/kaggle/input`, sửa lại đường dẫn train/valid/test nếu cần, "
            "và ghi bản YAML làm việc vào `/kaggle/working/data.yaml`."
        ),
        code(
            "INPUT_ROOT = Path('/kaggle/input')\n"
            "WORKING_DIR = Path('/kaggle/working')\n"
            "RUNS_DIR = WORKING_DIR / 'runs'\n"
            "RUN_NAME = 'phone_yolo11_t4x2'\n\n"
            "yaml_candidates = sorted(INPUT_ROOT.rglob('data.yaml'))\n"
            "if not yaml_candidates:\n"
            "    zip_candidates = sorted(INPUT_ROOT.rglob('*.zip'))\n"
            "    if zip_candidates:\n"
            "        extract_root = WORKING_DIR / 'extracted_dataset'\n"
            "        if extract_root.exists():\n"
            "            shutil.rmtree(extract_root)\n"
            "        extract_root.mkdir(parents=True, exist_ok=True)\n"
            "        print('No extracted data.yaml found. Extracting zip:', zip_candidates[0])\n"
            "        with zipfile.ZipFile(zip_candidates[0]) as zf:\n"
            "            zf.extractall(extract_root)\n"
            "        yaml_candidates = sorted(extract_root.rglob('data.yaml'))\n"
            "if not yaml_candidates:\n"
            "    raise FileNotFoundError('Không tìm thấy data.yaml trong /kaggle/input hoặc trong file .zip dataset.')\n\n"
            "source_yaml = yaml_candidates[0]\n"
            "dataset_root = source_yaml.parent\n"
            "print('Using dataset YAML:', source_yaml)\n"
            "print('Dataset root:', dataset_root)\n\n"
            "with open(source_yaml, 'r', encoding='utf-8') as f:\n"
            "    data_cfg = yaml.safe_load(f)\n\n"
            "data_cfg['path'] = str(dataset_root)\n"
            "data_cfg['train'] = 'train/images'\n"
            "data_cfg['val'] = 'valid/images'\n"
            "data_cfg['test'] = 'test/images'\n"
            "data_cfg['nc'] = 1\n"
            "data_cfg['names'] = ['Mobile-phone']\n\n"
            "work_yaml = WORKING_DIR / 'mobilephone_data.yaml'\n"
            "with open(work_yaml, 'w', encoding='utf-8') as f:\n"
            "    yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)\n\n"
            "print(work_yaml.read_text())\n"
        ),
        md("## 2. Kiểm tra dataset trước khi train"),
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
            "    bad = 0\n"
            "    for label in labels:\n"
            "        text = label.read_text(encoding='utf-8', errors='replace').strip()\n"
            "        for line in text.splitlines():\n"
            "            parts = line.split()\n"
            "            if len(parts) < 5:\n"
            "                bad += 1\n"
            "                continue\n"
            "            boxes += 1\n"
            "            try:\n"
            "                vals = list(map(float, parts[1:5]))\n"
            "                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:\n"
            "                    bad += 1\n"
            "            except Exception:\n"
            "                bad += 1\n"
            "    return {\n"
            "        'images': len(images),\n"
            "        'labels': len(labels),\n"
            "        'missing_labels': len(image_stems - label_stems),\n"
            "        'extra_labels': len(label_stems - image_stems),\n"
            "        'boxes': boxes,\n"
            "        'bad_label_lines': bad,\n"
            "    }\n\n"
            "stats = {split: count_split(split) for split in ['train', 'valid', 'test']}\n"
            "stats\n"
        ),
        code(
            "for split, row in stats.items():\n"
            "    assert row['images'] > 0, f'{split} không có ảnh'\n"
            "    assert row['labels'] > 0, f'{split} không có label'\n"
            "    assert row['missing_labels'] == 0, f'{split} thiếu label cho ảnh'\n"
            "    assert row['bad_label_lines'] == 0, f'{split} có dòng label lỗi'\n"
            "print('Dataset sanity check passed.')\n"
        ),
        md(
            "## 3. Train YOLO11\n\n"
            "Mặc định dùng `yolo11n.pt` để train nhanh và ổn định. Nếu muốn mạnh hơn, đổi `MODEL_NAME` thành "
            "`yolo11s.pt` hoặc `yolo11m.pt`, nhưng thời gian train và VRAM sẽ tăng."
        ),
        code(
            "MODEL_NAME = 'yolo11n.pt'\n"
            "EPOCHS = 80\n"
            "IMGSZ = 640\n"
            "BATCH = 32\n"
            "PATIENCE = 20\n\n"
            "if torch.cuda.device_count() >= 2:\n"
            "    DEVICE = '0,1'\n"
            "elif torch.cuda.device_count() == 1:\n"
            "    DEVICE = 0\n"
            "else:\n"
            "    DEVICE = 'cpu'\n\n"
            "print('Training device:', DEVICE)\n"
            "\n"
            "def run_train(device, batch):\n"
            "    model = YOLO(MODEL_NAME)\n"
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
            "        optimizer='auto',\n"
            "        cos_lr=True,\n"
            "        amp=True,\n"
            "        cache=False,\n"
            "        plots=True,\n"
            "    )\n"
            "\n"
            "try:\n"
            "    results = run_train(DEVICE, BATCH)\n"
            "except Exception as exc:\n"
            "    if DEVICE == '0,1':\n"
            "        print('Multi-GPU training failed, falling back to single GPU. Error:', exc)\n"
            "        DEVICE = 0\n"
            "        BATCH = min(BATCH, 16)\n"
            "        results = run_train(DEVICE, BATCH)\n"
            "    else:\n"
            "        raise\n"
        ),
        md("## 4. Validate trên tập test"),
        code(
            "best_pt = RUNS_DIR / 'detect' / RUN_NAME / 'weights' / 'best.pt'\n"
            "last_pt = RUNS_DIR / 'detect' / RUN_NAME / 'weights' / 'last.pt'\n"
            "assert best_pt.exists(), f'Không tìm thấy best.pt tại {best_pt}'\n\n"
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
            "print(metrics)\n"
        ),
        md("## 5. Chạy thử predict và đóng gói output"),
        code(
            "predict_dir = RUNS_DIR / 'predict_samples'\n"
            "sample_images = sorted((dataset_root / 'test' / 'images').glob('*'))[:24]\n"
            "assert sample_images, 'Không tìm thấy ảnh test để predict mẫu'\n\n"
            "best_model.predict(\n"
            "    source=[str(p) for p in sample_images],\n"
            "    conf=0.25,\n"
            "    imgsz=IMGSZ,\n"
            "    device=DEVICE,\n"
            "    project=str(RUNS_DIR),\n"
            "    name='predict_samples',\n"
            "    exist_ok=True,\n"
            "    save=True,\n"
            ")\n\n"
            "package_dir = WORKING_DIR / 'phone_detection_export'\n"
            "package_dir.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(best_pt, package_dir / 'phone_yolo11_best.pt')\n"
            "if last_pt.exists():\n"
            "    shutil.copy2(last_pt, package_dir / 'phone_yolo11_last.pt')\n"
            "shutil.copy2(work_yaml, package_dir / 'mobilephone_data.yaml')\n\n"
            "run_dir = RUNS_DIR / 'detect' / RUN_NAME\n"
            "for filename in ['results.csv', 'results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:\n"
            "    src = run_dir / filename\n"
            "    if src.exists():\n"
            "        shutil.copy2(src, package_dir / filename)\n\n"
            "zip_base = WORKING_DIR / 'phone_detection_training_results'\n"
            "zip_path = shutil.make_archive(str(zip_base), 'zip', package_dir)\n"
            "print('Export folder:', package_dir)\n"
            "print('Download zip:', zip_path)\n"
        ),
        md(
            "## 6. Cách dùng model trong dự án\n\n"
            "Sau khi tải `phone_detection_training_results.zip`, lấy file `phone_yolo11_best.pt` và đặt vào:\n\n"
            "`D:\\\\Work\\\\HocKy3\\\\ComputerVision\\\\MDDAD\\\\models\\\\phone_yolo11n.pt`\n\n"
            "Sau đó vào trang `/phone-test`, chọn **YOLO fine-tuned** hoặc nhập custom model path để test bounding box."
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
