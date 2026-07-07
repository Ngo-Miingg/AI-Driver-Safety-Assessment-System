from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_WORKSPACE = "m17865515473-163-com"
DEFAULT_PROJECT = "mobilephone-wusj2"
DEFAULT_VERSION = 7
DEFAULT_FORMAT = "yolov11"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Roboflow Universe MobilePhone dataset for phone detection."
    )
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key. Prefer ROBOFLOW_API_KEY env var.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="Roboflow export format, for example yolov11, yolov8, yolov5pytorch.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "dataset" / "roboflow_mobilephone_wusj2_v7"),
        help="Directory where the dataset zip and extracted files will be saved.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only download the zip; do not extract it.")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing Roboflow API key.", file=sys.stderr)
        print("Set it first in PowerShell:", file=sys.stderr)
        print("  $env:ROBOFLOW_API_KEY='YOUR_KEY_HERE'", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.project}_v{args.version}_{args.format}.zip"

    api_url = (
        f"https://api.roboflow.com/{args.workspace}/{args.project}/{args.version}/{args.format}"
        f"?api_key={args.api_key}"
    )
    print(f"Requesting export link: {args.workspace}/{args.project}/{args.version}/{args.format}")

    try:
        response = _read_json(api_url)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Roboflow API failed: HTTP {exc.code}", file=sys.stderr)
        print(body[:1000], file=sys.stderr)
        print("Known public versions for this Universe project: 7, 5, 2, 1.", file=sys.stderr)
        print("Example: python tools\\download_roboflow_mobilephone.py --version 7", file=sys.stderr)
        if args.format == "yolov11":
            print("Tip: if this export format is unavailable, try --format yolov8 or --format yolov5pytorch.", file=sys.stderr)
        return 1

    download_url = response.get("export", {}).get("link") or response.get("download") or response.get("link")
    if not download_url:
        print("Could not find a download link in Roboflow response:", file=sys.stderr)
        print(json.dumps(response, indent=2)[:2000], file=sys.stderr)
        return 1

    print(f"Downloading zip to: {zip_path}")
    _download_file(download_url, zip_path)
    print(f"Downloaded: {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.1f} MB)")

    if not args.no_extract:
        extract_dir = out_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to: {extract_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print("Done.")
        print(f"Dataset folder: {extract_dir}")

    return 0


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def _download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:
        with output_path.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
