from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

SCRATCH_FILES = {
    "debug.txt",
    "grammar_issues.txt",
    "replacement_log.txt",
    "replacement_wording_log.txt",
}

OVERLEAF_SCRATCH_PATTERNS = [
    "docs/overleaf_report_package/main_all*.txt",
    "docs/overleaf_report_package/main_p2.txt",
    "docs/overleaf_report_package/page2.txt",
    "docs/overleaf_report_package/pages*.txt",
    "docs/overleaf_report_package/temp_*.txt",
]


def is_inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def collect_candidates() -> list[Path]:
    candidates: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if path.is_dir() and path.name in GENERATED_DIR_NAMES:
            candidates.append(path)

    for name in SCRATCH_FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            candidates.append(path)

    for pattern in OVERLEAF_SCRATCH_PATTERNS:
        candidates.extend(PROJECT_ROOT.glob(pattern))

    unique: dict[str, Path] = {}
    for path in candidates:
        if is_inside_project(path):
            unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda p: str(p).lower())


def remove_path(path: Path) -> None:
    if not is_inside_project(path):
        raise RuntimeError(f"Refusing to remove path outside project: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview or remove generated handoff cleanup files.")
    parser.add_argument("--execute", action="store_true", help="Actually remove generated files.")
    args = parser.parse_args()

    candidates = collect_candidates()
    action = "Removing" if args.execute else "Would remove"

    if not candidates:
        print("No generated cleanup candidates found.")
        return

    for path in candidates:
        rel = path.relative_to(PROJECT_ROOT)
        print(f"{action}: {rel}")
        if args.execute:
            remove_path(path)

    if not args.execute:
        print("\nDry run only. Re-run with --execute to remove these generated files.")


if __name__ == "__main__":
    main()
