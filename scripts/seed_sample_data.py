"""Utility to point the app at an extracted dataset directory."""
from pathlib import Path
import shutil

SOURCE_DIR = Path("data/dataset_extracted")
TARGET_DIR = Path("data/docai_medical_records/data/sample_dataset")


def main() -> None:
    """Copy a small deterministic sample into the project for demos."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    sample_files = sorted(SOURCE_DIR.glob("*.png"))[:50]
    for source_path in sample_files:
        shutil.copy2(source_path, TARGET_DIR / source_path.name)
    print(f"Copied {len(sample_files)} files to {TARGET_DIR}")


if __name__ == "__main__":
    main()
