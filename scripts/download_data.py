"""Download and extract the chess games dataset.

Fetches gt1_8kElo_all.zip (Lichess games, White ELO > 1800) from the
adamkarvonen/chess_games dataset on HuggingFace, then extracts the CSV
into data/raw/. Skips the download if the zip is already there, and 
skips extraction if the CSV is already there.
"""

import argparse
import zipfile
from pathlib import Path

import requests

# Direct download link for the dataset
DATASET_URL = "https://huggingface.co/datasets/adamkarvonen/chess_games/resolve/main/gt1_8kElo_all.zip"

# CSV/ZIP files is stored in the data/raw/ directory
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_zip(url: str, dest_path: Path) -> None:
    """Stream the zip from HuggingFace to dest_path.

    Args:
        url: Direct download URL for the zip file.
        dest_path: Where to save the downloaded zip.
    """
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    downloaded_bytes = 0

    with open(dest_path, "wb") as f:
        # Download in chunks so we don't have to hold the whole file in memory.
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded_bytes += len(chunk)
            if total_bytes:
                percent = downloaded_bytes / total_bytes * 100
                print(f"\r  {downloaded_bytes / 1e6:.0f} MB / {total_bytes / 1e6:.0f} MB ({percent:.1f}%)", end="")
    print()


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract the CSV from the downloaded zip into dest_dir.

    Args:
        zip_path: Path to the downloaded zip file.
        dest_dir: Directory to extract into.
    """
    print(f"Extracting {zip_path.name} into {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip file after extraction (default: delete it to save disk space).",
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DATA_DIR / "gt1_8kElo_all.zip"
    csv_path = RAW_DATA_DIR / "gt1_8kElo_all.csv"

    if csv_path.exists():
        print(f"{csv_path} already exists, nothing to do.")
        return

    if not zip_path.exists():
        download_zip(DATASET_URL, zip_path)
    else:
        print(f"{zip_path} already downloaded, skipping download.")

    extract_zip(zip_path, RAW_DATA_DIR)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Deleted {zip_path} (pass --keep-zip to keep it next time).")


if __name__ == "__main__":
    main()
