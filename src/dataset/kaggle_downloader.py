"""
kaggle_downloader.py
--------------------
Downloads KITTI left color images from Kaggle directly into data/clean_base/.
No manual download needed.

Kaggle dataset used: klemenko/kitti-dataset
Contains: KITTI 2012/2015 stereo images — left color camera frames
Size: ~2GB (only images, no LiDAR or labels downloaded)

Prerequisites:
    1. pip install kaggle
    2. Place kaggle.json in:
       Windows → C:/Users/YOUR_NAME/.kaggle/kaggle.json
       Linux   → ~/.kaggle/kaggle.json
    Get kaggle.json from: kaggle.com → Profile → Settings → API → Create New Token

Usage:
    python src/dataset/kaggle_downloader.py
    OR:
    python run.py --mode download
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger("kaggle_downloader")

# ── Kaggle dataset identifiers ─────────────────────────────────────────────
# Primary:  klemenko/kitti-dataset  (KITTI 2012/2015 stereo, ~2GB images only)
# Fallback: garymk/kitti-3d-object-detection-dataset (full 3D detection set)
PRIMARY_DATASET  = "klemenko/kitti-dataset"
FALLBACK_DATASET = "garymk/kitti-3d-object-detection-dataset"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def check_kaggle_credentials():
    """Check that kaggle.json exists before attempting download."""
    import os
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        # Also check Windows env variable path
        env_path = os.environ.get("KAGGLE_CONFIG_DIR")
        if env_path and (Path(env_path) / "kaggle.json").exists():
            return True
        raise FileNotFoundError(
            "\n\nkaggle.json not found!\n"
            "Steps to fix:\n"
            "  1. Go to https://www.kaggle.com\n"
            "  2. Profile (top right) → Settings → API → Create New Token\n"
            "  3. Move the downloaded kaggle.json to:\n"
            f"     {kaggle_json}\n"
            "  4. Re-run this script.\n"
        )
    return True


def download_kitti_from_kaggle(output_dir: str = "data/clean_base", max_images: int = 7500):
    """
    Download KITTI images from Kaggle and extract them into output_dir.

    Args:
        output_dir  : destination folder for extracted images
        max_images  : stop copying after this many images (saves disk space)
    """
    check_kaggle_credentials()

    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "kaggle package not installed.\n"
            "Run: pip install kaggle"
        )

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = "data/_kaggle_temp"
    os.makedirs(temp_dir, exist_ok=True)

    # ── Download ───────────────────────────────────────────────────────────
    logger.info(f"Downloading {PRIMARY_DATASET} from Kaggle...")
    logger.info("This is ~2GB — expected time: 5–20 min depending on connection.")
    logger.info("Progress bar will appear below:\n")

    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            PRIMARY_DATASET,
            path    = temp_dir,
            unzip   = False,  # We'll extract selectively
            quiet   = False,
        )
        dataset_used = PRIMARY_DATASET
    except Exception as e:
        logger.warning(f"Primary dataset failed: {e}")
        logger.info(f"Trying fallback dataset: {FALLBACK_DATASET}")
        try:
            kaggle.api.dataset_download_files(
                FALLBACK_DATASET,
                path    = temp_dir,
                unzip   = False,
                quiet   = False,
            )
            dataset_used = FALLBACK_DATASET
        except Exception as e2:
            raise RuntimeError(
                f"Both datasets failed to download.\n"
                f"Primary error:  {e}\n"
                f"Fallback error: {e2}\n\n"
                "Possible causes:\n"
                "  1. kaggle.json is wrong or expired → regenerate it\n"
                "  2. You haven't accepted the dataset terms → open the\n"
                f"     dataset page on Kaggle and click 'Download' once manually:\n"
                f"     https://www.kaggle.com/datasets/{PRIMARY_DATASET}\n"
            )

    logger.info(f"\nDownload complete. Dataset used: {dataset_used}")

    # ── Find the zip file ──────────────────────────────────────────────────
    zip_files = list(Path(temp_dir).glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {temp_dir} after download.")

    zip_path = zip_files[0]
    logger.info(f"Zip file: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    # ── Extract images only ────────────────────────────────────────────────
    logger.info(f"\nExtracting images to {output_dir}/")
    logger.info(f"Stopping after {max_images} images to save disk space.")

    count = 0
    skipped_non_image = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()
        image_members = [
            m for m in all_members
            if Path(m).suffix.lower() in IMAGE_EXTENSIONS
            and not Path(m).name.startswith(".")
        ]

        logger.info(f"Total images in zip: {len(image_members)}")

        for member in image_members:
            if count >= max_images:
                logger.info(f"Reached max_images={max_images}, stopping extraction.")
                break

            # Flatten directory structure — save all images in output_dir directly
            filename = Path(member).name
            # Avoid collisions by prefixing with parent folder name
            parent   = Path(member).parent.name
            dest_name = f"{parent}_{filename}" if parent not in ("", ".") else filename
            dest_path = os.path.join(output_dir, dest_name)

            if os.path.exists(dest_path):
                count += 1
                continue

            with zf.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            count += 1

            if count % 500 == 0:
                logger.info(f"  Extracted {count}/{min(max_images, len(image_members))} images...")

    # ── Clean up zip (saves disk space) ───────────────────────────────────
    logger.info(f"\nCleaning up temp files...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────────────
    final_count = len([
        f for f in os.listdir(output_dir)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])

    logger.info("\n" + "="*55)
    logger.info("KITTI DOWNLOAD COMPLETE")
    logger.info("="*55)
    logger.info(f"  Images in {output_dir}: {final_count}")
    logger.info(f"  Ready for: python run.py --mode generate")
    logger.info("="*55)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--max_images", default=None, type=int,
                        help="Override max_images from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_img = args.max_images or cfg["dataset"]["max_images"]
    out_dir = cfg["paths"]["clean_base"]

    download_kitti_from_kaggle(output_dir=out_dir, max_images=max_img)
