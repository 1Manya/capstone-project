"""
extract_nuscenes.py
-------------------
Extracts front camera frames from nuScenes Mini dataset
and copies them into data/clean_base/ for use in stripe generation.

Requirements:
    pip install nuscenes-devkit

Usage:
    python src/utils/extract_nuscenes.py --nuscenes_root /path/to/nuscenes
    OR:
    python run.py --mode extract_nuscenes --nuscenes_root /path/to/nuscenes
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger("nuscenes_extractor")


def extract_nuscenes_frames(nuscenes_root: str, output_dir: str = "data/clean_base"):
    """
    Extract CAM_FRONT frames from nuScenes Mini.

    Args:
        nuscenes_root : path to nuScenes dataset folder
                        (should contain v1.0-mini/ subfolder)
        output_dir    : destination folder for extracted frames
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        logger.error(
            "nuscenes-devkit not installed.\n"
            "Install with: pip install nuscenes-devkit\n"
            "Then re-run this script."
        )
        return

    if not os.path.exists(nuscenes_root):
        logger.error(f"nuScenes root not found: {nuscenes_root}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Try mini first, then full
    for version in ["v1.0-mini", "v1.0-trainval"]:
        version_path = os.path.join(nuscenes_root, version)
        if os.path.exists(version_path):
            logger.info(f"Loading nuScenes {version}...")
            nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
            break
    else:
        logger.error(f"No valid nuScenes version found in {nuscenes_root}")
        return

    count = 0
    for sample in nusc.sample:
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data  = nusc.get("sample_data", cam_token)
        src_path  = os.path.join(nuscenes_root, cam_data["filename"])

        if not os.path.exists(src_path):
            continue

        # Name file with scene token to avoid collisions
        fname = f"nuscenes_{sample['token'][:8]}_front.jpg"
        dest  = os.path.join(output_dir, fname)
        shutil.copy2(src_path, dest)
        count += 1

    logger.info(f"Extracted {count} nuScenes front camera frames → {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", required=True, help="Path to nuScenes dataset root")
    parser.add_argument("--output_dir",    default="data/clean_base")
    args = parser.parse_args()
    extract_nuscenes_frames(args.nuscenes_root, args.output_dir)
