from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working/capstone-project")
DATA_ROOT = PROJECT_ROOT / "data"
IMAGE_ROOT = DATA_ROOT / "final_dataset"
MASK_ROOT = DATA_ROOT / "unet_masks_final"
LABELS_CANDIDATES = [IMAGE_ROOT / "labels.csv", DATA_ROOT / "labels.csv"]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
UNET_DIR = OUTPUT_ROOT / "unet"
SSD_DIR = OUTPUT_ROOT / "ssd"
FUSION_DIR = OUTPUT_ROOT / "fusion"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
DEVICE = "cuda"
