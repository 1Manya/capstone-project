# Capstone: Mask generation → U-Net → SSD → Fusion

This implementation follows the agreed pipeline **without a CNN**.

## Files
- `train_unet.py`: trains U-Net on images and generated masks.
- `unet_model.py`: U-Net architecture.
- `unet_features.py`: extracts per-image U-Net segmentation features.
- `ssd_features.py`: extracts object-detection features using pretrained SSD300 VGG16 COCO weights.
- `fusion_model.py`: MLP that combines U-Net and SSD features.
- `train_fusion.py`: trains the fusion classifier using only U-Net + SSD features.
- `common.py`, `config.py`: paths and dataset utilities.

## Expected Kaggle paths
- Images: `/kaggle/working/capstone-project/data/final_dataset/`
- Masks: `/kaggle/working/capstone-project/data/unet_masks_final/`

The labels CSV is searched at:
- `data/final_dataset/labels.csv`
- `data/labels.csv`

It must contain at least `path`, `split`, and `variation`, with split values `train`, `val`, and `test`.

## Run order
Run from the repository root:
```bash
python train_unet.py
python unet_features.py
python ssd_features.py
python train_fusion.py
```

## Resume behavior
- U-Net checkpoints every 100 batches and at each epoch. Rerun `train_unet.py` after interruption to resume from the last saved batch.
- U-Net feature extraction appends every 100 images and skips paths already present in its CSV.
- SSD feature extraction appends every 100 images and skips paths already present in its CSV.
- Fusion saves after every epoch and resumes from the last completed epoch.

## Important interpretation
1. Generated masks are heuristic pseudo-labels. U-Net scores against them measure agreement with pseudo-labels, not ground-truth real-laser segmentation.
2. SSD is pretrained COCO SSD300 VGG16 and is used for feature extraction, not trained from scratch.
3. Fusion receives **only U-Net and SSD features**. No CNN features or CNN prediction file are used.
4. Fusion currently predicts `variation` (clean or frequency category). This is a classification target, not a calibrated physical camera-reliability score.
5. Verify that all synthetic variants from one original source image were assigned to the same train/validation/test split to prevent source-image leakage.
6. Since clean/attacked pairing was abandoned, the SSD features here are per-image detection summaries, not paired object-level degradation measurements.
