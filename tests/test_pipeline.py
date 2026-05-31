"""
test_pipeline.py
----------------
Sanity checks to run BEFORE starting full dataset generation or training.

Checks:
  1. Config loads correctly
  2. Clean base images exist
  3. Stripe generator works on a dummy image
  4. Model forward pass works
  5. Ensemble forward pass works
  6. DataLoader works (if dataset already generated)

Run:
    python tests/test_pipeline.py
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "✓"
FAIL = "✗"


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"      Error: {e}")
        return False


def run_all_checks():
    print("\n" + "="*55)
    print("PIPELINE SANITY CHECKS")
    print("="*55)

    results = []

    # 1. Config
    def check_config():
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "paths" in cfg
        assert "dataset" in cfg
        assert "training" in cfg
        assert "model" in cfg
        assert len(cfg["dataset"]["variations"]) == 6, \
            f"Expected 6 variations, got {len(cfg['dataset']['variations'])}"
    results.append(check("Config loads with 6 variations", check_config))

    # 2. Clean base images
    def check_images():
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        clean_base = cfg["paths"]["clean_base"]
        assert os.path.exists(clean_base), \
            f"'{clean_base}' folder does not exist. Create it and add KITTI images."
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = [f for f in os.listdir(clean_base) if Path(f).suffix.lower() in exts]
        assert len(imgs) > 0, \
            f"No images found in '{clean_base}'. Add KITTI images there."
        print(f"      Found {len(imgs)} images in {clean_base}")
    results.append(check("Clean base images exist", check_images))

    # 3. Stripe generator
    def check_stripe_gen():
        from src.dataset.stripe_generator import inject_stripes
        dummy = np.zeros((375, 1242, 3), dtype=np.uint8) + 128
        cfg_test = {
            "freq_range": [6, 8], "width_range": [1, 2],
            "intensity_range": [160, 190], "angle": 0,
            "coverage": 0.85, "noise_sigma": 6,
        }
        result = inject_stripes(dummy, cfg_test, seed=42)
        assert result.shape == dummy.shape
        assert result.dtype == np.uint8
        # Check stripes were actually injected (result should differ from input)
        assert not np.array_equal(result, dummy)
    results.append(check("Stripe generator injects stripes correctly", check_stripe_gen))

    # 4. Angled stripe generator
    def check_angle():
        from src.dataset.stripe_generator import inject_stripes
        dummy = np.zeros((375, 1242, 3), dtype=np.uint8) + 128
        cfg_test = {
            "freq_range": [6, 10], "width_range": [2, 3],
            "intensity_range": [160, 190], "angle_range": [15, 25],
            "coverage": 0.8, "noise_sigma": 7,
        }
        result = inject_stripes(dummy, cfg_test, seed=99)
        assert result.shape == dummy.shape
    results.append(check("Angled stripe variation works", check_angle))

    # 5. Model forward pass
    def check_model():
        import torch, yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        from src.models.cnn import build_model
        model = build_model(cfg)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
    results.append(check("Single CNN forward pass", check_model))

    # 6. Ensemble forward pass
    def check_ensemble():
        import torch, yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        from src.models.ensemble import build_ensemble
        from src.dataset.dataloader import get_variation_names
        names = get_variation_names(cfg)
        ensemble = build_ensemble(cfg, names)
        dummy = torch.randn(2, 3, 224, 224)
        probs = ensemble.predict_proba(dummy)
        assert probs.shape == (2, 2), f"Expected (2,2), got {probs.shape}"
        assert abs(probs.sum(dim=1).mean().item() - 1.0) < 0.01, "Probs should sum to 1"
    results.append(check("Ensemble (6 CNNs) forward pass", check_ensemble))

    # 7. DataLoader (only if dataset exists)
    def check_dataloader():
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        csv_path = os.path.join(cfg["paths"]["final_dataset"], "labels.csv")
        if not os.path.exists(csv_path):
            print("      (Skipped — dataset not generated yet, this is fine)")
            return
        from src.dataset.dataloader import get_dataloaders
        loaders = get_dataloaders(cfg)
        for split in ["train", "val", "test"]:
            assert split in loaders
            assert len(loaders[split].dataset) > 0
    results.append(check("DataLoader (if dataset exists)", check_dataloader))

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("\n" + "="*55)
    print(f"Results: {n_pass}/{len(results)} passed")
    if n_fail > 0:
        print(f"  {n_fail} check(s) failed — fix errors above before proceeding.")
    else:
        print("  All checks passed — safe to run dataset generation.")
    print("="*55 + "\n")
    return n_fail == 0


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
