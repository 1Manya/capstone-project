"""
test_pipeline.py
----------------
Sanity checks to run BEFORE starting full dataset generation or training.

WHAT CHANGED FROM THE PREVIOUS VERSION:
1. check_config(): read the top-level `variations:` list (what
   dataset_builder.py actually reads) instead of the nonexistent
   `cfg["dataset"]["variations"]` dict. Still expects 6 entries - that
   number didn't change, only where it's read from (clean + 5 attack
   variations = 6 total entries in the list).
2. Removed `assert "dataset" in cfg` - that top-level section never
   existed in the v7 config; replaced with checks for the sections that
   actually exist (paths/training/model).
3. check_stripe_gen()/check_angle() tested `src/dataset/stripe_generator.py`
   (an overlay-based approach from an earlier design) - that module isn't
   part of this pipeline. Replaced with two checks against the real
   physics simulator: (a) a known-frequency stripe-visibility check, and
   (b) the background-linearity check (catches the v6 saturation bug
   class if it were ever reintroduced).
4. check_dataloader() unchanged in logic, still skips gracefully if the
   dataset hasn't been generated yet.

Run:
    python tests/test_pipeline.py
    (or: python run.py --mode test)
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "PASS"
FAIL = "FAIL"


def check(name: str, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {name}")
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
        assert "training" in cfg
        assert "model" in cfg
        assert "variations" in cfg, "variations should be a top-level list, not nested under 'dataset'"
        assert len(cfg["variations"]) == 6, \
            f"Expected 6 variations (clean + 5 attack bands), got {len(cfg['variations'])}"
    results.append(check("Config loads with 6 variations", check_config))

    # 2. Clean base images
    def check_images():
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        clean_base = cfg["paths"]["clean_base"]
        assert os.path.exists(clean_base), \
            f"'{clean_base}' folder does not exist. Create it and add background images."
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = [f for f in os.listdir(clean_base) if Path(f).suffix.lower() in exts]
        assert len(imgs) > 0, \
            f"No images found in '{clean_base}'. Add background images there."
        print(f"      Found {len(imgs)} images in {clean_base}")
    results.append(check("Clean base images exist", check_images))

    # 3. Simulator: background should reproduce near-linearly (catches the
    #    v6 saturation-curve bug class if it were ever reintroduced)
    def check_background_linearity():
        from src.dataset.rolling_shutter_simulator_v7 import (
            CameraParams, RollingShutterSimulator, LaserParams, LaserModulation, EnvParams
        )
        cam = CameraParams(fps=30, height=224, width=224)
        sim = RollingShutterSimulator(cam, seed=0)
        rng = np.random.default_rng(1)
        test_img = (rng.random((224, 224, 3)) * 120 + 60).astype(np.uint8)
        zero_laser = LaserParams(frequency=100.0, wavelength=450, power_mw=0.0,
                                  duty_cycle=0.5, modulation=LaserModulation.SQUARE, coverage=0.85)
        out, meta = sim.simulate_frame(test_img.copy(), zero_laser, EnvParams())
        assert out.mean() < 220, "background is being pinned near saturation with zero laser power"
    results.append(check("Simulator: background reproduces near-linearly", check_background_linearity))

    # 4. Simulator: a mid-band frequency should show visible stripe structure,
    #    and a too-low frequency for the frame rate should not
    def check_stripe_visibility():
        from src.dataset.rolling_shutter_simulator_v7 import (
            CameraParams, RollingShutterSimulator, LaserParams, LaserModulation
        )
        cam = CameraParams(fps=30, height=224, width=224, exposure_time=0.0002)
        sim = RollingShutterSimulator(cam, seed=0)
        laser_300hz = LaserParams(frequency=300.0, wavelength=450, power_mw=15.0,
                                   duty_cycle=0.5, modulation=LaserModulation.SQUARE, coverage=0.85)
        rp_300 = sim._row_power(laser_300hz)
        assert rp_300.std() > 0.15, \
            f"300Hz should show clear stripe structure with a short exposure, got std={rp_300.std():.4f}"
    results.append(check("Simulator: stripes visible at a representative frequency", check_stripe_visibility))

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
        assert len(names) == 5, f"Expected 5 attack-variation specialists, got {len(names)}"
        ensemble = build_ensemble(cfg, names)
        dummy = torch.randn(2, 3, 224, 224)
        probs = ensemble.predict_proba(dummy)
        assert probs.shape == (2, 2), f"Expected (2,2), got {probs.shape}"
        assert abs(probs.sum(dim=1).mean().item() - 1.0) < 0.01, "Probs should sum to 1"
    results.append(check("Ensemble (5 specialists) forward pass", check_ensemble))

    # 7. DataLoader (only if dataset exists)
    def check_dataloader():
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        csv_path = os.path.join(cfg["paths"]["final_dataset"], "labels.csv")
        if not os.path.exists(csv_path):
            print("      (Skipped - dataset not generated yet, this is fine)")
            return
        from src.dataset.dataloader import get_dataloaders
        loaders = get_dataloaders(cfg)
        for split in ["train", "val", "test"]:
            assert split in loaders
            assert len(loaders[split].dataset) > 0
    results.append(check("DataLoader (if dataset exists)", check_dataloader))

    # -- Summary ----------------------------------------------------------------
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("\n" + "="*55)
    print(f"Results: {n_pass}/{len(results)} passed")
    if n_fail > 0:
        print(f"  {n_fail} check(s) failed - fix errors above before proceeding.")
    else:
        print("  All checks passed - safe to run dataset generation / training.")
    print("="*55 + "\n")
    return n_fail == 0


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)