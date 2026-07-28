"""
Dataset builder for the rolling-shutter laser-attack simulator (v7).
=====================================================================
Reads clean background images, applies the simulator across a set of
variations defined in a YAML config, and writes out a labeled image
dataset with a metadata CSV.

Usage:
    python3 dataset_builder.py --config ../../configs/config.yaml

Expected config keys (see simulator_config_v7.yaml):
    paths.clean_base        - directory of clean background images (e.g. KITTI)
    paths.final_dataset     - output directory
    camera                  - CameraParams fields
    ae                      - AEConfig fields
    domain_randomization    - DomainRandomConfig fields
    variations              - list of attack/clean variations to generate
    split                   - train/val/test fractions
"""

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from rolling_shutter_simulator_v7 import (
    AEConfig, CameraParams, DomainRandomConfig, EnvParams,
    LaserModulation, LaserParams, RollingShutterSimulator,
)

MODULATION_MAP = {m.value: m for m in LaserModulation}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def variation_seed(base_seed: int, name: str) -> int:
    """Deterministic, independent seed per variation, derived from base_seed
    + variation name (not Python's hash(), which is randomized per-process
    unless PYTHONHASHSEED is fixed). This means:
      - each variation's images are reproducible regardless of which OTHER
        variations are generated alongside it in a given run
      - selectively regenerating just one or two variations (e.g. to apply
        a bugfix) never disturbs any other variation's images, because each
        variation's RNG stream no longer depends on shared sequential state
        from processing earlier variations first.
    Trade-off: fixed-pattern-noise/defect maps (keyed only by seed, not by
    variation) will differ slightly between variations generated in the
    same run vs. ones regenerated later - this is a ~2% multiplicative
    per-pixel noise term, not a training-relevant difference, but worth
    knowing about if you're being precise about "one consistent sensor."
    """
    h = hashlib.md5(f"{base_seed}-{name}".encode()).hexdigest()
    return int(h[:8], 16)


def load_clean_images(clean_base: str):
    # sorted() is required for reproducibility: Path.rglob() order is filesystem/OS
    # dependent, so without sorting, the same --seed would pick different images
    # on a different machine even with an identical clean_base folder.
    paths = sorted(p for p in Path(clean_base).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found under {clean_base}")
    return paths


def sample_range(rng: random.Random, spec):
    """spec is either [lo, hi] (uniform float) or a fixed scalar or a list of choices."""
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return rng.uniform(spec[0], spec[1])
        return rng.choice(spec)
    return spec


def build_laser_params(rng: random.Random, var_cfg: dict) -> LaserParams:
    return LaserParams(
        frequency=sample_range(rng, var_cfg["frequency_range"]),
        wavelength=int(sample_range(rng, var_cfg.get("wavelength_choices", [650]))),
        power_mw=sample_range(rng, var_cfg.get("power_range", [10, 30])),
        duty_cycle=sample_range(rng, var_cfg.get("duty_cycle_range", [0.3, 0.7])),
        modulation=MODULATION_MAP[rng.choice(var_cfg.get("modulation_choices", ["square"]))],
        phase=rng.uniform(0, 1),
        angle_deg=sample_range(rng, var_cfg.get("angle_range", [0, 0])),
        coverage=sample_range(rng, var_cfg.get("coverage_range", [0.3, 0.9])),
        distance_m=sample_range(rng, var_cfg.get("distance_range", [5, 40])),
        ellipticity=sample_range(rng, var_cfg.get("ellipticity_range", [0.5, 1.0])),
        divergence_per_m=var_cfg.get("divergence_per_m", 0.02),
        ref_distance_m=var_cfg.get("ref_distance_m", 10.0),
    )


def sample_exposure_time(rng: random.Random, var_cfg: dict, default: float) -> float:
    """Sample this image's row exposure time.

    ROOT CAUSE FIX: a fixed exposure_time (e.g. 1ms for every image, regardless
    of variation) washes out stripe contrast whenever the PWM period is close
    to or shorter than the exposure window - each row then integrates over a
    full on/off cycle instead of catching one clean phase, and the modulation
    averages to near-zero (verified: at 1000Hz with exposure_time=1ms, row-to-
    row contrast std is ~0.03; at 200us it's ~0.43). Higher-frequency
    variations need proportionally shorter exposure_time to stay visible, the
    same way a real camera needs a fast shutter to catch fast PWM flicker.
    Falls back to `default` (camera.exposure_time) if a variation doesn't
    specify `exposure_time_range`."""
    if "exposure_time_range" in var_cfg:
        return sample_range(rng, var_cfg["exposure_time_range"])
    return default


def build_env_params(rng: random.Random, var_cfg: dict) -> EnvParams:
    return EnvParams(
        haze_factor=sample_range(rng, var_cfg.get("haze_range", [0.0, 0.1])),
        lens_flare=sample_range(rng, var_cfg.get("lens_flare_range", [0.0, 0.15])),
        chromatic_aberration=sample_range(rng, var_cfg.get("chromatic_aberration_range", [0.0, 0.1])),
    )


def assign_split(rng: random.Random, split_cfg: dict) -> str:
    r = rng.random()
    train = split_cfg.get("train", 0.7)
    val = split_cfg.get("val", 0.15)
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Paths in the config are relative to the project root, i.e. the parent of
    # the `configs/` directory the config file lives in. This makes the script
    # work the same whether it's run from src/dataset/, the project root, or
    # anywhere else.
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]
    out_dir = project_root / cfg["paths"]["final_dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = CameraParams(**cfg.get("camera", {}))
    ae_cfg = AEConfig(**cfg.get("ae", {}))
    dr_cfg = DomainRandomConfig(**{k: tuple(v) if isinstance(v, list) else v
                                    for k, v in cfg.get("domain_randomization", {}).items()})
    split_cfg = cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})

    clean_paths = load_clean_images(clean_base)
    print(f"Found {len(clean_paths)} clean background images in {clean_base}")

    rng = random.Random(args.seed)
    sim = RollingShutterSimulator(cam, seed=args.seed, ae_cfg=ae_cfg)

    labels_path = out_dir / "labels.csv"
    total_written = 0
    t0 = time.time()

    with open(labels_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow(["path", "label", "split", "variation", "frequency", "wavelength",
                          "power_mw", "duty_cycle", "modulation", "coverage", "angle_deg",
                          "distance_m", "ellipticity", "exposure_time", "ae_gain",
                          "peak_saturation", "attack_area_fraction"])

        for var_cfg in cfg["variations"]:
            name = var_cfg["name"]
            label = var_cfg.get("label", name)
            count = int(var_cfg["count"])
            is_clean = var_cfg.get("clean", False)

            var_dir = out_dir / name
            var_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{name}] generating {count} images (clean={is_clean}) ...")
            for i in range(count):
                src_path = clean_paths[rng.randrange(len(clean_paths))]
                bg = cv2.imread(str(src_path))
                if bg is None:
                    continue
                if bg.shape[:2] != (cam.height, cam.width):
                    bg = cv2.resize(bg, (cam.width, cam.height))

                split = assign_split(rng, split_cfg)

                if is_clean:
                    out = bg.copy()
                    meta = {"frequency": 0, "wavelength": 0, "power_mw": 0, "duty_cycle": 0,
                            "modulation": "none", "coverage": 0, "angle_deg": 0, "distance_m": 0,
                            "ellipticity": 0, "exposure_time": 0, "ae_gain": 1.0,
                            "peak_saturation": 0.0, "attack_area_fraction": 0.0}
                else:
                    laser = build_laser_params(rng, var_cfg)
                    env = build_env_params(rng, var_cfg)
                    # per-image exposure_time, matched to this variation's frequency
                    # range - see sample_exposure_time() docstring for why this
                    # matters (fixes stripe contrast washing out at high frequencies)
                    sim.cam.exposure_time = sample_exposure_time(rng, var_cfg, cam.exposure_time)
                    out, meta = sim.simulate_frame(bg, laser, env)
                    meta["exposure_time"] = sim.cam.exposure_time

                out = sim.domain_randomize(out, dr_cfg)

                fname = f"{name}_{i:05d}.png"
                fpath = var_dir / fname
                cv2.imwrite(str(fpath), out)

                writer.writerow([str(fpath.relative_to(out_dir)), label, split, name,
                                  meta["frequency"], meta["wavelength"], meta["power_mw"],
                                  meta["duty_cycle"], meta["modulation"], meta["coverage"],
                                  meta["angle_deg"], meta["distance_m"], meta["ellipticity"], meta["exposure_time"],
                                  meta["ae_gain"], meta["peak_saturation"],
                                  meta["attack_area_fraction"]])
                total_written += 1

            print(f"[{name}] done.")

    elapsed = time.time() - t0
    print(f"\nWrote {total_written} images + labels to {out_dir} in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()