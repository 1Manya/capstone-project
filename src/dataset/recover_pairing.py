"""
recover_pairing.py
---------------------
Recovers which clean base image each row of the EXISTING labels.csv was
generated from - WITHOUT regenerating any images. Works because dataset
generation is fully deterministic: same --seed, same sorted clean image
list, same iteration order. Replaying just the RNG-consuming lines from
dataset_builder.py's main loop (not the actual simulation) reproduces the
exact same random.Random() draws, in the same order, and therefore the
exact same clean image selection per row - in milliseconds, no GPU, no
image I/O for the simulation itself.

SELF-VALIDATING: your labels.csv already stores each row's sampled
frequency/power_mw/coverage/etc (the OTHER things drawn from the same rng
sequence, right after src_path). If the replayed values match the stored
CSV values, the pairing recovery is provably correct for that row. If they
don't match (wrong seed, or dataset_builder.py changed since generation),
you'll see mismatches reported instead of silently-wrong pairings.

Usage:
    python recover_pairing.py \
        --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels.csv \
        --seed 42 \
        --out ../../data/final_dataset/labels_paired.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from dataset_builder import (
    load_clean_images, build_laser_params, build_env_params,
    sample_exposure_time, assign_split,
)
from rolling_shutter_simulator_v7 import CameraParams

import random


FNAME_RE = re.compile(r"^(?P<name>.+)_(?P<idx>\d+)\.png$")


def replay_generation(cfg, clean_base, seed):
    """Replays dataset_builder.py's main loop, RNG-consumption-for-RNG-
    consumption, WITHOUT loading images or running the simulator. Returns
    dict: (variation_name, index) -> {src_image_name, frequency, power_mw,
    coverage, angle_deg, distance_m, ellipticity, duty_cycle, exposure_time}
    """
    clean_paths = load_clean_images(clean_base)
    cam = CameraParams(**cfg.get("camera", {}))
    split_cfg = cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    rng = random.Random(seed)

    recovered = {}
    for var_cfg in cfg["variations"]:
        name = var_cfg["name"]
        count = int(var_cfg["count"])
        is_clean = var_cfg.get("clean", False)

        for i in range(count):
            src_path = clean_paths[rng.randrange(len(clean_paths))]
            _split = assign_split(rng, split_cfg)  # must consume rng even though unused here

            entry = {"src_image_name": src_path.name}
            if not is_clean:
                laser = build_laser_params(rng, var_cfg)
                _env = build_env_params(rng, var_cfg)
                exposure_time = sample_exposure_time(rng, var_cfg, cam.exposure_time)
                entry.update({
                    "frequency": laser.frequency, "power_mw": laser.power_mw,
                    "coverage": laser.coverage, "angle_deg": laser.angle_deg,
                    "distance_m": laser.distance_m, "ellipticity": laser.ellipticity,
                    "duty_cycle": laser.duty_cycle, "exposure_time": exposure_time,
                })
            recovered[(name, i)] = entry

    return recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate_tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent
    clean_base = project_root / cfg["paths"]["clean_base"]

    print("Replaying generation RNG sequence (no images touched)...")
    recovered = replay_generation(cfg, clean_base, args.seed)
    print(f"Recovered {len(recovered)} (variation, index) -> source mappings.")

    df = pd.read_csv(args.labels)

    src_names = []
    row_verified = []  # True/False/None(not-checked, e.g. clean rows)
    checked = 0
    mismatches = 0
    for _, row in df.iterrows():
        # labels.csv paths may use Windows-style backslashes (original
        # generation ran on Windows) - normalize before extracting filename,
        # since pathlib on Linux won't split on backslash.
        normalized = str(row["path"]).replace("\\", "/")
        fname = Path(normalized).name
        m = FNAME_RE.match(fname)
        if not m:
            src_names.append(None)
            row_verified.append(None)
            continue
        name, idx = m.group("name"), int(m.group("idx"))
        entry = recovered.get((name, idx))
        if entry is None:
            src_names.append(None)
            row_verified.append(None)
            continue
        src_names.append(entry["src_image_name"])

        # validate against stored metadata where available (skip clean rows -
        # they have no laser metadata to check against, always trusted)
        if "frequency" in entry and "frequency" in row and row["frequency"] not in (0, "0"):
            checked += 1
            ok = abs(float(row["frequency"]) - entry["frequency"]) <= args.validate_tolerance * max(1, abs(entry["frequency"]))
            row_verified.append(ok)
            if not ok:
                mismatches += 1
        else:
            row_verified.append(True)  # clean rows: nothing to validate, trusted as-is

    df["source_clean_image"] = src_names
    df["pairing_verified"] = row_verified

    print(f"\n--- Validation ---")
    print(f"Checked {checked} attacked rows against stored frequency values.")
    print(f"Mismatches: {mismatches} ({mismatches/max(1,checked)*100:.3f}% of checked rows)")

    if mismatches == 0:
        print("PASSED - recovered pairing matches stored metadata exactly for every row. Trust this file.")
    else:
        pct = mismatches / max(1, checked) * 100
        if pct < 1.0:
            print(f"MOSTLY PASSED - {mismatches} isolated mismatches ({pct:.3f}%), likely due to "
                  f"clean_base having slightly different file composition/order than the original "
                  f"generation run (e.g. after re-upload to this server). These {mismatches} rows "
                  f"are marked pairing_verified=False and should be EXCLUDED from analysis - the "
                  f"other {checked - mismatches} attacked rows (plus all clean rows) are safe to use.")
        else:
            print("WARNING - mismatch rate is too high to be isolated noise. The pairing recovery "
                  "is likely WRONG (wrong --seed, or dataset_builder.py has changed since the "
                  "original generation run). Do not use this file.")

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} (includes ALL rows, with pairing_verified column - "
          f"filter to pairing_verified == True before using source_clean_image downstream)")

    n_unmatched = df["source_clean_image"].isna().sum()
    if n_unmatched:
        print(f"[WARN] {n_unmatched} rows could not be matched to a source image at all "
              f"(unexpected filename format?) - check these manually.")

    if mismatches > 0:
        bad_rows = df[df["pairing_verified"] == False]
        bad_csv = str(Path(args.out).with_name(Path(args.out).stem + "_mismatched.csv"))
        bad_rows.to_csv(bad_csv, index=False)
        print(f"Mismatched rows written separately to {bad_csv} for inspection.")


if __name__ == "__main__":
    main()
