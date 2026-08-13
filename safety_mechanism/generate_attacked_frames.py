"""
generate_attacked_frames.py
=============================
Takes the clean frames extracted by extract_frames.py and runs your REAL
rolling-shutter simulator on each one, producing matching attacked versions.
This is what makes the dashboard's "attacked" panel show genuinely
stripe-corrupted images, instead of the clean image always being displayed
regardless of the attack toggle.

Pre-generates and saves to disk (rather than running the simulator live
inside Streamlit) so the live demo is fast and can't hang/lag mid-presentation.

Requires rolling_shutter_simulator_v7.py to be importable (same folder, or
on your PYTHONPATH).

Usage:
    python3 generate_attacked_frames.py --input frames --output frames_attacked \
        --variation freq_high_fine
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from rolling_shutter_simulator_v7 import (
    AEConfig, CameraParams, DomainRandomConfig, EnvParams,
    LaserModulation, LaserParams, RollingShutterSimulator,
)

MODULATION_MAP = {m.value: m for m in LaserModulation}

# Same variation profiles as your dataset_builder.py config, so the demo
# frames are generated with realistic, consistent parameters - not arbitrary
# one-off values.
VARIATION_PROFILES = {
    "freq_low_wide": dict(frequency_range=(60, 200), wavelength_choices=[405, 450, 532, 650],
                           power_range=(10, 35), duty_cycle_range=(0.3, 0.7),
                           modulation_choices=["square", "sine", "triangle"],
                           coverage_range=(0.4, 0.95), angle_range=(-15, 15), distance_range=(5, 30),
                           ellipticity_range=(0.5, 1.0)),
    "freq_high_fine": dict(frequency_range=(800, 2000), wavelength_choices=[450, 532, 650, 808, 1064],
                            power_range=(10, 45), duty_cycle_range=(0.2, 0.8),
                            modulation_choices=["square", "sine", "pulse"],
                            coverage_range=(0.25, 0.85), angle_range=(-20, 20), distance_range=(5, 45),
                            ellipticity_range=(0.4, 1.0)),
    "freq_ultra_aliasing": dict(frequency_range=(2000, 5000), wavelength_choices=[450, 532, 650, 808],
                                 power_range=(15, 50), duty_cycle_range=(0.3, 0.7),
                                 modulation_choices=["square", "pulse", "flicker"],
                                 coverage_range=(0.3, 0.9), angle_range=(-20, 20), distance_range=(5, 45),
                                 ellipticity_range=(0.4, 1.0)),
}


def build_laser(rng: random.Random, profile: dict) -> LaserParams:
    freq = rng.uniform(*profile["frequency_range"])
    return LaserParams(
        frequency=freq,
        wavelength=rng.choice(profile["wavelength_choices"]),
        power_mw=rng.uniform(*profile["power_range"]),
        duty_cycle=rng.uniform(*profile["duty_cycle_range"]),
        modulation=MODULATION_MAP[rng.choice(profile["modulation_choices"])],
        phase=rng.uniform(0, 1),
        angle_deg=rng.uniform(*profile["angle_range"]),
        coverage=rng.uniform(*profile["coverage_range"]),
        distance_m=rng.uniform(*profile["distance_range"]),
        ellipticity=rng.uniform(*profile["ellipticity_range"]),
    )


def adaptive_exposure_time(rng: random.Random, frequency: float, frame_time: float,
                            fraction_range=(0.15, 0.30)) -> float:
    """Same fix as dataset_builder.py - exposure tied to THIS frame's own
    sampled frequency, so stripes stay visible instead of washing out."""
    period = 1.0 / frequency
    frac = rng.uniform(*fraction_range)
    return float(min(period * frac, frame_time * 0.9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="frames")
    ap.add_argument("--output", default="frames_attacked")
    ap.add_argument("--variation", default="freq_high_fine", choices=list(VARIATION_PROFILES.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {in_dir} - run extract_frames.py first.")
        return

    cam = CameraParams(height=args.image_size, width=args.image_size)
    sim = RollingShutterSimulator(cam, seed=args.seed, ae_cfg=AEConfig())
    dr_cfg = DomainRandomConfig()
    rng = random.Random(args.seed)
    profile = VARIATION_PROFILES[args.variation]

    print(f"Generating {len(frame_paths)} attacked frames (variation={args.variation}) ...")
    for p in frame_paths:
        bg = cv2.imread(str(p))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        if bg.shape[:2] != (args.image_size, args.image_size):
            bg = cv2.resize(bg, (args.image_size, args.image_size))

        laser = build_laser(rng, profile)
        env = EnvParams(
            haze_factor=rng.uniform(0.0, 0.12),
            lens_flare=rng.uniform(0.0, 0.2),
        )
        sim.cam.exposure_time = adaptive_exposure_time(rng, laser.frequency, cam.frame_time)

        out, meta = sim.simulate_frame(bg.copy(), laser, env)
        out = sim.domain_randomize(out, dr_cfg)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), out_bgr)

    print(f"Done. Attacked frames saved to {out_dir}/")
    print("Point demo_dashboard.py at BOTH --frames-dir and --attacked-frames-dir")


if __name__ == "__main__":
    main()
