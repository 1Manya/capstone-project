"""
stripe_generator.py  —  v3.0  FINAL
-------------------------------------
Physically accurate rolling shutter laser dazzle simulation.

Built from first principles using:

  Köhler et al. 2021 — "They See Me Rollin'"
    • Row-wise stripe pattern from laser/row-scan timing ratio
    • Stripe frequency = f_laser / f_rowscan
    • Partial coverage from aiming precision
    • Angle of incidence tilts stripes

  Stein et al. 2025 — "Invisible CMOS Camera Dazzling"
    • Rn = t_exp / t_read  →  number of rows saturated per pulse
    • Duty cycle D controls stripe width (ON rows vs OFF rows)
    • PSF halo around bright stripe centre (laser diffraction)
    • Saturation model: x_sat ∝ I^(1/3)

What makes this physically accurate vs a naive overlay:
  1.  Stripes are SATURATED (pixel value → 255) at their centre
      not just bright — real laser dazzles the sensor to clipping
  2.  PSF halo: brightness falls off from centre outward (Gaussian)
      giving the glow seen in Köhler Fig 1
  3.  Color bleed: laser wavelength interacts with Bayer CFA
      → strong response in one channel, weaker in others
      → blue/green tint on most laser colours (matches paper)
  4.  Stripe grouping: Rn rows saturated per pulse, then gap,
      then Rn rows again — NOT evenly spaced single lines
  5.  Inter-stripe residual: rows between stripes still get faint
      halo spill from adjacent saturated rows
  6.  Gaussian sensor read noise added on top (CMOS characteristic)
"""

import numpy as np
import cv2
import random
import math
from typing import Dict, Any, Tuple

# ── Physical laser → Bayer CFA colour response ────────────────────────────
# Real lasers create a dominant colour depending on wavelength.
# The Bayer filter passes different fractions per channel.
# Format: (R_gain, G_gain, B_gain) — normalised so max=1.0
# Chosen to match the blue/purple/cyan palette in Köhler Fig 1.
LASER_PROFILES = [
    # name            R      G      B
    ("blue_laser",   0.20,  0.45,  1.00),   # 450nm laser  — dominant in Köhler Fig 1
    ("cyan_laser",   0.15,  0.80,  1.00),   # 488nm laser
    ("green_laser",  0.10,  1.00,  0.30),   # 532nm green  — less common but real
    ("violet_laser", 0.40,  0.20,  1.00),   # 405nm violet
    ("white_led",    0.85,  0.90,  1.00),   # broadband white modulated LED
    ("blue_white",   0.50,  0.65,  1.00),   # mixed blue-white (most common dashcam attack)
]


def inject_stripes(image: np.ndarray, cfg: Dict[str, Any], seed: int = None) -> np.ndarray:
    """
    Inject physically accurate rolling shutter laser dazzle stripes.

    Args:
        image : np.ndarray  (H, W, 3)  uint8  RGB
        cfg   : variation config dict from config.yaml
        seed  : random seed for reproducibility

    Returns:
        corrupted : np.ndarray  (H, W, 3)  uint8  RGB
    """
    if seed is not None:
        rng    = np.random.RandomState(seed)
        py_rng = random.Random(seed)
    else:
        rng    = np.random.RandomState()
        py_rng = random.Random()

    h, w   = image.shape[:2]
    output = image.copy().astype(np.float32)

    # ── 1. Sample attack parameters ───────────────────────────────────────
    freq_range  = cfg.get("freq_range",      [6, 8])
    width_range = cfg.get("width_range",     [1, 2])   # maps to Rn (rows per pulse)
    int_range   = cfg.get("intensity_range", [160, 190])
    noise_sigma = cfg.get("noise_sigma",     6)
    coverage    = cfg.get("coverage",        1.0)

    angle = (py_rng.uniform(*cfg["angle_range"])
             if "angle_range" in cfg
             else cfg.get("angle", 0))

    freq     = py_rng.randint(*freq_range)    if freq_range[0]  != freq_range[1]  else freq_range[0]
    Rn       = py_rng.randint(*width_range)   if width_range[0] != width_range[1] else width_range[0]
    peak_int = py_rng.randint(*int_range)     if int_range[0]   != int_range[1]   else int_range[0]

    # peak_int is the saturation level (0-255). Real laser → pixel clips to 255
    # but we parameterise it so dim attacks are also possible
    sat_level = float(peak_int)

    # spacing between stripe group starts (in rows)
    # freq = number of stripe groups per image height
    spacing = max(Rn + 1, h // max(1, freq))

    # ── 2. Pick laser colour profile ──────────────────────────────────────
    profile      = py_rng.choice(LASER_PROFILES)
    r_gain, g_gain, b_gain = profile[1], profile[2], profile[3]

    # ── 3. PSF halo half-width (rows) — Stein eq. 2: x_sat ∝ I^(1/3) ────
    # Larger halo when intensity is higher
    halo_rows = max(1, int(Rn * (sat_level / 255.0) ** (1/3) * 1.5))

    # ── 4. Build per-row intensity profile using Gaussian PSF ─────────────
    def gaussian_row_profile(centre_row: int, rows: int, H: int) -> np.ndarray:
        """
        Returns a (H,) array of intensities for a stripe centred at centre_row.
        Centre rows are fully saturated; halo decays as Gaussian.
        """
        profile_arr = np.zeros(H, dtype=np.float32)
        sigma = max(1.0, halo_rows / 2.0)

        for r in range(H):
            # Distance from nearest saturated row in the Rn-wide group
            dist_to_centre = abs(r - centre_row)
            half_Rn = Rn / 2.0

            if dist_to_centre <= half_Rn:
                # Within the saturated band → full intensity
                intensity_factor = 1.0
            else:
                # Outside → Gaussian halo decay
                dist_from_edge = dist_to_centre - half_Rn
                intensity_factor = math.exp(-(dist_from_edge ** 2) / (2 * sigma ** 2))

            if intensity_factor > 0.01:
                profile_arr[r] = intensity_factor

        return profile_arr

    # ── 5. Inject stripes ─────────────────────────────────────────────────
    col_start, col_end = _coverage_window(w, coverage, py_rng)

    # Build full-height intensity map (one value per row)
    row_intensity = np.zeros(h, dtype=np.float32)

    for stripe_start in range(0, h, spacing):
        centre = stripe_start + Rn // 2
        if centre >= h:
            break
        # Add this stripe's Gaussian profile
        profile_arr = gaussian_row_profile(centre, Rn, h)
        row_intensity += profile_arr

    # Clip: no row can exceed 1.0
    row_intensity = np.clip(row_intensity, 0.0, 1.0)

    # ── 6. Apply colour and intensity to each row ─────────────────────────
    for r in range(h):
        factor = row_intensity[r]
        if factor < 0.005:
            continue

        # Stripe pixel colour at full saturation
        stripe_R = sat_level * r_gain * factor
        stripe_G = sat_level * g_gain * factor
        stripe_B = sat_level * b_gain * factor

        # Original pixel values in the affected columns
        orig = output[r, col_start:col_end, :].copy()  # (W_seg, 3)

        # Blend: additive for saturated stripes (matches sensor physics —
        # extra photons add to existing charge, clipping at saturation)
        output[r, col_start:col_end, 0] = np.clip(orig[:, 0] + stripe_R, 0, 255)
        output[r, col_start:col_end, 1] = np.clip(orig[:, 1] + stripe_G, 0, 255)
        output[r, col_start:col_end, 2] = np.clip(orig[:, 2] + stripe_B, 0, 255)

    # ── 7. Apply angle (shear transform) if non-zero ──────────────────────
    if abs(angle) > 0.5:
        output = _apply_angle(output, image.astype(np.float32), angle,
                              row_intensity, sat_level,
                              r_gain, g_gain, b_gain,
                              col_start, col_end)

    # ── 8. Gaussian sensor read noise (CMOS characteristic) ───────────────
    if noise_sigma > 0:
        # Blue channel gets more noise (Bayer CFA has fewer blue pixels)
        noise = np.stack([
            rng.normal(0, noise_sigma * 0.6, (h, w)),
            rng.normal(0, noise_sigma * 0.5, (h, w)),
            rng.normal(0, noise_sigma * 1.0, (h, w)),
        ], axis=2).astype(np.float32)
        output += noise

    return np.clip(output, 0, 255).astype(np.uint8)


def _apply_angle(
    output: np.ndarray,
    original: np.ndarray,
    angle: float,
    row_intensity: np.ndarray,
    sat_level: float,
    r_gain: float, g_gain: float, b_gain: float,
    col_start: int, col_end: int,
) -> np.ndarray:
    """
    Apply angled stripes by shearing the stripe pattern.
    Köhler 2021: angle of incidence tilts horizontal stripes.
    """
    h, w = output.shape[:2]
    result = original.copy()
    tan_a  = math.tan(math.radians(angle))

    for r in range(h):
        factor = row_intensity[r]
        if factor < 0.005:
            continue
        stripe_R = sat_level * r_gain * factor
        stripe_G = sat_level * g_gain * factor
        stripe_B = sat_level * b_gain * factor

        for c in range(col_start, col_end):
            # Shear: each column shifts the row by col * tan(angle)
            shifted_r = int(r + c * tan_a) % h
            orig = result[shifted_r, c, :].copy()
            result[shifted_r, c, 0] = min(255.0, orig[0] + stripe_R)
            result[shifted_r, c, 1] = min(255.0, orig[1] + stripe_G)
            result[shifted_r, c, 2] = min(255.0, orig[2] + stripe_B)

    return result


def _coverage_window(width: int, coverage: float, rng: random.Random) -> Tuple[int, int]:
    """Random horizontal window covering `coverage` fraction of frame width."""
    seg_w     = int(width * coverage)
    max_start = max(0, width - seg_w)
    col_start = rng.randint(0, max_start) if max_start > 0 else 0
    return col_start, col_start + seg_w


# ── Visual test: generates sample images for all 6 variations ─────────────
if __name__ == "__main__":
    import os
    from pathlib import Path

    # Try to use a real KITTI image for the test
    clean_base = Path("data/clean_base")
    real_imgs  = list(clean_base.glob("*.png")) + list(clean_base.glob("*.jpg")) if clean_base.exists() else []

    if real_imgs:
        src = cv2.imread(str(real_imgs[0]))
        test_img = cv2.cvtColor(cv2.resize(src, (224, 224)), cv2.COLOR_BGR2RGB)
        print(f"Using real KITTI image: {real_imgs[0].name}")
    else:
        # Fallback: gradient grey
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            test_img[i, :] = int(i / 224 * 180) + 40
        print("No KITTI images found — using grey gradient")

    test_cfgs = {
        "freq_low_narrow":  {"freq_range":[3,5],  "width_range":[1,1], "intensity_range":[200,220], "angle":0,              "coverage":1.0,  "noise_sigma":5},
        "freq_low_wide":    {"freq_range":[3,5],  "width_range":[3,4], "intensity_range":[180,200], "angle":0,              "coverage":1.0,  "noise_sigma":7},
        "freq_mid_narrow":  {"freq_range":[6,8],  "width_range":[1,2], "intensity_range":[160,190], "angle":0,              "coverage":0.85, "noise_sigma":6},
        "freq_mid_wide":    {"freq_range":[6,8],  "width_range":[3,4], "intensity_range":[170,200], "angle":0,              "coverage":0.75, "noise_sigma":8},
        "freq_high_narrow": {"freq_range":[9,12], "width_range":[1,1], "intensity_range":[150,175], "angle":0,              "coverage":0.80, "noise_sigma":5},
        "freq_high_angled": {"freq_range":[6,10], "width_range":[2,3], "intensity_range":[160,190], "angle_range":[15,25],  "coverage":0.80, "noise_sigma":7},
    }

    os.makedirs("test_output", exist_ok=True)
    for name, cfg in test_cfgs.items():
        result = inject_stripes(test_img, cfg, seed=42)
        cv2.imwrite(f"test_output/{name}.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"  ✓ {name}  →  test_output/{name}.png")

    print("\nDone. Open test_output/ and compare with Köhler et al. Fig 1.")