from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from config import IMAGE_ROOT, LABELS_CANDIDATES, SEED

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_labels_path():
    for p in LABELS_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "labels.csv not found. Expected one of: " +
        ", ".join(str(p) for p in LABELS_CANDIDATES)
    )

def load_labels():
    df = pd.read_csv(get_labels_path())
    required = {"path", "split", "variation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    df["path"] = df["path"].astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["variation"] = df["variation"].astype(str).str.strip()
    return df

def resolve_image_path(path_value, variation=None):
    p = Path(str(path_value))
    if p.is_absolute() and p.exists():
        return p
    candidates = [IMAGE_ROOT / p]
    if variation:
        candidates.append(IMAGE_ROOT / str(variation) / p)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Image not found for path={path_value}, variation={variation}")

def relative_image_path(path_value, variation=None):
    p = Path(str(path_value))
    if p.is_absolute():
        try:
            return p.relative_to(IMAGE_ROOT)
        except ValueError:
            pass
    # If path already starts with a variation directory, do not prepend it.
    if len(p.parts) > 1 and p.parts[0] in {
        "clean", "freq_low_wide", "freq_mid_narrow",
        "freq_high_fine", "freq_ultra_aliasing", "freq_random_full"
    }:
        return p
    if variation:
        return Path(str(variation)) / p
    return p

def get_split(df, name):
    part = df[df["split"] == name].copy()
    if part.empty:
        raise ValueError(
            f"No rows found for split='{name}'. Found: {sorted(df['split'].unique())}"
        )
    return part
