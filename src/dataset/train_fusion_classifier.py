"""
train_joint_fusion.py
-----------------------
ONE combined model for rolling-shutter attack detection.

Input per frame:
  (a) the image itself                     -> ResNet18 branch (512-d embedding)
  (b) SSD detection-confidence statistics  -> small MLP branch (16-d embedding)
      (num_detections, mean/max/std confidence, low-confidence count; produced
       by extract_detection_features.py with a FROZEN SSD300)
Both embeddings are concatenated and fed to one classification head trained
jointly with cross-entropy. Output: P(attacked) for the frame.

Why the detector branch uses precomputed features: SSD stays frozen, so its
statistics are fixed per image and can be computed once. Only the ResNet18,
the MLP and the head are trained.

ABLATION IS BUILT IN (--mode):
  cnn     image only              (does the detector signal add anything?)
  det     detector features only  (how much does the detector signal carry alone?)
  fusion  both                    (the proposed combined model)
  all     runs the three above and prints one comparison table
If `fusion` does not beat `cnn` on the held-out test split, the detector
signal is not adding information and the paper should say so.

SPLITTING: by source background image (source_clean_image), NOT by frame. The
dataset reuses ~7,500 backgrounds across ~97,500 images; a frame-level split
puts the same road scene in train and test. Optionally hold out a whole attack
variation (--holdout_variation) to test generalisation to an unseen frequency
band.

CONFIDENCE: the "confidence of surety" is P(attacked) after temperature
scaling fitted on the validation split (never on test). Calibration error
(ECE) is reported before and after.

CLASS BALANCE: the full dataset is ~87% attacked, so plain accuracy is
misleading. Reported: balanced accuracy, ROC-AUC, recall (attack detection
rate), specificity (clean frames correctly passed), TPR at 1% false-alarm rate.

Typical use (run from src/dataset/):
    # 1) features for a balanced subset (SSD inference; do this once)
    python extract_detection_features.py --config ../../configs/config.yaml \
        --labels ../../data/final_dataset/labels_paired.csv \
        --out ../../data/final_dataset/det_features_balanced.csv \
        --max_per_variation 2000
    # 2) train + ablate
    python train_joint_fusion.py --config ../../configs/config.yaml \
        --det_features ../../data/final_dataset/det_features_balanced.csv \
        --out_dir ../../data/final_dataset/joint_fusion_results --mode all
    # 3) classify one new frame with the trained fusion model
    python train_joint_fusion.py --config ../../configs/config.yaml \
        --predict_image some_frame.png \
        --checkpoint ../../data/final_dataset/joint_fusion_results/model_fusion.pt
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    TORCH_OK = True
except ImportError:  # pure-numpy helpers below stay importable/testable without torch
    TORCH_OK = False

DET_COLS = ["num_detections", "mean_confidence", "max_confidence", "std_confidence", "low_conf_count"]
LOG_COLS = {"num_detections", "low_conf_count"}      # heavy-tailed counts -> log1p
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224


# ---------------------------------------------------------------------------
# pure helpers (numpy / pandas / sklearn only)
# ---------------------------------------------------------------------------

def norm_path(p):
    """labels.csv was written on Windows (backslashes); make paths portable."""
    return str(p).replace("\\", "/")


def group_split(df, group_col, fracs=(0.70, 0.15, 0.15), seed=42):
    """Assign every row to train/val/test so that all rows sharing a group
    (a source background image) land in the SAME part."""
    df = df.dropna(subset=[group_col]).copy()
    groups = np.array(sorted(df[group_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    part_of = {}
    for g in groups[:n_tr]:
        part_of[g] = "train"
    for g in groups[n_tr:n_tr + n_va]:
        part_of[g] = "val"
    for g in groups[n_tr + n_va:]:
        part_of[g] = "test"
    df["part"] = df[group_col].map(part_of)
    # hard guarantee: no background appears in two parts
    sets = {k: set(df.loc[df["part"] == k, group_col]) for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "group leakage across parts"
    return df


def apply_holdout(df, holdout_variation):
    """Remove one attack variation from train/val (it stays in test), to test
    generalisation to an attack regime never seen in training."""
    if not holdout_variation:
        return df
    held = [v.strip() for v in str(holdout_variation).split(",") if v.strip()]
    keep = (df["part"] == "test") | (~df["variation"].isin(held))
    return df[keep].copy()


def det_matrix(df):
    X = df[DET_COLS].astype(float).fillna(0.0).copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.values.astype(np.float32)


def fit_standardizer(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Single temperature T minimising NLL on held-out logits (Guo et al. 2017)."""
    from scipy.optimize import minimize_scalar
    labels = np.asarray(labels).astype(int)
    idx = np.arange(len(labels))

    def nll(log_t):
        p = softmax_np(logits / np.exp(log_t))
        return -np.mean(np.log(np.clip(p[idx, labels], 1e-12, 1.0)))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def ece_score(p_attacked, y, n_bins=15):
    """Expected calibration error of the predicted-class confidence."""
    p = np.asarray(p_attacked, dtype=float)
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)          # in [0.5, 1]
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    recall = tp / n_pos if n_pos else float("nan")
    specificity = tn / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0 else float("nan"))
    auc = tpr1 = float("nan")
    if n_pos and n_neg:
        auc = float(roc_auc_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        tpr1 = float(np.interp(0.01, fpr, tpr))
    return {
        "n": int(len(y)), "base_rate_attacked": n_pos / len(y),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "recall_attacked": recall, "specificity_clean": specificity,
        "precision_attacked": precision, "f1_attacked": f1,
        "roc_auc": auc, "tpr_at_1pct_fpr": tpr1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# torch model / data
# ---------------------------------------------------------------------------

if TORCH_OK:

    class FrameDataset(Dataset):
        def __init__(self, df, root, det_X, need_image):
            self.paths = [str(Path(root) / norm_path(p)) for p in df["path"]]
            self.y = df["is_attacked"].astype(int).values
            self.det = det_X
            self.need_image = need_image

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            if self.need_image:
                import cv2
                img = cv2.imread(self.paths[i])
                if img is None:
                    raise FileNotFoundError(self.paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
            else:
                x = torch.zeros(1)
            return x, torch.from_numpy(self.det[i]), int(self.y[i])

    class FusionNet(nn.Module):
        """mode: 'cnn' | 'det' | 'fusion'."""

        def __init__(self, mode, n_det, pretrained=True, dropout=0.3):
            super().__init__()
            self.mode = mode
            dim = 0
            if mode in ("cnn", "fusion"):
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone = torchvision.models.resnet18(weights=weights)
                self.backbone.fc = nn.Identity()
                dim += 512
            if mode in ("det", "fusion"):
                self.det_mlp = nn.Sequential(
                    nn.Linear(n_det, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
                dim += 16
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))

        def forward(self, img, det):
            parts = []
            if self.mode in ("cnn", "fusion"):
                parts.append(self.backbone(img))
            if self.mode in ("det", "fusion"):
                parts.append(self.det_mlp(det))
            return self.head(torch.cat(parts, dim=1))

    def run_epoch(model, loader, device, criterion, optimizer=None):
        train = optimizer is not None
        model.train(train)
        total, loss_sum = 0, 0.0
        all_logits, all_y = [], []
        with torch.set_grad_enabled(train):
            for img, det, y in loader:
                img, det, y = img.to(device), det.to(device), y.to(device)
                logits = model(img, det)
                loss = criterion(logits, y)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(y)
                total += len(y)
                all_logits.append(logits.detach().cpu().numpy())
                all_y.append(y.cpu().numpy())
        return loss_sum / total, np.concatenate(all_logits), np.concatenate(all_y)


def train_one_mode(mode, parts, root, args, device):
    tr, va, te = parts["train"], parts["val"], parts["test"]
    mu, sd = fit_standardizer(det_matrix(tr))        # statistics from TRAIN only

    def build(df, shuffle, drop_last=False):
        X = (det_matrix(df) - mu) / sd
        ds = FrameDataset(df, root, X, need_image=mode in ("cnn", "fusion"))
        return DataLoader(ds, batch_size=max(1, min(args.batch_size, len(df))), shuffle=shuffle,
                          drop_last=drop_last, num_workers=args.num_workers)

    train_loader = build(tr, True, drop_last=True)
    val_loader = build(va, False)
    test_loader = build(te, False)

    model = FusionNet(mode, len(DET_COLS), pretrained=not args.no_pretrained,
                      dropout=args.dropout).to(device)
    counts = np.bincount(tr["is_attacked"].astype(int).values, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    lr = args.lr if args.lr is not None else (1e-3 if mode == "det" else 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc, best_state, bad = float("-inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_logits, va_y = run_epoch(model, val_loader, device, criterion)
        va_p = softmax_np(va_logits)[:, 1]
        va_auc = roc_auc_score(va_y, va_p) if len(set(va_y)) > 1 else float("nan")
        print(f"[{mode}] epoch {epoch}/{args.epochs}  train_loss {tr_loss:.4f}  "
              f"val_loss {va_loss:.4f}  val_auc {va_auc:.4f}", flush=True)
        score = va_auc if va_auc == va_auc else -va_loss
        if score > best_auc:
            best_auc, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{mode}] early stop (no val improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    _, va_logits, va_y = run_epoch(model, val_loader, device, criterion)
    _, te_logits, te_y = run_epoch(model, test_loader, device, criterion)

    T = fit_temperature(va_logits, va_y)             # fitted on VAL only
    p_raw = softmax_np(te_logits)[:, 1]
    p_cal = softmax_np(te_logits / T)[:, 1]

    m_raw = binary_metrics(te_y, p_raw)
    m_cal = binary_metrics(te_y, p_cal)
    m_cal["ece_raw"] = ece_score(p_raw, te_y)
    m_cal["ece_calibrated"] = ece_score(p_cal, te_y)
    m_cal["temperature"] = T

    preds = te[["path", "variation", "is_attacked"]].copy()
    preds["p_attacked_raw"] = p_raw
    preds["p_attacked_calibrated"] = p_cal
    per_var = {}
    for var, g in preds.groupby("variation"):
        if var == "clean":
            per_var[var] = {"n": int(len(g)), "specificity_clean": float((g["p_attacked_calibrated"] < 0.5).mean()),
                            "mean_p_attacked": float(g["p_attacked_calibrated"].mean())}
        else:
            per_var[var] = {"n": int(len(g)), "recall_attacked": float((g["p_attacked_calibrated"] >= 0.5).mean()),
                            "mean_p_attacked": float(g["p_attacked_calibrated"].mean())}

    os.makedirs(args.out_dir, exist_ok=True)
    preds.to_csv(os.path.join(args.out_dir, f"predictions_test_{mode}.csv"), index=False)
    with open(os.path.join(args.out_dir, f"metrics_{mode}.json"), "w") as f:
        json.dump({"mode": mode, "test_calibrated": m_cal, "test_uncalibrated": m_raw,
                   "per_variation_test": per_var,
                   "split_sizes": {k: int(len(v)) for k, v in parts.items()}}, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "mode": mode, "temperature": T,
                "det_mu": mu.tolist(), "det_sd": sd.tolist(), "det_cols": DET_COLS,
                "dropout": args.dropout}, os.path.join(args.out_dir, f"model_{mode}.pt"))
    return m_cal, per_var


def predict_image(args):
    """Classify ONE new frame with a trained checkpoint (runs SSD for the detector features)."""
    import cv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu")
    mode = ck["mode"]
    img = cv2.imread(args.predict_image)
    if img is None:
        sys.exit(f"cannot read {args.predict_image}")
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    det_row = np.zeros((1, len(DET_COLS)), dtype=np.float32)
    if mode in ("det", "fusion"):
        from extract_detection_features import extract_features, load_model as load_ssd
        ssd, pre = load_ssd(device)
        f = extract_features(ssd, pre, img, args.det_conf, device)   # same SSD + same floor as training
        df = pd.DataFrame([f])
        det_row = (det_matrix(df) - np.array(ck["det_mu"], dtype=np.float32)) / np.array(ck["det_sd"], dtype=np.float32)
        print("detector features:", {k: round(float(v), 4) for k, v in f.items()})

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)

    model = FusionNet(mode, len(DET_COLS), pretrained=False, dropout=ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    with torch.no_grad():
        logits = model(x.to(device), torch.from_numpy(det_row.astype(np.float32)).to(device)).cpu().numpy()
    p = float(softmax_np(logits / ck["temperature"])[0, 1])
    print(json.dumps({"is_attacked": p >= 0.5, "p_attacked": round(p, 4),
                      "confidence_in_decision": round(max(p, 1 - p), 4)}))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--det_features", help="CSV from extract_detection_features.py")
    ap.add_argument("--labels", default=None,
                    help="labels_paired.csv; only needed if the features CSV lacks source_clean_image")
    ap.add_argument("--out_dir", default="joint_fusion_results")
    ap.add_argument("--mode", choices=["cnn", "det", "fusion", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (cnn/fusion), 1e-3 (det)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="do not load ImageNet weights")
    ap.add_argument("--holdout_variation", default=None,
                    help="comma-separated variations excluded from train/val and evaluated on test only. NOTE freq_random_full spans 60-5000 Hz, so to truly hold out the ultra band use freq_ultra_aliasing,freq_random_full")
    ap.add_argument("--limit", type=int, default=None, help="debug: randomly keep only N rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_done", action="store_true",
                    help="skip any mode whose model_<mode>.pt and metrics_<mode>.json already exist in --out_dir "
                         "(restart-safe on Kaggle: a killed session only redoes the unfinished model)")
    ap.add_argument("--predict_image", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--det_conf", type=float, default=0.15, help="SSD confidence floor for features")
    args = ap.parse_args()

    if not TORCH_OK:
        sys.exit("torch/torchvision are not importable in this environment.")

    if args.predict_image:
        if not args.checkpoint:
            sys.exit("--predict_image needs --checkpoint")
        predict_image(args)
        return
    if not args.det_features:
        sys.exit("--det_features is required for training")

    import yaml
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = config_path.parent.parent / cfg["paths"]["final_dataset"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = pd.read_csv(args.det_features)
    if "source_clean_image" not in df.columns or df["source_clean_image"].isna().all():
        if not args.labels:
            sys.exit("features CSV has no source_clean_image; pass --labels labels_paired.csv")
        lab = pd.read_csv(args.labels)[["path", "source_clean_image"]]
        lab["_k"] = lab["path"].map(norm_path)
        df["_k"] = df["path"].map(norm_path)
        df = df.drop(columns=["source_clean_image"], errors="ignore").merge(
            lab.drop(columns=["path"]), on="_k", how="left").drop(columns=["_k"])
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)

    df = group_split(df, "source_clean_image", seed=args.seed)
    df = apply_holdout(df, args.holdout_variation)
    parts = {k: df[df["part"] == k].reset_index(drop=True) for k in ("train", "val", "test")}
    for k, v in parts.items():
        print(f"{k}: {len(v)} images, {v['source_clean_image'].nunique()} backgrounds, "
              f"{v['is_attacked'].mean():.1%} attacked")

    modes = ["cnn", "det", "fusion"] if args.mode == "all" else [args.mode]
    summary = {}
    for mode in modes:
        mpath = os.path.join(args.out_dir, f"metrics_{mode}.json")
        if args.skip_done and os.path.isfile(mpath) and os.path.isfile(os.path.join(args.out_dir, f"model_{mode}.pt")):
            print(f"\n=== mode {mode}: already trained, loading saved metrics (--skip_done) ===")
            with open(mpath) as f:
                summary[mode] = json.load(f)["test_calibrated"]
            continue
        print(f"\n=== training mode: {mode} ===")
        summary[mode], per_var = train_one_mode(mode, parts, root, args, device)

    cols = ["balanced_accuracy", "roc_auc", "recall_attacked", "specificity_clean",
            "precision_attacked", "f1_attacked", "tpr_at_1pct_fpr", "fp", "fn",
            "ece_raw", "ece_calibrated"]
    table = pd.DataFrame(summary).T[cols]
    print("\n=== TEST-SET COMPARISON (backgrounds never seen in training) ===")
    print(table.round(4).to_string())
    table.to_csv(os.path.join(args.out_dir, "ablation_table.csv"))
    print(f"\nbase rate of 'attacked' in test: {summary[modes[0]]['base_rate_attacked']:.1%} "
          f"(accuracy alone is misleading when this is far from 50%)")
    print(f"saved to {args.out_dir}/ (ablation_table.csv, metrics_*.json, predictions_test_*.csv, model_*.pt)")


if __name__ == "__main__":
    main()
