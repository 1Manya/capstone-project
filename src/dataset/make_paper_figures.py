"""
make_paper_figures.py
------------------------
Publication-style figures + LaTeX tables from your result CSVs. Every
figure is written as vector PDF (for LaTeX) and PNG (for slides), at IEEE
double-column width (7.16 in), colorblind-safe palette, hatch patterns so
bars stay distinguishable in greyscale print.

INPUTS (each optional; a figure whose input is missing is skipped):
  --detectors LABEL=path/to/box_level_results.csv [LABEL=... ...]
        one entry per detector, e.g.
        SSD300=.../results_ssd/box_level_results_ssd.csv
        YOLOv8n=.../results_yolov8n/box_level_results_full.csv
        YOLOv8m=.../results_yolov8m/box_level_results_full.csv
        (CSV must have image_id, power_mw, coverage -> produced by the updated
         attack_effect_analysis_full.py / run_ssd_detection.py)
  --classifier_dir  folder written by train_joint_fusion.py
        (predictions_test_{cnn,det,fusion}.csv and ablation_table.csv)

OUTPUTS (in --out_dir):
  fig_detector_degradation   (a) ASR by attack profile  (b) beam overlap of
                             suppressed vs retained objects  (c) dose-response
  fig_detector_confidence    (a) ECDF of clean-frame confidence, suppressed vs
                             retained  (b) induced false positives
  fig_classifier_evaluation  (a) ROC  (b) reliability diagram  (c) detection
                             rate per attack profile
  table_detectors.tex / .csv, table_classifier.tex / .csv
  captions_and_definitions.txt   exact metric definitions to paste in the paper

STATISTICS: confidence intervals are 95% percentile CIs from a CLUSTER
bootstrap that resamples whole IMAGES (objects in one image are not
independent). Classifier detection rates use Wilson intervals.

Usage:
    python make_paper_figures.py \
        --detectors SSD300=r_ssd/box_level_results_ssd.csv YOLOv8n=r_n/box_level_results_full.csv \
        --classifier_dir joint_fusion_results --out_dir paper_figures
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\"]
PROFILE_ORDER = ["freq_low_wide", "freq_mid_narrow", "freq_high_fine", "freq_ultra_aliasing", "freq_random_full"]
PROFILE_LABEL = {   # frequency bands taken from config.yaml (frequency_range per variation); unit Hz
    "freq_low_wide": "Low\n60–200",
    "freq_mid_narrow": "Mid\n200–800",
    "freq_high_fine": "High\n0.8–2k",
    "freq_ultra_aliasing": "Ultra\n2k–5k",
    "freq_random_full": "Full\n60–5k",
    "clean": "Clean\n(TNR)",
}
XLABEL_BAND = "Attack profile: modulation-frequency band (Hz)"
MODE_SHORT = {"cnn": "CNN", "det": "Detector feat.", "fusion": "CNN + detector"}
MODE_LABEL = {"cnn": "CNN (image only)", "det": "Detector features only", "fusion": "CNN + detector (fusion)"}
OVERLAP_THR = 0.3   # beam-overlap threshold used for 'inside the attack region'


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavuserif",
    })


def pretty(v):
    return PROFILE_LABEL.get(v, v.replace("_", " "))


def panel_tag(ax, tag):
    ax.text(-0.16, 1.06, tag, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save_fig(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- statistics

def cluster_boot_ratio(num, den, n_boot=1000, seed=0, chunk=100):
    """sum(num)/sum(den) with a 95% percentile CI, resampling clusters (images).
    num, den: one entry per image."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return np.nan, np.nan, np.nan
    point = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        bn, bd = num[idx].sum(1), den[idx].sum(1)
        ok = bd > 0
        out.append(bn[ok] / bd[ok])
    r = np.concatenate(out)
    return point, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def yerr(y, lo, hi):
    """Asymmetric error bars, clipped at 0 (float error can make a half-width -1e-17 at rates of exactly 0 or 1)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    return np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def per_image_table(box_df):
    """One row per (variation, image) with the counts every metric needs."""
    d = box_df.copy()
    lost, matched, fp = d["outcome"] == "lost", d["outcome"] == "matched", d["outcome"] == "new_false_positive"
    d["lost"], d["matched"], d["fp"] = lost.astype(int), matched.astype(int), fp.astype(int)
    d["lost_in_region"] = (lost & (d["attack_region_overlap"] >= OVERLAP_THR)).astype(int)
    d["ov_lost"] = np.where(lost, d["attack_region_overlap"], 0.0)
    d["ov_matched"] = np.where(matched, d["attack_region_overlap"], 0.0)
    d["dconf_sum"] = np.where(matched, d["conf_drop"], 0.0)
    cols = ["lost", "matched", "fp", "lost_in_region", "ov_lost", "ov_matched", "dconf_sum"]
    g = d.groupby(["variation", "image_id"], as_index=False)[cols].sum()
    g["base"] = g["lost"] + g["matched"]
    return g


def profiles_present(df):
    return [p for p in PROFILE_ORDER if p in set(df["variation"])] + \
           [p for p in sorted(set(df["variation"])) if p not in PROFILE_ORDER and p != "clean"]


# ---------------------------------------------------------------- detector figures

def detector_tables(dets, n_boot, seed):
    """Metrics per detector x profile (+ pooled 'All attacks')."""
    rows, per_img = [], {}
    for label, df in dets.items():
        img = per_image_table(df)
        per_img[label] = (df, img)
        groups = [(p, img[img["variation"] == p]) for p in profiles_present(df)] + [("All attacks", img)]
        for prof, g in groups:
            asr = cluster_boot_ratio(g["lost"], g["base"], n_boot, seed)
            sar = cluster_boot_ratio(g["lost_in_region"], g["lost"], n_boot, seed)
            fpr = cluster_boot_ratio(g["fp"] * 100.0, g["base"], n_boot, seed)
            prec = cluster_boot_ratio(g["matched"], g["matched"] + g["fp"], n_boot, seed)
            dconf = cluster_boot_ratio(g["dconf_sum"], g["matched"], n_boot, seed)
            rec = 1 - asr[0] if asr[0] == asr[0] else np.nan
            f1 = 2 * prec[0] * rec / (prec[0] + rec) if (prec[0] + rec) > 0 else np.nan
            rows.append({
                "detector": label, "profile": prof, "n_images": int((g["base"] > 0).sum()),
                "n_baseline_objects": int(g["base"].sum()),
                "ASR": asr[0], "ASR_lo": asr[1], "ASR_hi": asr[2],
                "SAR": sar[0], "SAR_lo": sar[1], "SAR_hi": sar[2],
                "FP_per_100": fpr[0], "FP_lo": fpr[1], "FP_hi": fpr[2],
                "Precision": prec[0], "F1": f1,
                "dConf": dconf[0], "dConf_lo": dconf[1], "dConf_hi": dconf[2],
            })
    return pd.DataFrame(rows), per_img


def fig_detector_degradation(tab, per_img, out_dir, n_boot, seed):
    labels = list(per_img)
    profs = [p for p in PROFILE_ORDER if p in set(tab["profile"])] + \
            [p for p in tab["profile"].unique() if p not in PROFILE_ORDER and p != "All attacks"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1.5, 1.55, 1.0]})
    x = np.arange(len(profs))
    w = 0.8 / max(1, len(labels))

    # (a) ASR by profile
    ax = axes[0]
    for i, lab in enumerate(labels):
        t = tab[(tab["detector"] == lab)].set_index("profile").reindex(profs)
        y = t["ASR"].values * 100
        err = yerr(y, t["ASR_lo"].values * 100, t["ASR_hi"].values * 100)
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, y, w, yerr=err, capsize=1.5, color=OKABE_ITO[i % 6],
               hatch=HATCHES[i % 5], edgecolor="white", linewidth=0.4, error_kw={"elinewidth": 0.7}, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([pretty(p) for p in profs], fontsize=5.5)
    ax.set_ylabel("Attack Success Rate, ASR (%)"); ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel(XLABEL_BAND, fontsize=6.5)
    ax.legend(frameon=False, ncol=min(3, len(labels)), loc="upper center", handlelength=1.2, columnspacing=1.0)
    panel_tag(ax, "(a)")

    # (b) beam overlap: suppressed vs retained
    ax = axes[1]
    lab0 = labels[0]
    width = 0.38
    for j, (kind, col, hatch) in enumerate([("Suppressed", "ov_lost", ""), ("Retained", "ov_matched", "///")]):
        ys, lo, hi = [], [], []
        for prof in profs:
            g = per_img[lab0][1]
            g = g[g["variation"] == prof]
            den = g["lost"] if kind == "Suppressed" else g["matched"]
            p, l, h = cluster_boot_ratio(g[col], den, n_boot, seed)
            ys.append(p); lo.append(l); hi.append(h)
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.bar(x + (j - 0.5) * width, ys, width, yerr=yerr(ys, lo, hi), capsize=1.5,
               color=["#D55E00", "#0072B2"][j], hatch=hatch, edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 0.7}, label=kind)
    ax.set_xticks(x); ax.set_xticklabels([pretty(p) for p in profs], fontsize=5.5)
    ax.set_ylabel("Mean beam–object overlap"); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(XLABEL_BAND, fontsize=6.5)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.5, columnspacing=1.0,
              title=f"{lab0}", title_fontsize=6.5)
    panel_tag(ax, "(b)")

    # (c) dose-response on severity = power x coverage
    ax = axes[2]
    drew = False
    for i, lab in enumerate(labels):
        df = per_img[lab][0]
        if not {"power_mw", "coverage"} <= set(df.columns):
            continue
        d = df[df["outcome"].isin(["lost", "matched"])].dropna(subset=["power_mw", "coverage"]).copy()
        if d.empty:
            continue
        d["sev"] = d["power_mw"].astype(float) * d["coverage"].astype(float)
        d["bin"] = pd.qcut(d["sev"], 6, duplicates="drop")
        d["lost"] = (d["outcome"] == "lost").astype(int)
        d["base"] = 1
        xs, ys, lo, hi = [], [], [], []
        for b, g in d.groupby("bin", observed=True):
            gi = g.groupby("image_id")[["lost", "base"]].sum()
            p, l, h = cluster_boot_ratio(gi["lost"], gi["base"], n_boot, seed)
            xs.append(g["sev"].median()); ys.append(p * 100); lo.append(l * 100); hi.append(h * 100)
        ax.plot(xs, ys, marker="o", ms=3, color=OKABE_ITO[i % 6], label=lab)
        ax.fill_between(xs, lo, hi, color=OKABE_ITO[i % 6], alpha=0.18, linewidth=0)
        drew = True
    if drew:
        ax.set_xlabel("Severity, $P\\times c$ (mW)")
        ax.set_ylabel("ASR (%)"); ax.set_ylim(0, 105)
        ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    else:
        ax.text(0.5, 0.5, "power_mw / coverage\ncolumns not present", ha="center", va="center", fontsize=7)
        ax.set_axis_off()
    panel_tag(ax, "(c)")
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, out_dir, "fig_detector_degradation")


def fig_detector_confidence(tab, per_img, out_dir, n_boot, seed):
    labels = list(per_img)
    profs = [p for p in PROFILE_ORDER if p in set(tab["profile"])] + \
            [p for p in tab["profile"].unique() if p not in PROFILE_ORDER and p != "All attacks"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    for i, lab in enumerate(labels):
        df = per_img[lab][0]
        for outcome, ls in (("matched", "-"), ("lost", "--")):
            v = np.sort(df.loc[df["outcome"] == outcome, "clean_conf"].values)
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls=ls, color=OKABE_ITO[i % 6])
    ax.set_xlabel("Clean-frame detection confidence"); ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 1.0)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=OKABE_ITO[i % 6], lw=1.4, label=lab) for i, lab in enumerate(labels)]
    handles += [Line2D([0], [0], color="k", lw=1.1, ls="-", label="retained"),
                Line2D([0], [0], color="k", lw=1.1, ls="--", label="suppressed")]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    panel_tag(ax, "(a)")

    ax = axes[1]
    x = np.arange(len(profs)); w = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        t = tab[tab["detector"] == lab].set_index("profile").reindex(profs)
        y = t["FP_per_100"].values
        err = yerr(y, t["FP_lo"].values, t["FP_hi"].values)
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, y, w, yerr=err, capsize=1.5, color=OKABE_ITO[i % 6],
               hatch=HATCHES[i % 5], edgecolor="white", linewidth=0.4, error_kw={"elinewidth": 0.7}, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([pretty(p) for p in profs], fontsize=5.5)
    ax.set_ylabel("Induced false positives\nper 100 baseline detections")
    ax.set_xlabel(XLABEL_BAND, fontsize=6.5)
    ax.set_ylim(0, float(np.nanmax(tab["FP_hi"])) * 1.35)
    ax.legend(frameon=False, ncol=min(3, len(labels)), fontsize=6.5, loc="upper center")
    panel_tag(ax, "(b)")
    fig.tight_layout(w_pad=1.2)
    save_fig(fig, out_dir, "fig_detector_confidence")


# ---------------------------------------------------------------- classifier figures

def load_predictions(cdir):
    preds = {}
    for m in ("cnn", "det", "fusion"):
        p = os.path.join(cdir, f"predictions_test_{m}.csv")
        if os.path.isfile(p):
            preds[m] = pd.read_csv(p)
    return preds


def reliability_curve(p_attacked, y, n_bins=10):
    p, y = np.asarray(p_attacked, float), np.asarray(y).astype(int)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig_classifier(preds, out_dir):
    modes = [m for m in ("cnn", "det", "fusion") if m in preds]
    colors = {"cnn": OKABE_ITO[0], "det": OKABE_ITO[4], "fusion": OKABE_ITO[1]}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.75]})

    ax = axes[0]
    for m in modes:
        d = preds[m]
        y, p = d["is_attacked"].values, d["p_attacked_calibrated"].values
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colors[m], label=f"{MODE_SHORT[m]} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
    ax.set_xscale("symlog", linthresh=1e-2); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Attack detection rate (TPR)")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.27), handlelength=1.4)
    panel_tag(ax, "(a)")

    ax = axes[1]
    ref = "fusion" if "fusion" in preds else modes[0]
    d = preds[ref]
    for col, ls, name in (("p_attacked_raw", "--", "uncalibrated"), ("p_attacked_calibrated", "-", "temperature-scaled")):
        xs, ys, ns = reliability_curve(d[col].values, d["is_attacked"].values)
        ax.plot(xs, ys, ls=ls, marker="o", ms=2.5, color=colors[ref], label=name)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Predicted P(attacked)"); ax.set_ylabel("Observed attack frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(MODE_LABEL[ref], fontsize=7, pad=2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    panel_tag(ax, "(b)")

    ax = axes[2]
    cats = ["clean"] + [p for p in PROFILE_ORDER if p in set(preds[modes[0]]["variation"])]
    x = np.arange(len(cats)); w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        d = preds[m]; ys, lo, hi = [], [], []
        for c in cats:
            g = d[d["variation"] == c]
            k = int((g["p_attacked_calibrated"] < 0.5).sum()) if c == "clean" else int((g["p_attacked_calibrated"] >= 0.5).sum())
            p, l, h = wilson(k, len(g))
            ys.append(p); lo.append(l); hi.append(h)
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.bar(x + (i - (len(modes) - 1) / 2) * w, ys, w, yerr=yerr(ys, lo, hi), capsize=1.2,
               color=colors[m], hatch=HATCHES[i % 5], edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 0.6}, label=MODE_SHORT[m])
    ax.set_xticks(x); ax.set_xticklabels([pretty(c) for c in cats], fontsize=5.8)
    ax.set_ylabel("Correct-decision rate"); ax.set_ylim(0, 1.3); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel(XLABEL_BAND, fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.3, ncol=3, loc="upper center", columnspacing=1.0)
    panel_tag(ax, "(c)")
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, out_dir, "fig_classifier_evaluation")


# ---------------------------------------------------------------- tables / text

def latex_detector_table(tab):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Object-detector degradation under the rolling-shutter laser attack. ASR: attack success rate; "
         r"SAR: spatial attribution rate; FP$_{100}$: induced false positives per 100 baseline detections; "
         r"$\Delta$Conf: mean confidence drop of retained detections. 95\% cluster-bootstrap CIs over images.}",
         r"\label{tab:detector_degradation}", r"\footnotesize", r"\begin{tabular}{llrrcccc}", r"\toprule",
         r"Detector & Attack profile & $N_{\mathrm{obj}}$ & $N_{\mathrm{img}}$ & ASR (\%) & SAR (\%) & FP$_{100}$ & $\Delta$Conf \\",
         r"\midrule"]
    for det, g in tab.groupby("detector", sort=False):
        first = True
        for _, r in g.iterrows():
            name = r["profile"] if r["profile"] == "All attacks" else PROFILE_LABEL.get(r["profile"], r["profile"]).replace("\n", " ")
            L.append(f"{det if first else ''} & {name} & {r['n_baseline_objects']} & {r['n_images']} & "
                     f"{r['ASR']*100:.1f} [{r['ASR_lo']*100:.1f}, {r['ASR_hi']*100:.1f}] & "
                     f"{r['SAR']*100:.1f} & {r['FP_per_100']:.1f} & {r['dConf']:.3f} \\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L).replace("–", "--")


def latex_classifier_table(abl):
    cols = [("balanced_accuracy", "Bal.\\ acc."), ("roc_auc", "ROC-AUC"), ("recall_attacked", "TPR"),
            ("specificity_clean", "TNR"), ("f1_attacked", "F1"), ("tpr_at_1pct_fpr", "TPR@1\\%FPR"),
            ("ece_calibrated", "ECE")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Attack classification on held-out backgrounds. TPR/TNR at threshold 0.5; ECE after temperature scaling.}",
         r"\label{tab:classifier}", r"\footnotesize", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Model & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for m in abl.index:
        L.append(MODE_LABEL.get(m, m) + " & " + " & ".join(f"{abl.loc[m, c]:.3f}" for c, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


DEFINITIONS = """METRIC DEFINITIONS (paste into Section: Evaluation Metrics)

Reference detections. No human-annotated boxes exist for the synthetic frames, so the detections
a detector produces on the clean version of a frame (confidence >= tau) are the reference set,
following the practice of Kohler et al. (ACSAC 2021).
Matching. A reference detection is RETAINED if the attacked frame contains a box with
IoU >= {iou} {cls}; otherwise it is SUPPRESSED. Attacked-frame boxes matching no reference
detection are induced false positives (FP).
ASR (attack success rate) = suppressed / (suppressed + retained), pooled over reference
detections. NOTE: this equals Kohler et al.'s "hidden objects" fraction only when matching is
class-agnostic with IoU > 0.5; state the setting you ran.
SAR (spatial attribution rate) = fraction of suppressed detections whose box overlaps the beam
footprint by >= {ov:.0%} (footprint = beam-intensity mask >= 0.3, recomputed from the stored
laser parameters of each frame). Compare with the overlap of retained detections (Fig. detector_degradation b)
to judge whether suppression is spatially localised or a global frame degradation.
FP_100 = induced false positives per 100 reference detections.
dConf = mean(clean confidence - attacked confidence) over retained detections.
Severity = laser power (mW) x beam coverage (fraction of frame width).
Confidence intervals: 95% percentile cluster bootstrap resampling images (n_boot = {nb}).
Classifier: TPR = attacked frames flagged; TNR = clean frames passed; TPR@1%FPR = TPR at the
operating point with 1% false alarms (a false alarm may trigger a degraded driving mode);
ECE = expected calibration error of the predicted-class confidence, 15 bins; probabilities are
temperature-scaled on a validation split disjoint from the test split. Splits are by source
background image, so no scene appears in two splits.

SUGGESTED CAPTIONS
Fig. detector_degradation: (a) Attack success rate per attack profile (frequency band) and detector.
 (b) Mean overlap between each reference detection and the beam footprint, for suppressed vs
 retained detections. (c) ASR versus attack severity (laser power x coverage), sextile bins.
 Error bars/bands: 95% cluster-bootstrap CI over images.
Fig. detector_confidence: (a) Empirical CDF of clean-frame confidence for retained and suppressed
 detections. (b) False positives induced by the attack per 100 reference detections.
Fig. classifier_evaluation: (a) ROC curves (log-scaled false-alarm axis). (b) Reliability diagram
 before/after temperature scaling. (c) Correct-decision rate per attack profile; Wilson 95% CI.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="*", default=[], help="LABEL=path/to/box_level_results.csv ...")
    ap.add_argument("--classifier_dir", default=None)
    ap.add_argument("--out_dir", default="paper_figures")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iou_used", default="0.3", help="IoU threshold you ran detection with (for the definitions text)")
    ap.add_argument("--class_agnostic_used", action="store_true", help="set if you ran with --class_agnostic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_style()

    dets = {}
    for item in args.detectors:
        if "=" not in item:
            raise SystemExit(f"--detectors entries must look like LABEL=path.csv, got {item!r}")
        lab, path = item.split("=", 1)
        df = pd.read_csv(path)
        if "image_id" not in df.columns:
            raise SystemExit(f"{path} has no image_id column - rerun detection with the updated scripts")
        dets[lab] = df

    if dets:
        print("Detector figures...")
        tab, per_img = detector_tables(dets, args.n_boot, args.seed)
        tab.to_csv(os.path.join(args.out_dir, "table_detectors.csv"), index=False)
        with open(os.path.join(args.out_dir, "table_detectors.tex"), "w") as f:
            f.write(latex_detector_table(tab))
        fig_detector_degradation(tab, per_img, args.out_dir, args.n_boot, args.seed)
        fig_detector_confidence(tab, per_img, args.out_dir, args.n_boot, args.seed)
        print(tab[tab["profile"] == "All attacks"][["detector", "n_images", "n_baseline_objects", "ASR", "ASR_lo", "ASR_hi", "SAR"]]
              .round(3).to_string(index=False))

    if args.classifier_dir:
        preds = load_predictions(args.classifier_dir)
        if preds:
            print("Classifier figures...")
            fig_classifier(preds, args.out_dir)
            ab = os.path.join(args.classifier_dir, "ablation_table.csv")
            if os.path.isfile(ab):
                abl = pd.read_csv(ab, index_col=0)
                abl.to_csv(os.path.join(args.out_dir, "table_classifier.csv"))
                with open(os.path.join(args.out_dir, "table_classifier.tex"), "w") as f:
                    f.write(latex_classifier_table(abl))
        else:
            print(f"no predictions_test_*.csv in {args.classifier_dir}; skipping classifier figures")

    with open(os.path.join(args.out_dir, "captions_and_definitions.txt"), "w") as f:
        f.write(DEFINITIONS.format(iou=args.iou_used, cls="(class-agnostic)" if args.class_agnostic_used else "and the same class",
                                   ov=OVERLAP_THR, nb=args.n_boot))
    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
