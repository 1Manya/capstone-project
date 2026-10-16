"""
generate_paper_metrics.py
----------------------------
Computes publication-standard metrics from a box_level_results CSV
(must include the "image_id" column - from the updated
attack_effect_analysis_full.py / run_ssd_detection.py).

Metrics, using terminology standard in the adversarial-attack-on-vision
literature (matches Kohler et al.'s own "hide X% of objects" framing):

  ASR   - Attack Success Rate: % of baseline (clean-frame) detections
          suppressed under attack. This is Kohler et al.'s own metric.
  Recall_attack  - post-attack recall, treating clean-frame detections as
                   a pseudo-ground-truth reference (standard practice when
                   no human-annotated boxes exist)
  Precision_attack - post-attack precision, penalized by adversarially-
                     induced false positives
  F1_attack      - harmonic mean of the above
  dConf (ΔConf)  - mean confidence degradation on retained (matched)
                   detections
  SAR   - Spatial Attribution Rate: of SUPPRESSED detections, what
          fraction were spatially located inside the attack-affected
          region (vs. lost due to global frame degradation elsewhere)
  FPR_induced    - mean number of adversarially-induced false positives
                   per image

All stats are computed PER IMAGE first, then aggregated (mean ± std)
across images per variation - this is the statistically correct approach
for error bars; aggregating across individual boxes would inflate N and
understate variance.

Usage:
    python generate_paper_metrics.py \
        --box_csv ../../data/final_dataset/attack_effect_full_results/box_level_results_full.csv \
        --out_dir ../../data/final_dataset/attack_effect_full_results \
        --label yolov8m
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def per_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (var, img), g in df.groupby(["variation", "image_id"]):
        n_matched = len(g[g["outcome"] == "matched"])
        n_lost = len(g[g["outcome"] == "lost"])
        n_new = len(g[g["outcome"] == "new_false_positive"])
        n_baseline = n_matched + n_lost  # clean-frame detections for this image

        if n_baseline == 0:
            continue  # nothing to measure suppression against for this image

        asr = n_lost / n_baseline
        recall = n_matched / n_baseline
        precision = n_matched / (n_matched + n_new) if (n_matched + n_new) > 0 else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 and not np.isnan(precision) else np.nan)

        matched_rows = g[g["outcome"] == "matched"]
        dconf = matched_rows["conf_drop"].mean() if n_matched > 0 else np.nan

        lost_rows = g[g["outcome"] == "lost"]
        n_lost_in_region = len(lost_rows[lost_rows["attack_region_overlap"] >= 0.3])
        sar = n_lost_in_region / n_lost if n_lost > 0 else np.nan

        rows.append({
            "variation": var, "image_id": img,
            "ASR": asr, "Recall_attack": recall, "Precision_attack": precision,
            "F1_attack": f1, "dConf": dconf, "SAR": sar, "FP_induced": n_new,
        })
    return pd.DataFrame(rows)


def aggregate(per_img: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ASR", "Recall_attack", "Precision_attack", "F1_attack", "dConf", "SAR", "FP_induced"]
    rows = []
    for var, g in per_img.groupby("variation"):
        row = {"variation": var, "n_images": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex_table(agg: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Detection degradation under rolling-shutter laser attack, by variation.}",
        r"\label{tab:detection_degradation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variation & ASR (\%) & Recall & Precision & $\Delta$Conf & SAR (\%) \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{r['variation'].replace('_', ' ')} & "
            f"{r['ASR_mean']*100:.1f} $\\pm$ {r['ASR_std']*100:.1f} & "
            f"{r['Recall_attack_mean']:.3f} & "
            f"{r['Precision_attack_mean']:.3f} & "
            f"{r['dConf_mean']:.3f} & "
            f"{r['SAR_mean']*100:.1f} $\\pm$ {r['SAR_std']*100:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plot_combo(agg: pd.DataFrame, out_path: str, label: str):
    """Combo bar+line chart with error bars, same style as Sayles et al.
    Figure 5 (bars = ASR, line = Recall_attack, error bars = std across
    images)."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(agg))
    width = 0.5

    bars = ax1.bar(x, agg["ASR_mean"] * 100, width, yerr=agg["ASR_std"] * 100,
                    capsize=4, color="steelblue", label="Attack Success Rate (ASR)")
    ax1.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace("_", "\n") for v in agg["variation"]], fontsize=9)

    ax2 = ax1.twinx()
    ax2.errorbar(x, agg["Recall_attack_mean"], yerr=agg["Recall_attack_std"],
                 color="firebrick", marker="o", linewidth=2, capsize=4,
                 label="Post-Attack Recall")
    ax2.set_ylabel("Post-Attack Recall", fontsize=11)
    ax2.set_ylim(0, 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title(f"Detection Degradation Under Rolling-Shutter Laser Attack ({label})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Chart saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box_csv", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.box_csv)

    if "image_id" not in df.columns:
        raise ValueError(
            "This box_level_results CSV has no image_id column - it was generated "
            "before per-image tracking was added. Rerun detection with the updated "
            "attack_effect_analysis_full.py / run_ssd_detection.py."
        )

    per_img = per_image_stats(df)
    agg = aggregate(per_img)

    print(f"\n=== Publication Metrics ({args.label}) ===\n")
    display_cols = ["variation", "n_images", "ASR_mean", "ASR_std", "Recall_attack_mean",
                     "Precision_attack_mean", "F1_attack_mean", "dConf_mean", "SAR_mean", "FP_induced_mean"]
    print(agg[display_cols].to_string(index=False))

    out_csv = os.path.join(args.out_dir, f"publication_metrics_{args.label}.csv")
    agg.to_csv(out_csv, index=False)
    print(f"\nFull metrics saved to {out_csv}")

    latex = to_latex_table(agg)
    out_tex = os.path.join(args.out_dir, f"publication_table_{args.label}.tex")
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {out_tex}")

    out_chart = os.path.join(args.out_dir, f"publication_chart_{args.label}.png")
    plot_combo(agg, out_chart, args.label)


if __name__ == "__main__":
    main()
