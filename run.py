"""
run.py
------
Single entry point for the entire pipeline.

WHAT CHANGED FROM THE PREVIOUS VERSION:
1. `--mode generate` was calling `from src.dataset.dataset_builder import
   build_dataset` - that function didn't exist (dataset_builder.py only had
   a CLI `main()`). This would have raised ImportError the moment you ran
   `python run.py --mode generate`. Fixed by adding `build_dataset(config_path,
   seed)` to dataset_builder.py (see that file) - this now works.
2. Added `--seed` (default 42), passed through to generate mode, matching
   dataset_builder.py's own CLI default.
3. `download` / `extract_nuscenes` / `evaluate` / `compare` modes reference
   modules that don't exist yet in this pipeline (kaggle_downloader.py,
   extract_nuscenes.py, evaluation/evaluate.py) - not needed to reach
   training, so not built here. Wrapped in a friendly ImportError message
   instead of a raw traceback, so if you reach for one of these before it's
   built, the error tells you what's missing instead of just crashing.

Usage:
    python run.py --mode test
    python run.py --mode generate --seed 42
    python run.py --mode train --model single
    python run.py --mode train --model ensemble
    python run.py --mode evaluate --model ensemble   (not yet implemented)
    python run.py --mode compare                      (not yet implemented)
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add repo root to Python path so all imports work regardless of cwd
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _not_implemented(mode: str, missing_module: str):
    print(f"\n--mode {mode} isn't implemented yet in this pipeline - it needs "
          f"'{missing_module}', which hasn't been built. This isn't required "
          f"to reach training (generate -> train works today); it's a later step.\n")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Laser Dazzle Attack Detection Pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["test", "download", "extract_nuscenes", "generate", "train", "evaluate", "compare"],
    )
    parser.add_argument("--model",         default="ensemble", choices=["single", "ensemble"])
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--nuscenes_root", default=None)
    parser.add_argument("--checkpoint",    default=None)
    parser.add_argument("--split",         default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    if args.mode == "download":
        try:
            from src.dataset.kaggle_downloader import download_kitti_from_kaggle
        except ImportError:
            _not_implemented("download", "src/dataset/kaggle_downloader.py")
        download_kitti_from_kaggle(
            output_dir=cfg["paths"]["clean_base"],
            max_images=cfg.get("dataset", {}).get("max_images", 10000),
        )

    elif args.mode == "test":
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "test_pipeline",
            pathlib.Path(__file__).parent / "tests" / "test_pipeline.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run_all_checks = mod.run_all_checks
        success = run_all_checks()
        sys.exit(0 if success else 1)

    elif args.mode == "extract_nuscenes":
        if args.nuscenes_root is None:
            print("ERROR: --nuscenes_root is required.")
            sys.exit(1)
        try:
            from src.utils.extract_nuscenes import extract_nuscenes_frames
        except ImportError:
            _not_implemented("extract_nuscenes", "src/utils/extract_nuscenes.py")
        extract_nuscenes_frames(
            nuscenes_root=args.nuscenes_root,
            output_dir=cfg["paths"]["clean_base"],
        )

    elif args.mode == "generate":
        print("\nStarting dataset generation...")
        from src.dataset.dataset_builder import build_dataset
        build_dataset(args.config, args.seed)

    elif args.mode == "train":
        from src.training.train import train_single, train_ensemble
        if args.model == "single":
            train_single(cfg)
        else:
            train_ensemble(cfg)

    elif args.mode == "evaluate":
        try:
            from src.evaluation.evaluate import evaluate_model, print_full_report, evaluate_per_variation
        except ImportError:
            _not_implemented("evaluate", "src/evaluation/evaluate.py")
        import torch
        from src.dataset.dataloader import get_dataloaders, get_variation_names

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaders  = get_dataloaders(cfg, variation_filter=None)
        ckpt_dir = cfg["paths"]["checkpoints"]

        if args.model == "single":
            from src.models.cnn import build_model
            ckpt  = args.checkpoint or os.path.join(ckpt_dir, "single_cnn_best.pth")
            model = build_model(cfg)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device).eval()
            m = evaluate_model(model, loaders[args.split], device, is_ensemble=False)
            print_full_report(m, "Single CNN", args.split)
            evaluate_per_variation(model, cfg, args.split, device, is_ensemble=False)
        else:
            from src.models.ensemble import load_ensemble
            ensemble = load_ensemble(get_variation_names(cfg), ckpt_dir, cfg, device)
            m = evaluate_model(ensemble, loaders[args.split], device, is_ensemble=True)
            print_full_report(m, "Ensemble CNN", args.split)
            evaluate_per_variation(ensemble, cfg, args.split, device, is_ensemble=True)

    elif args.mode == "compare":
        try:
            from src.evaluation.evaluate import compare_single_vs_ensemble
        except ImportError:
            _not_implemented("compare", "src/evaluation/evaluate.py")
        compare_single_vs_ensemble(cfg)


if __name__ == "__main__":
    main()