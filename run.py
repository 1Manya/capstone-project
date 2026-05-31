"""
run.py
------
Single entry point for the entire pipeline.

Usage:
    python run.py --mode test
    python run.py --mode download
    python run.py --mode generate
    python run.py --mode train --model single
    python run.py --mode train --model ensemble
    python run.py --mode evaluate --model ensemble
    python run.py --mode compare
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add repo root to Python path so all imports work
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Laser Dazzle Attack Detection Pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["test", "download", "extract_nuscenes", "generate", "train", "evaluate", "compare"],
    )
    parser.add_argument("--model",         default="ensemble", choices=["single", "ensemble"])
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--nuscenes_root", default=None)
    parser.add_argument("--checkpoint",    default=None)
    parser.add_argument("--split",         default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    if args.mode == "download":
        from src.dataset.kaggle_downloader import download_kitti_from_kaggle
        download_kitti_from_kaggle(
            output_dir=cfg["paths"]["clean_base"],
            max_images=cfg["dataset"]["max_images"],
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
        from src.utils.extract_nuscenes import extract_nuscenes_frames
        extract_nuscenes_frames(
            nuscenes_root=args.nuscenes_root,
            output_dir=cfg["paths"]["clean_base"],
        )

    elif args.mode == "generate":
        print("\nStarting dataset generation...")
        from src.dataset.dataset_builder import build_dataset
        build_dataset(args.config)

    elif args.mode == "train":
        from src.training.train import train_single, train_ensemble
        if args.model == "single":
            train_single(cfg)
        else:
            train_ensemble(cfg)

    elif args.mode == "evaluate":
        import torch
        from src.dataset.dataloader import get_dataloaders, get_variation_names
        from src.evaluation.evaluate import evaluate_model, print_full_report, evaluate_per_variation

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
        from src.evaluation.evaluate import compare_single_vs_ensemble
        compare_single_vs_ensemble(cfg)


if __name__ == "__main__":
    main()
