# file: train_walk_forward.py

import argparse
import time

from src import config
from src.training_pipeline import ModelTrainer


def main(model_type: str = "LightGBM"):
    print(f"--- Starting Walk-Forward Training Pipeline ({model_type}) ---")
    start_time = time.time()

    # Strategy grid and training params are centralized in config so that
    # target generation and training never drift apart.
    strategies = config.STRATEGY_GRID
    binary_thresholds_pct = config.BINARY_THRESHOLDS_PCT
    top_n_features = config.TOP_N_FEATURES
    seeds = config.ENSEMBLE_SEEDS
    num_folds = config.NUM_VALIDATION_FOLDS

    trainer = ModelTrainer(num_folds=num_folds)
    trainer.run(
        strategies=strategies,
        binary_thresholds_pct=binary_thresholds_pct,
        model_type=model_type,
        top_n=top_n_features,
        seeds=seeds,
    )

    end_time = time.time()
    print(f"\n--- Full Pipeline Complete in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward model training.")
    parser.add_argument(
        "--model-type",
        default="LightGBM",
        help="Estimator family to train (e.g. LightGBM, TabPFN). Default: LightGBM.",
    )
    args = parser.parse_args()
    main(model_type=args.model_type)
