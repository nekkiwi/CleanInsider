# file: train_walk_forward.py

import time
from src.training_pipeline import ModelTrainer

def main():
    print("--- Starting Walk-Forward Training Pipeline ---")
    start_time = time.time()

    strategies = [
        ('1w', 0.05, -0.05),
        # Add more if desired...
    ]

    binary_thresholds_pct = [0, 2, 5]
    model_type = "LightGBM"
    top_n_features = 50
    seeds = [42, 123, 2024]
    num_folds = 6 # This defines how many walk-forward validation steps to run

    trainer = ModelTrainer(num_folds=num_folds)

    # --- THIS IS THE FIX ---
    # The 'run' method knows where the test set is based on its internal paths.
    # We do not need to pass a 'test_fold' argument here.
    trainer.run(
        strategies=strategies,
        binary_thresholds_pct=binary_thresholds_pct,
        model_type=model_type,
        top_n=top_n_features,
        seeds=seeds
    )
    # --- END OF FIX ---

    end_time = time.time()
    print(f"\n--- Full Pipeline Complete in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
