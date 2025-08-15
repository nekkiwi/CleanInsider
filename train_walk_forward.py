# file: train_walk_forward.py

import time
from src.training_pipeline import ModelTrainer

def main():
    print("--- Starting Walk-Forward Training Pipeline ---")
    start_time = time.time()

    strategies = [('1w', 0.10, -0.10),
                  ('1w', 0.15, -0.10)]
    binary_thresholds_pct = [0,0.5,1]
    model_type = "LightGBM"
    top_n_features = 10
    seeds = [42, 123, 2024, 456, 567]
    
    # This now correctly controls the number of walk-forward validation folds.
    # The final test set is handled automatically.
    num_folds = 6 

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
    main()
