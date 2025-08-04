# file: train_walk_forward.py

import time
from src.training_pipeline import ModelTrainer

def main():
    """
    Main entry point to configure and run the walk-forward training and
    evaluation pipeline.
    """
    print("--- Starting Walk-Forward Training Pipeline ---")
    start_time = time.time()

    # --- Strategy & Model Configuration ---
    
    # Define the target strategies to model, as tuples of:
    # (lookahead_period, take_profit_%, stop_loss_%)
    strategies = [
        ('1w', 0.05, -0.05),
        # ('1m', 0.10, -0.10),
        # ('3m', 0.15, -0.15),
    ]
    
    # Define the thresholds (in percent) for converting continuous alpha targets
    # into binary classification targets (e.g., alpha > 2% is a "buy" signal).
    binary_thresholds_pct = [0, 2, 5] 

    model_type = "LightGBM"  # Supported: "RandomForest", "LightGBM"
    top_n_features = 50      # The number of features to select
    seeds = [42, 123, 2024]  # Use multiple seeds for robust results
    
    # --- Pipeline Execution ---
    # Initialize the trainer with the number of walk-forward folds (e.g., 6)
    model_trainer = ModelTrainer(n_splits=6)
    
    model_trainer.run(
        strategies=strategies,
        binary_thresholds_pct=binary_thresholds_pct,
        model_type=model_type,
        top_n=top_n_features,
        seeds=seeds
    )
    
    end_time = time.time()
    print(f"\n--- Full Pipeline Complete in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()

