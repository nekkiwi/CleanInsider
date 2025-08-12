# file: src/scrape_targets.py

import time
from tqdm import tqdm

# --- Import the main function from each step of the pipeline ---
# Note: This assumes you have renamed the files to be valid Python modules.
from src.scrapers.target_scraper.create_master_event_list import create_master_event_list
from src.scrapers.target_scraper.calculate_master_targets import calculate_master_targets
from src.scrapers.target_scraper.assemble_final_targets import assemble_final_targets
# Note: Spread estimation step removed; spreads are now handled in targets directly

def run_target_generation_pipeline(config, target_combinations: list, n_splits: int = 7, batch_size: int = 100, debug: bool = False):
    """
    Main pipeline that orchestrates the three steps of target generation.
    This provides a single entry point for the entire target creation process.
    """
    print("\n--- Starting Assembled Target Generation Pipeline ---")
    start_time = time.time()
    
    # Main pipeline progress bar (few steps) should persist
    with tqdm(total=3, desc="Target Generation Pipeline", leave=True) as main_pbar:
        # --- STEP 1: Create the master "to-do" list of all unique events ---
        # This is a fast operation that gathers all the work to be done.
        create_master_event_list(config, n_splits=n_splits)
        main_pbar.update(1)

        # --- STEP 2: Run the long, batch-processed calculation ---
        # This is the memory-efficient workhorse that can be restarted if it fails.
        calculate_master_targets(config, target_combinations, batch_size=batch_size, debug=debug)
        main_pbar.update(1)

        # --- STEP 3: Run the final, fast assembly step ---
        # This takes the calculated master targets and creates the final per-fold files.
        assemble_final_targets(config, n_splits=n_splits)
        main_pbar.update(1)

    # STEP 4 removed (legacy spread estimator). Spreads are integrated into targets.
    
    end_time = time.time()
    print(f"\n--- Target Generation Pipeline Complete in {end_time - start_time:.2f} seconds ---")

