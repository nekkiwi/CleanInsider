# file: src/scrape_targets.py

import time

# --- Import the main function from each step of the pipeline ---
# Note: This assumes you have renamed the files to be valid Python modules.
from scrapers.target_scraper.create_master_event_list import create_master_event_list
from scrapers.target_scraper.calculate_master_targets import calculate_master_targets
from scrapers.target_scraper.assemble_final_targets import assemble_final_targets

def run_target_generation_pipeline(config, target_combinations: list, n_splits: int = 7, batch_size: int = 100, debug: bool = False):
    """
    Main pipeline that orchestrates the three steps of target generation.
    This provides a single entry point for the entire target creation process.
    """
    print("\n--- Starting Assembled Target Generation Pipeline ---")
    start_time = time.time()
    
    # --- STEP 1: Create the master "to-do" list of all unique events ---
    # This is a fast operation that gathers all the work to be done.
    create_master_event_list(config, n_splits=n_splits)
    
    # --- STEP 2: Run the long, batch-processed calculation ---
    # This is the memory-efficient workhorse that can be restarted if it fails.
    calculate_master_targets(config, target_combinations, batch_size=batch_size, debug=debug)
    
    # --- STEP 3: Run the final, fast assembly step ---
    # This takes the calculated master targets and creates the final per-fold files.
    assemble_final_targets(config, n_splits=n_splits)
    
    end_time = time.time()
    print(f"\n--- Target Generation Pipeline Complete in {end_time - start_time:.2f} seconds ---")

