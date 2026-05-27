# file: src/scrape_targets.py

import time

from src.scrapers.target_scraper.assemble_final_targets import assemble_final_targets
from src.scrapers.target_scraper.calculate_master_targets import (
    calculate_master_targets,
)
from src.scrapers.target_scraper.calculate_spread_estimator import (
    generate_spread_estimates,
)
from src.scrapers.target_scraper.create_master_event_list import (
    create_master_event_list,
)


def run_target_generation_pipeline(
    config,
    target_combinations: list,
    n_splits: int = 7,
    batch_size: int = 100,
    debug: bool = False,
    recompute: bool = True,
):
    """
    Orchestrates the four steps of target generation.

    Args:
        config: Project config module.
        target_combinations: List of {time, tp, sl} dicts to label.
        n_splits: Total time splits (includes the held-out test set).
        batch_size: Event batch size for the (restartable) target calculation.
        debug: Verbose target calculation.
        recompute: When True (default) rebuild the master event list, recompute
            targets, and reassemble per-fold labels. When False, skip straight to
            the spread-estimate step (assumes targets already exist on disk).
    """
    print("\n--- Starting Assembled Target Generation Pipeline ---")
    start_time = time.time()

    if recompute:
        # --- STEP 1: Create the master "to-do" list of all unique events ---
        # Fast operation that gathers all the work to be done.
        create_master_event_list(config, n_splits=n_splits)

        # --- STEP 2: Run the long, batch-processed calculation ---
        # Memory-efficient workhorse that can be restarted if it fails.
        calculate_master_targets(
            config, target_combinations, batch_size=batch_size, debug=debug
        )

        # --- STEP 3: Run the final, fast assembly step ---
        # Takes the calculated master targets and creates the per-fold label files.
        assemble_final_targets(config, n_splits=n_splits)
    else:
        print("--- Steps 1-3 skipped (recompute=False): regenerating spreads only ---")

    # --- STEP 4: Generate spread estimates ---
    generate_spread_estimates(
        master_events_path=config.MASTER_EVENT_LIST_PATH,
        ohlcv_db_path=config.STOOQ_DATABASE_PATH,
        targets_base_path=config.TARGETS_OUTPUT_PATH,
        num_folds=n_splits - 1,  # n_splits includes the test set
    )

    end_time = time.time()
    print(
        f"\n--- Target Generation Pipeline Complete in {end_time - start_time:.2f} seconds ---"
    )
