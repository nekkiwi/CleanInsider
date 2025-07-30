# file: download_data.py

from pathlib import Path

from secfsdstools.a_config.configmodel import Configuration
from secfsdstools.update import update


def setup_and_update_database():
    """
    Downloads and processes all SEC Filings into a local database,
    including an optimized Parquet version for fast reading.
    """
    # Define the project directory
    project_dir = Path.cwd()

    # Define the paths for the different parts of the database within your project
    db_path = project_dir / "../../data/sec_database/database"
    dld_path = project_dir / "../../data/sec_database/downloads"
    parquet_path = project_dir / "../../data/sec_database/parquet"

    print("--- Local SEC Database Setup ---")
    print(f"Project Directory: {project_dir}")
    print(f"Index/JSONs (db_dir) will be at: {db_path}")
    print(f"ZIP Downloads (download_dir) will be at: {dld_path}")
    print(f"Fast Parquet files (parquet_dir) will be at: {parquet_path}")
    print("-" * 30)

    # Create the configuration object.
    # This defines where all the files will be stored.
    config = Configuration(
        db_dir=str(db_path),
        download_dir=str(dld_path),
        parquet_dir=str(parquet_path),  # <-- This is key for fast reading
        user_agent_email="your.name@yourdomain.com",  # <-- IMPORTANT: Change this
    )

    print("🚀 Starting the download and processing...")
    print("This will take a very long time for the first run, please be patient.")

    # This single function handles everything:
    # 1. Downloads the zip files to download_dir.
    # 2. Unzips the data into db_dir.
    # 3. Creates the SQLite index in db_dir.
    # 4. Exports the data into fast Parquet files in parquet_dir.
    update(config)

    print("\n✅ Database download and processing complete.")
    print("You can now run the 'analyze_filings.py' script.")


if __name__ == "__main__":
    # Before running, make sure to delete any old database folders to start fresh.
    print("ATTENTION: Ensure you have deleted any old 'data/sec_database' folders.")
    input("Press Enter to continue...")
    setup_and_update_database()
