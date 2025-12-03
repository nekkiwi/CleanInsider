# src/alpaca/google_drive.py
"""
Google Drive and Sheets integration for model storage and trade logging.
"""

import base64
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import config

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("[WARN] Google API not installed. Install google-api-python-client.")


class GoogleDriveClient:
    """
    Client for Google Drive and Sheets operations.
    
    Supports:
    - Model file upload/download (Drive API)
    - Trade logging to Google Sheets
    - Performance tracking in Google Sheets
    """
    
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets"
    ]
    
    # Sheet names in the logging spreadsheet
    TRADE_LOG_SHEET = "Trade Log"
    PERFORMANCE_SHEET = "Performance"
    
    def __init__(
        self,
        credentials_json: str = None,
        models_folder_id: str = None,
        log_sheet_id: str = None
    ):
        """
        Initialize Google Drive/Sheets client.
        
        Args:
            credentials_json: Service account credentials JSON (base64, path, or raw)
            models_folder_id: Google Drive folder ID for models
            log_sheet_id: Google Sheets spreadsheet ID for logging
        """
        self.models_folder_id = models_folder_id or config.GDRIVE_MODELS_FOLDER_ID
        self.log_sheet_id = log_sheet_id or config.GDRIVE_LOG_SHEET_ID
        self.drive_service = None
        self.sheets_service = None
        
        if not GDRIVE_AVAILABLE:
            return
        
        creds_input = credentials_json or config.GOOGLE_DRIVE_CREDENTIALS
        
        if not creds_input:
            print("[WARN] Google credentials not configured")
            return
        
        try:
            # Try to decode as base64 first (for GitHub Actions)
            try:
                creds_json = base64.b64decode(creds_input).decode("utf-8")
                creds_dict = json.loads(creds_json)
            except Exception:
                # Assume it's a file path or raw JSON
                if os.path.isfile(creds_input):
                    with open(creds_input, "r") as f:
                        creds_dict = json.load(f)
                else:
                    creds_dict = json.loads(creds_input)
            
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=self.SCOPES
            )
            
            self.drive_service = build("drive", "v3", credentials=credentials)
            self.sheets_service = build("sheets", "v4", credentials=credentials)
            print("[INFO] Google Drive/Sheets client initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Google client: {e}")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.drive_service is not None
    
    # =========================================================================
    # GOOGLE SHEETS LOGGING
    # =========================================================================
    
    def _ensure_sheet_exists(self, sheet_name: str) -> bool:
        """Ensure a sheet exists in the spreadsheet, create if not."""
        if not self.sheets_service or not self.log_sheet_id:
            return False
        
        try:
            # Get spreadsheet metadata
            spreadsheet = self.sheets_service.spreadsheets().get(
                spreadsheetId=self.log_sheet_id
            ).execute()
            
            # Check if sheet exists
            existing_sheets = [s["properties"]["title"] for s in spreadsheet.get("sheets", [])]
            
            if sheet_name in existing_sheets:
                return True
            
            # Create the sheet
            request = {
                "requests": [{
                    "addSheet": {
                        "properties": {"title": sheet_name}
                    }
                }]
            }
            self.sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=self.log_sheet_id,
                body=request
            ).execute()
            
            print(f"[INFO] Created sheet: {sheet_name}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to ensure sheet exists: {e}")
            return False
    
    def _get_sheet_row_count(self, sheet_name: str) -> int:
        """Get the current number of rows in a sheet."""
        if not self.sheets_service or not self.log_sheet_id:
            return 0
        
        try:
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.log_sheet_id,
                range=f"'{sheet_name}'!A:A"
            ).execute()
            
            values = result.get("values", [])
            return len(values)
        except Exception:
            return 0
    
    def _append_rows(self, sheet_name: str, rows: List[List], include_header: bool = False) -> bool:
        """Append rows to a sheet."""
        if not self.sheets_service or not self.log_sheet_id:
            return False
        
        try:
            # Ensure sheet exists
            if not self._ensure_sheet_exists(sheet_name):
                return False
            
            # Check if we need to add header
            current_rows = self._get_sheet_row_count(sheet_name)
            
            if current_rows == 0 and include_header and rows:
                # Sheet is empty, rows[0] should be the header
                pass
            elif current_rows > 0 and include_header and rows:
                # Sheet has data, skip the header row
                rows = rows[1:] if len(rows) > 1 else []
            
            if not rows:
                return True
            
            # Append data
            body = {"values": rows}
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=self.log_sheet_id,
                range=f"'{sheet_name}'!A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body
            ).execute()
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to append rows: {e}")
            return False
    
    def log_trades(self, trades_df: pd.DataFrame, model_id: str = "default") -> bool:
        """
        Log trades to Google Sheets.
        
        Appends trade data to the "Trade Log" sheet.
        
        Args:
            trades_df: DataFrame with trade info
            model_id: Identifier for which model generated these trades
            
        Returns:
            True if successful
        """
        if not self.sheets_service or not self.log_sheet_id:
            print("[WARN] Sheets not connected, skipping trade log")
            return False
        
        if trades_df.empty:
            return True
        
        try:
            # Prepare data for sheets
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Define columns for trade log (Model at END for backwards compatibility)
            header = [
                "Timestamp", "Date", "Ticker", "Shares", "Price", 
                "Dollar Amount", "Spread (bps)", "Position Size", 
                "Confidence", "Predicted Return", "Order Status", "Model"
            ]
            
            rows = [header]
            
            for _, row in trades_df.iterrows():
                trade_row = [
                    timestamp,
                    datetime.date.today().isoformat(),
                    str(row.get("Ticker", "")),
                    int(row.get("shares", 0)),
                    round(float(row.get("price", row.get("Price", 0))), 2),
                    round(float(row.get("dollar_size", 0)), 2),
                    round(float(row.get("spread_bps", 0)), 1) if "spread_bps" in row else "",
                    round(float(row.get("base_size", row.get("position_size", 0))), 3),
                    round(float(row.get("confidence", 0)), 3) if "confidence" in row else "",
                    round(float(row.get("predicted_return", 0)), 4) if "predicted_return" in row else "",
                    str(row.get("order_status", "")),
                    model_id  # Model at the end
                ]
                rows.append(trade_row)
            
            success = self._append_rows(self.TRADE_LOG_SHEET, rows, include_header=True)
            
            if success:
                print(f"[INFO] Logged {len(trades_df)} trades for model '{model_id}' to Google Sheets")
            
            return success
        except Exception as e:
            print(f"[ERROR] Failed to log trades: {e}")
            return False
    
    def log_performance(self, metrics: Dict, model_id: str = "default") -> bool:
        """
        Log performance metrics to Google Sheets.
        
        Appends a row to the "Performance" sheet.
        
        Args:
            metrics: Dict of performance metrics
            model_id: Identifier for which model this performance is for
            
        Returns:
            True if successful
        """
        if not self.sheets_service or not self.log_sheet_id:
            print("[WARN] Sheets not connected, skipping performance log")
            return False
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date = datetime.date.today().isoformat()
            
            # Define columns for performance log (Model at END for backwards compatibility)
            header = [
                "Timestamp", "Date", "Portfolio Value", "Cash", "Equity",
                "Num Trades", "Total Invested", "Num Positions",
                "Daily PnL", "Daily PnL %", "Notes", "Model"
            ]
            
            row = [
                timestamp,
                date,
                round(float(metrics.get("portfolio_value", 0)), 2),
                round(float(metrics.get("cash", 0)), 2),
                round(float(metrics.get("equity", 0)), 2),
                int(metrics.get("num_trades", 0)),
                round(float(metrics.get("total_invested", 0)), 2),
                int(metrics.get("num_positions", 0)),
                round(float(metrics.get("daily_pnl", 0)), 2) if "daily_pnl" in metrics else "",
                round(float(metrics.get("daily_pnl_pct", 0)), 4) if "daily_pnl_pct" in metrics else "",
                str(metrics.get("notes", "")),
                model_id  # Model at the end
            ]
            
            success = self._append_rows(self.PERFORMANCE_SHEET, [header, row], include_header=True)
            
            if success:
                print(f"[INFO] Logged performance for model '{model_id}' to Google Sheets")
            
            return success
        except Exception as e:
            print(f"[ERROR] Failed to log performance: {e}")
            return False
    
    # =========================================================================
    # GOOGLE DRIVE - MODEL STORAGE
    # =========================================================================
    
    def list_files(self, folder_id: str, file_type: str = None) -> List[Dict]:
        """List files in a folder."""
        if not self.drive_service:
            return []
        
        try:
            query = f"'{folder_id}' in parents and trashed = false"
            if file_type:
                query += f" and mimeType = '{file_type}'"
            
            results = self.drive_service.files().list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime, size)"
            ).execute()
            
            return results.get("files", [])
        except Exception as e:
            print(f"[ERROR] Failed to list files: {e}")
            return []
    
    def download_file(self, file_id: str, destination_path: Path) -> bool:
        """Download a file from Google Drive."""
        if not self.drive_service:
            return False
        
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            
            print(f"[INFO] Downloaded {destination_path.name}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to download file: {e}")
            return False
    
    def upload_file(self, source_path: Path, folder_id: str, file_name: str = None) -> Optional[str]:
        """Upload a file to Google Drive."""
        if not self.drive_service:
            return None
        
        if not source_path.exists():
            print(f"[ERROR] File not found: {source_path}")
            return None
        
        try:
            file_metadata = {
                "name": file_name or source_path.name,
                "parents": [folder_id]
            }
            
            media = MediaFileUpload(str(source_path), resumable=True)
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
            
            file_id = file.get("id")
            print(f"[INFO] Uploaded {source_path.name} -> {file_id}")
            return file_id
        except Exception as e:
            print(f"[ERROR] Failed to upload file: {e}")
            return None
    
    def download_models(self, local_models_path: Path = None, strategy: str = None) -> int:
        """
        Download models from Google Drive.
        
        Supports two folder structures:
        1. Flattened: model_xxx/weights/, model_xxx/preprocessing/
        2. Nested: strategy/fold_X/seed_Y/
        """
        if not self.drive_service or not self.models_folder_id:
            return 0
        
        local_path = local_models_path or config.MODELS_PATH
        local_path.mkdir(parents=True, exist_ok=True)
        
        items = self.list_files(self.models_folder_id)
        downloaded = 0
        
        for item in items:
            item_name = item["name"]
            
            # Skip items that don't match strategy filter
            if strategy and strategy not in item_name:
                continue
            
            if item["mimeType"] == "application/vnd.google-apps.folder":
                # Check if this is a model folder with weights/preprocessing structure
                sub_items = self.list_files(item["id"])
                sub_names = [s["name"] for s in sub_items]
                
                if "weights" in sub_names or "preprocessing" in sub_names:
                    # Flattened structure: model_xxx/weights/, model_xxx/preprocessing/
                    for sub_item in sub_items:
                        if sub_item["name"] == "weights":
                            # Download weights to models path, reconstructing fold structure
                            downloaded += self._download_flattened_weights(
                                sub_item["id"], 
                                local_path / self._model_name_to_strategy(item_name)
                            )
                        elif sub_item["name"] == "preprocessing":
                            # Download preprocessing to preprocessing path
                            preproc_path = config.PREPROCESSING_ARTIFACTS_PATH
                            preproc_path.mkdir(parents=True, exist_ok=True)
                            downloaded += self._download_folder_recursive(sub_item["id"], preproc_path)
                else:
                    # Nested structure: strategy/fold_X/seed_Y/
                    downloaded += self._download_folder_recursive(item["id"], local_path / item_name)
            else:
                if self.download_file(item["id"], local_path / item_name):
                    downloaded += 1
        
        print(f"[INFO] Downloaded {downloaded} model files")
        return downloaded
    
    def _model_name_to_strategy(self, model_name: str) -> str:
        """Convert model name like 'model_1w_tp5_sl5' to strategy '1w_tp0p05_sl-0p05'."""
        # Map common patterns
        mappings = {
            "model_1w_tp5_sl5": "1w_tp0p05_sl-0p05",
            "model_2w_tp5_sl5": "2w_tp0p05_sl-0p05",
            "model_1m_tp5_sl5": "1m_tp0p05_sl-0p05",
        }
        return mappings.get(model_name, model_name)
    
    def _download_flattened_weights(self, folder_id: str, local_folder: Path) -> int:
        """
        Download flattened weight files and reconstruct fold/seed structure.
        
        Files like 'fold1_seed42_classifier.pkl' go to 'fold_1/seed_42/classifier.pkl'
        """
        items = self.list_files(folder_id)
        downloaded = 0
        
        for item in items:
            if item["mimeType"] == "application/vnd.google-apps.folder":
                continue
                
            name = item["name"]
            
            # Parse filename: fold1_seed42_classifier.pkl -> fold_1/seed_42/classifier.pkl
            if name.startswith("fold") and "_seed" in name:
                parts = name.split("_")
                fold_num = parts[0].replace("fold", "")
                seed_num = parts[1].replace("seed", "")
                file_name = "_".join(parts[2:])  # Rest is the filename
                
                target_path = local_folder / f"fold_{fold_num}" / f"seed_{seed_num}" / file_name
            else:
                target_path = local_folder / name
            
            if self.download_file(item["id"], target_path):
                downloaded += 1
        
        return downloaded
    
    def _download_folder_recursive(self, folder_id: str, local_folder: Path) -> int:
        """Recursively download folder contents."""
        items = self.list_files(folder_id)
        downloaded = 0
        
        for item in items:
            if item["mimeType"] == "application/vnd.google-apps.folder":
                downloaded += self._download_folder_recursive(item["id"], local_folder / item["name"])
            else:
                if self.download_file(item["id"], local_folder / item["name"]):
                    downloaded += 1
        
        return downloaded
    
    def upload_models(self, local_models_path: Path = None, strategy: str = None) -> int:
        """Upload models to Google Drive."""
        if not self.drive_service or not self.models_folder_id:
            return 0
        
        local_path = local_models_path or config.MODELS_PATH
        
        if strategy:
            local_path = local_path / strategy
        
        if not local_path.exists():
            print(f"[ERROR] Models path not found: {local_path}")
            return 0
        
        uploaded = self._upload_folder_recursive(local_path, self.models_folder_id)
        print(f"[INFO] Uploaded {uploaded} model files")
        return uploaded
    
    def _upload_folder_recursive(self, local_folder: Path, parent_folder_id: str) -> int:
        """Recursively upload folder contents."""
        uploaded = 0
        
        for item in local_folder.iterdir():
            if item.is_dir():
                folder_id = self._create_folder(item.name, parent_folder_id)
                if folder_id:
                    uploaded += self._upload_folder_recursive(item, folder_id)
            else:
                if self.upload_file(item, parent_folder_id):
                    uploaded += 1
        
        return uploaded
    
    def _create_folder(self, folder_name: str, parent_folder_id: str) -> Optional[str]:
        """Create a folder in Google Drive."""
        try:
            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_folder_id]
            }
            
            folder = self.drive_service.files().create(
                body=file_metadata,
                fields="id"
            ).execute()
            
            return folder.get("id")
        except Exception as e:
            print(f"[ERROR] Failed to create folder: {e}")
            return None
