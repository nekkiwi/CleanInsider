import os
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    import gspread
    from gspread_dataframe import set_with_dataframe
except Exception:
    gspread = None
    set_with_dataframe = None


def log_to_google_sheets(signals_df: pd.DataFrame, sheet_title: str = 'InsiderAlgoBot_Logs') -> bool:
    if signals_df is None or signals_df.empty:
        return False
    if gspread is None:
        return False
    creds_json = os.getenv('GOOGLE_SHEET_CREDS_JSON')
    if not creds_json:
        return False
    import json
    sa = gspread.service_account_from_dict(json.loads(creds_json))
    try:
        sh = sa.open(sheet_title)
    except gspread.SpreadsheetNotFound:
        sh = sa.create(sheet_title)
    ws = sh.sheet1
    df = signals_df.copy()
    df.insert(0, 'timestamp_utc', pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
    if set_with_dataframe is None:
        outdir = Path('results') / 'bot'
        outdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        (outdir / f'logs_{ts}.csv').write_text(df.to_csv(index=False))
        return False
    existing = ws.get_all_values()
    if len(existing) <= 1:
        set_with_dataframe(ws, df, include_column_header=True)
    else:
        start_row = len(existing) + 1
        set_with_dataframe(ws, df, row=start_row, include_column_header=False)
    return True


