from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import joblib

from src import config as cfg
from src.scrapers.feature_scraper.scrape_openinsider import scrape_openinsider_date_range
from src.scrapers.feature_scraper.load_annual_statements import generate_annual_statements
from src.scrapers.feature_scraper.load_technical_indicators import generate_technical_indicators
from src.scrapers.feature_scraper.build_event_ohlcv import build_event_ohlcv_datasets
from src.preprocess.fold_processor import FoldProcessor
from src.preprocess.utils import load_preprocessing_artifacts
from src.training_pipeline import ModelTrainer

from typing import Tuple


@dataclass
class BotArgs:
    timepoint: str
    tp: float
    sl: float
    threshold_pct: int
    lookback_days: int
    gate: float


def _get_strategy_tuple(timepoint: str, tp: float, sl: float) -> Tuple[str, float, float]:
    return (timepoint, float(tp), float(sl))


def _load_deploy_processor(num_folds: int) -> FoldProcessor:
    # Try direct preprocessing directory first (for deployed models)
    artifacts_dir = cfg.FEATURES_OUTPUT_PATH / 'preprocessing'
    if not artifacts_dir.exists() or not (artifacts_dir / 'scaler.pkl').exists():
        # Fallback to fold-specific directory for local development
        artifacts_dir = cfg.FEATURES_OUTPUT_PATH / 'preprocessing' / f'fold_{num_folds}'
    scaler, outlier_bounds, imputation_values, columns_info = load_preprocessing_artifacts(artifacts_dir)
    proc = FoldProcessor()
    proc.scaler = scaler
    proc.outlier_bounds = outlier_bounds or {}
    # imputation_values is a dict mapping column->median
    proc.imputation_values_for_unscaled = pd.Series(imputation_values or {})
    proc.columns_to_scale = columns_info.get('columns_to_scale', []) if columns_info else []
    proc.final_columns = columns_info.get('final_columns', []) if columns_info else []
    return proc


def _prepare_features_for_inference(raw_df: pd.DataFrame, processor: FoldProcessor, selected_features: list[str]) -> pd.DataFrame:
    processed = processor.transform(raw_df)
    common = [c for c in selected_features if c in processed.columns]
    return processed[common]


def _load_latest_model(trainer: ModelTrainer, strategy: tuple, threshold_pct: int | float):
    fold = trainer.num_folds - 2
    seed = 42
    classifier, _, metadata = trainer.load_model(strategy=strategy, fold=fold, seed=seed, threshold_pct=threshold_pct, model_type='classifier')
    if classifier is None:
        raise FileNotFoundError("Classifier model not found for the provided strategy/fold/seed.")
    imputation_values = pd.Series(metadata.get('imputation_values', {}))
    selected_features = metadata.get('selected_features', [])
    return classifier, selected_features, imputation_values, metadata.get('fold', fold)


def run_live_bot_once(args: BotArgs) -> pd.DataFrame:
    state_dir = Path('data') / 'bot'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / 'last_check.txt'
    end_date = datetime.utcnow().date()
    if state_path.exists():
        try:
            last = pd.to_datetime(state_path.read_text().strip()).date()
        except Exception:
            last = end_date - timedelta(days=args.lookback_days)
    else:
        last = end_date - timedelta(days=args.lookback_days)
    start_date = min(last, end_date - timedelta(days=args.lookback_days))

    base_df = scrape_openinsider_date_range(start_date, end_date)
    if base_df.empty:
        return pd.DataFrame()
    state_path.write_text(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))

    components_dir = cfg.FEATURES_COMPONENTS_PATH
    components_dir.mkdir(parents=True, exist_ok=True)

    annual_path = components_dir / 'inference_annuals.parquet'
    generate_annual_statements(base_df, str(annual_path), sec_parquet_dir=str(cfg.EDGAR_DOWNLOAD_PATH), request_header=cfg.REQUESTS_HEADER)
    annual_df = pd.read_parquet(annual_path) if annual_path.exists() else pd.DataFrame()

    try:
        build_event_ohlcv_datasets(
            base_df=base_df,
            db_path_str=str(cfg.STOOQ_DATABASE_PATH),
            past_output_path=Path(cfg.OHLCV_PAST_COMPONENT_PATH),
            future_output_path=Path(cfg.OHLCV_FUTURE_COMPONENT_PATH),
            past_lookback_calendar_days=400,
            future_lookahead_trading_days=126,
            n_jobs=-2,
        )
    except Exception:
        pass

    tech_path = components_dir / 'inference_technicals.parquet'
    generate_technical_indicators(base_df, str(cfg.STOOQ_DATABASE_PATH), str(tech_path))
    tech_df = pd.read_parquet(tech_path) if tech_path.exists() else pd.DataFrame()

    merged = base_df.copy()
    if not annual_df.empty:
        merged = pd.merge(merged, annual_df, on=["Ticker","Filing Date"], how="inner")
    if not tech_df.empty:
        merged = pd.merge(merged, tech_df, on=["Ticker","Filing Date"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    trainer = ModelTrainer(num_folds=5)
    strategy = _get_strategy_tuple(args.timepoint, args.tp, args.sl)
    classifier, selected_features, imputation_values, model_fold = _load_latest_model(trainer, strategy, args.threshold_pct)
    deploy_processor = _load_deploy_processor(num_folds=model_fold)

    # Ensure identifier columns are present for preprocessing and downstream joining/logging.
    for id_col in ['split_id', 'Ticker', 'Filing Date']:
        if id_col not in merged.columns:
            if id_col == 'split_id':
                merged[id_col] = 0
            # Ticker and Filing Date come from scrape; no default if missing
    # IMPORTANT: Do NOT drop identifiers before transform; the fitted scaler may expect 'split_id'.
    X_inf = _prepare_features_for_inference(merged, deploy_processor, selected_features)
    X_inf = X_inf.fillna(imputation_values)
    if X_inf.empty:
        return pd.DataFrame()
    proba = classifier.predict_proba(X_inf)[:, 1]
    out = merged.loc[X_inf.index].copy()
    out['pred_proba'] = proba
    print(out)
    # Primary: threshold gate
    if args.gate is not None:
        out = out[out['pred_proba'] >= float(args.gate)]
    return out


