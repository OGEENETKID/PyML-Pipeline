import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
 
log = logging.getLogger(__name__)
 
 
def run_etl(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Run the full ETL process. Returns (X, y)."""
    data_cfg = cfg["data"]
 
    log.info(f"Extracting from: {data_cfg['source']}")
    df = _extract(data_cfg)
    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
 
    log.info("Transforming data…")
    df = _transform(df, data_cfg)
    log.info(f"  After transform: {len(df):,} rows")
 
    target = data_cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
 
    X = df.drop(columns=[target])
    y = df[target]
 
    log.info(f"  Features: {list(X.columns)}")
    log.info(f"  Target  : {target} (unique values: {y.nunique()})")
 
    return X, y
 
 
def _extract(data_cfg: Dict) -> pd.DataFrame:
    source = data_cfg["source"]
    fmt = data_cfg.get("format", "csv").lower()
 
    readers = {
        "csv":     pd.read_csv,
        "parquet": pd.read_parquet,
        "json":    pd.read_json,
        "excel":   pd.read_excel,
    }
 
    if fmt not in readers:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(readers)}")
 
    kwargs = data_cfg.get("read_kwargs", {})
    return readers[fmt](source, **kwargs)
 
 
def _transform(df: pd.DataFrame, data_cfg: Dict) -> pd.DataFrame:
    # Drop configured columns
    drop_cols = data_cfg.get("drop_columns", [])
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        log.debug(f"  Dropped columns: {drop_cols}")
 
    # Drop rows with too many nulls
    null_threshold = data_cfg.get("null_row_threshold", 0.5)
    before = len(df)
    df = df.dropna(thresh=int(len(df.columns) * (1 - null_threshold)))
    dropped = before - len(df)
    if dropped:
        log.debug(f"  Dropped {dropped} rows with >{null_threshold*100:.0f}% nulls")
 
    # Rename columns
    rename_map = data_cfg.get("rename_columns", {})
    if rename_map:
        df = df.rename(columns=rename_map)
 
    return df.reset_index(drop=True)
