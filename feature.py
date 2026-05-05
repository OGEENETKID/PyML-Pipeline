import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
 
log = logging.getLogger(__name__)
 
 
def run_feature_engineering(
    cfg: Dict[str, Any],
    data: Tuple[pd.DataFrame, pd.Series],
) -> Dict[str, Any]:
    """Apply feature engineering steps from config. Returns train/test splits."""
    if data is None:
        raise RuntimeError("No data provided to feature engineering step.")
 
    X, y = data
    feat_cfg = cfg["features"]
 
    log.info("Encoding categorical features…")
    X = _encode_categoricals(X, feat_cfg)
 
    log.info("Imputing missing values…")
    X = _impute(X, feat_cfg)
 
    log.info("Scaling numeric features…")
    X, scaler = _scale(X, feat_cfg)
 
    log.info("Splitting train / test…")
    split_cfg = feat_cfg.get("split", {})
    test_size  = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)
    stratify = y if split_cfg.get("stratify", False) else None
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
 
    log.info(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
 
    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": list(X.columns),
    }
 
 
def _encode_categoricals(X: pd.DataFrame, feat_cfg: Dict) -> pd.DataFrame:
    strategy = feat_cfg.get("encoding", "label")
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
 
    if not cat_cols:
        return X
 
    X = X.copy()
    if strategy == "onehot":
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        log.debug(f"  One-hot encoded {len(cat_cols)} columns → {len(X.columns)} cols")
    else:
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))
        log.debug(f"  Label encoded: {cat_cols}")
 
    return X
 
 
def _impute(X: pd.DataFrame, feat_cfg: Dict) -> pd.DataFrame:
    num_strategy = feat_cfg.get("impute_numeric", "mean")
    cat_strategy = feat_cfg.get("impute_categorical", "most_frequent")
 
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
 
    X = X.copy()
    if num_cols:
        imp = SimpleImputer(strategy=num_strategy)
        X[num_cols] = imp.fit_transform(X[num_cols])
 
    if cat_cols:
        imp = SimpleImputer(strategy=cat_strategy)
        X[cat_cols] = imp.fit_transform(X[cat_cols])
 
    return X
 
 
def _scale(X: pd.DataFrame, feat_cfg: Dict):
    method = feat_cfg.get("scaling", "standard")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
 
    scalers = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "none":     None,
    }
 
    scaler = scalers.get(method)
    if scaler is None or not num_cols:
        return X, None
 
    X = X.copy()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, scaler
