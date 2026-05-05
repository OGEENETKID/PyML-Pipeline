import logging
import numpy as np
from typing import Dict, Any
 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix,
)
 
log = logging.getLogger(__name__)
 
CLASSIFIERS = {
    "random_forest":        RandomForestClassifier,
    "logistic_regression":  LogisticRegression,
    "gradient_boosting":    GradientBoostingClassifier,
    "svm":                  SVC,
    "knn":                  KNeighborsClassifier,
}
 
REGRESSORS = {
    "random_forest":        RandomForestRegressor,
    "linear_regression":    LinearRegression,
    "ridge":                Ridge,
    "svr":                  SVR,
}
 
 
def run_evaluation(cfg: Dict[str, Any], features: Dict) -> Dict[str, Any]:
    if features is None:
        raise RuntimeError("No feature data provided to evaluation step.")
 
    model_cfg = cfg["model"]
    task      = model_cfg.get("task", "classification")
    algo      = model_cfg.get("algorithm", "random_forest")
    params    = model_cfg.get("params", {})
 
    X_train = features["X_train"]
    X_test  = features["X_test"]
    y_train = features["y_train"]
    y_test  = features["y_test"]
 
    log.info(f"Task: {task}  |  Algorithm: {algo}")
    model = _build_model(task, algo, params)
 
    log.info("Training model…")
    model.fit(X_train, y_train)
 
    log.info("Evaluating on test set…")
    y_pred = model.predict(X_test)
 
    if task == "classification":
        metrics = _classification_metrics(y_test, y_pred)
    else:
        metrics = _regression_metrics(y_test, y_pred)
 
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info(f"  {k:25s}: {v:.4f}")
 
    return {
        "model":   model,
        "metrics": metrics,
        "y_pred":  y_pred,
        "y_test":  y_test,
        "task":    task,
    }
 
 
def _build_model(task, algo, params):
    registry = CLASSIFIERS if task == "classification" else REGRESSORS
    if algo not in registry:
        raise ValueError(f"Unknown algorithm '{algo}' for task '{task}'. "
                         f"Available: {list(registry)}")
    return registry[algo](**params)
 
 
def _classification_metrics(y_true, y_pred) -> Dict:
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "report":    classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
 
 
def _regression_metrics(y_true, y_pred) -> Dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "mse":  mse,
        "rmse": np.sqrt(mse),
    }
 
