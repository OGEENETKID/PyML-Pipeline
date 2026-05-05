import yaml
from pathlib import Path
from typing import Any, Dict
 
 
REQUIRED_KEYS = ["pipeline", "data", "features", "model"]
 
 
def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
 
    _validate(cfg, path)
    return cfg
 
 
def _validate(cfg: Dict, path: Path):
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise ValueError(f"Config '{path}' missing required section: '{key}'")
 
    if "source" not in cfg["data"]:
        raise ValueError("data.source is required")
 
    if "target" not in cfg["data"]:
        raise ValueError("data.target is required")
