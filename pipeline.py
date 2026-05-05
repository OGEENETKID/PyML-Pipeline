import argparse
import logging
import sys
import time
from pathlib import Path
 
from pyml.config import load_config
from pyml.etl import run_etl
from pyml.features import run_feature_engineering
from pyml.evaluate import run_evaluation
from pyml.report import generate_report
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pyml")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="PyML Pipeline — Modular ML pipeline runner"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["etl", "features", "evaluate", "report"],
        default=["etl", "features", "evaluate", "report"],
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()
 
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
 
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
 
    log.info(f"Loading config: {config_path}")
    cfg = load_config(config_path)
 
    start = time.time()
    log.info(f"Starting pipeline: {cfg['pipeline']['name']}")
 
    artifacts = {}
 
    if "etl" in args.steps:
        log.info("── Step 1/4: ETL ──────────────────────────")
        artifacts["data"] = run_etl(cfg)
 
    if "features" in args.steps:
        log.info("── Step 2/4: Feature Engineering ──────────")
        artifacts["features"] = run_feature_engineering(cfg, artifacts.get("data"))
 
    if "evaluate" in args.steps:
        log.info("── Step 3/4: Model Evaluation ──────────────")
        artifacts["results"] = run_evaluation(cfg, artifacts.get("features"))
 
    if "report" in args.steps:
        log.info("── Step 4/4: Report Generation ─────────────")
        generate_report(cfg, artifacts)
 
    elapsed = time.time() - start
    log.info(f"Pipeline complete in {elapsed:.2f}s ✓")
 
 
if __name__ == "__main__":
    main()
