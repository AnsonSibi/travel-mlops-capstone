"""Model monitoring helpers: prediction logging and data-drift detection (PSI).

The API appends every request + prediction to logs/predictions.jsonl. The
Airflow monitoring DAG (or `python -m travel_ml.monitoring`) compares those
live inputs against the training distribution saved at training time.

PSI (Population Stability Index) rule of thumb:
    < 0.1  no significant change
    0.1-0.2 moderate shift - keep an eye on it
    > 0.2  significant drift - retrain
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from travel_ml import config

logger = logging.getLogger(__name__)

PREDICTION_LOG = config.LOGS_DIR / "predictions.jsonl"
DRIFT_REPORT = config.REPORTS_DIR / "drift_report.json"
PSI_ALERT = 0.2
EPS = 1e-6


def build_reference_stats(X: pd.DataFrame, bins: int = 10) -> dict:
    """Store the training distribution of every feature (bin edges + shares)."""
    stats = {"numeric": {}, "categorical": {}, "n_rows": int(len(X))}
    for col in config.NUMERIC_FEATURES:
        edges = np.unique(np.quantile(X[col], np.linspace(0, 1, bins + 1)))
        counts, _ = np.histogram(X[col], bins=edges)
        stats["numeric"][col] = {"edges": edges.tolist(), "shares": (counts / counts.sum()).tolist(),
                                 "mean": float(X[col].mean()), "std": float(X[col].std())}
    for col in config.CATEGORICAL_FEATURES:
        stats["categorical"][col] = X[col].value_counts(normalize=True).to_dict()
    return stats


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_drift(reference: dict, live: pd.DataFrame) -> dict:
    """PSI for each feature between the reference stats and live inputs."""
    features = {}
    for col, ref in reference["numeric"].items():
        if col not in live:
            continue
        edges = np.array(ref["edges"], dtype=float)
        values = np.clip(live[col].astype(float), edges[0], edges[-1])
        counts, _ = np.histogram(values, bins=edges)
        features[col] = psi(ref["shares"], counts / max(counts.sum(), 1))
    for col, ref in reference["categorical"].items():
        if col not in live:
            continue
        shares = live[col].value_counts(normalize=True)
        cats = sorted(set(ref) | set(shares.index))
        features[col] = psi([ref.get(c, 0) for c in cats], [shares.get(c, 0) for c in cats])

    drifted = [c for c, v in features.items() if v > PSI_ALERT]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_live_rows": int(len(live)),
        "psi": {k: round(v, 4) for k, v in features.items()},
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "threshold": PSI_ALERT,
    }


def log_prediction(features: dict, prediction: float, model_version: str, path: Path = PREDICTION_LOG):
    """Append one prediction to the JSONL log (cheap, append-only)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model_version": model_version, "prediction": prediction, **features}
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:  # logging must never break serving
        logger.warning("could not write prediction log: %s", exc)


def load_prediction_log(path: Path = PREDICTION_LOG, last_n: int | None = None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    return df.tail(last_n) if last_n else df


def run_drift_check(log_path: Path = PREDICTION_LOG, min_rows: int = 50) -> dict:
    reference = json.loads(config.REFERENCE_STATS_PATH.read_text())
    live = load_prediction_log(log_path)
    if len(live) < min_rows:
        report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "n_live_rows": int(len(live)), "drift_detected": False,
                  "note": f"not enough live rows (need {min_rows})"}
    else:
        report = compute_drift(reference, live)
    DRIFT_REPORT.write_text(json.dumps(report, indent=2))
    logger.info("drift report: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a data-drift check on logged predictions")
    parser.add_argument("--log", default=str(PREDICTION_LOG))
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(run_drift_check(Path(args.log), args.min_rows), indent=2))
