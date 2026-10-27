"""CI quality gate: fail the build if the serving model is below the agreed bar.

    python scripts/check_model_quality.py --min-r2 0.95 --max-rmse 50
"""
import argparse
import json
import sys

from travel_ml import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-r2", type=float, default=0.95)
    parser.add_argument("--max-rmse", type=float, default=50.0)
    args = parser.parse_args()

    meta = json.loads(config.REGRESSION_META_PATH.read_text())
    m = meta["metrics"]
    print(f"model={meta['model_name']} run={meta['mlflow_run_id']} rmse={m['rmse']:.3f} r2={m['r2']:.5f}")
    failures = []
    if m["r2"] < args.min_r2:
        failures.append(f"R2 {m['r2']:.4f} < {args.min_r2}")
    if m["rmse"] > args.max_rmse:
        failures.append(f"RMSE {m['rmse']:.2f} > {args.max_rmse}")
    if failures:
        print("QUALITY GATE FAILED:", "; ".join(failures))
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
