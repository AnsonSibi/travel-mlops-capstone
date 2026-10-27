"""Train, compare and register flight price regression models with MLflow.

Usage:
    python -m travel_ml.train_regression            # train all candidates
    python -m travel_ml.train_regression --models random_forest

Each candidate becomes one MLflow run (params, metrics, model artefact).
The best run by test RMSE is registered in the MLflow Model Registry and gets
the "champion" alias only if it beats the current champion. The winning
pipeline is also exported to models/ as a joblib file for the Flask API.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from travel_ml import config
from travel_ml.data import load_flights, validate
from travel_ml.features import build_preprocessor, split_xy
from travel_ml.monitoring import build_reference_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_regression")


def candidate_models() -> dict:
    """Model name -> (estimator, needs_scaling)."""
    return {
        "ridge_baseline": (Ridge(alpha=1.0), True),
        "decision_tree": (DecisionTreeRegressor(max_depth=20, min_samples_leaf=5,
                                                random_state=config.RANDOM_STATE), False),
        "random_forest": (RandomForestRegressor(n_estimators=60, max_depth=20, min_samples_leaf=5,
                                                n_jobs=-1, random_state=config.RANDOM_STATE), False),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.2,
                                                                 max_leaf_nodes=63,
                                                                 random_state=config.RANDOM_STATE), False),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def route_holdout_metrics(pipe, X, y, n_splits=3, sample=60000) -> dict:
    """Generalisation to routes the model has never seen (GroupKFold on route).

    A random split puts every route in both train and test, so a tree can just
    memorise route prices. Holding out whole routes tells us how the model
    would price a brand new route - an important robustness check.
    """
    Xs = X.sample(min(sample, len(X)), random_state=config.RANDOM_STATE)
    ys = y.loc[Xs.index]
    routes = Xs[["from", "to"]].apply(lambda r: "|".join(sorted(r)), axis=1)
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(Xs, ys, routes):
        model = clone(pipe).fit(Xs.iloc[tr], ys.iloc[tr])
        scores.append(regression_metrics(ys.iloc[te], model.predict(Xs.iloc[te])))
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}


def setup_mlflow() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        exp_id = mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                          artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return exp_id


def current_champion_rmse(client: MlflowClient):
    """RMSE of the model that currently holds the champion alias (None if no champion)."""
    try:
        mv = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("test_rmse")


def train(selected=None) -> dict:
    df = load_flights()
    validate(df, "flights")
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    logger.info("train=%d test=%d", len(X_train), len(X_test))

    setup_mlflow()
    results = []
    for name, (estimator, scale) in candidate_models().items():
        if selected and name not in selected:
            continue
        pipe = Pipeline([("preprocess", build_preprocessor(scale_numeric=scale)), ("model", estimator)])
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"model_family": name, "dataset": "flights.csv",
                             "trigger": "airflow" if _in_airflow() else "manual"})
            mlflow.log_params({"model": name, "n_train": len(X_train), "n_test": len(X_test),
                               "features": ",".join(X.columns)})
            mlflow.log_params({f"hp_{k}": v for k, v in estimator.get_params().items()
                               if isinstance(v, (int, float, str, bool)) or v is None})

            start = time.time()
            pipe.fit(X_train, y_train)
            train_secs = time.time() - start

            test_m = regression_metrics(y_test, pipe.predict(X_test))
            train_m = regression_metrics(y_train, pipe.predict(X_train))
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
            mlflow.log_metric("train_seconds", train_secs)
            holdout_m = route_holdout_metrics(pipe, X_train, y_train)
            mlflow.log_metrics({f"unseen_route_{k}": v for k, v in holdout_m.items()})

            signature = infer_signature(X_test.head(50), pipe.predict(X_test.head(50)))
            # cloudpickle keeps our custom date-feature step loadable (skops would reject it)
            mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                                     input_example=X_test.head(3),
                                     serialization_format="cloudpickle")
            logger.info("%-24s rmse=%.3f mae=%.3f r2=%.5f | unseen-route r2=%.3f (%.1fs)", name,
                        test_m["rmse"], test_m["mae"], test_m["r2"], holdout_m["r2"], train_secs)
            results.append({"name": name, "run_id": run.info.run_id, "pipeline": pipe, **test_m,
                            "unseen_route_rmse": holdout_m["rmse"], "unseen_route_r2": holdout_m["r2"]})

    best = min(results, key=lambda r: r["rmse"])
    logger.info("best model: %s (rmse=%.3f)", best["name"], best["rmse"])
    promoted = register_and_promote(best)
    export_for_serving(best, X_train, promoted)
    return {k: v for k, v in best.items() if k != "pipeline"} | {"promoted": promoted}


def register_and_promote(best: dict) -> bool:
    """Register the best run; move the champion alias only if it improves RMSE."""
    client = MlflowClient()
    champion_rmse = current_champion_rmse(client)
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", config.REGISTERED_MODEL_NAME)
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version, "model_family", best["name"])

    if champion_rmse is None or best["rmse"] <= champion_rmse:
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, mv.version)
        logger.info("version %s promoted to @%s (previous rmse=%s)", mv.version,
                    config.CHAMPION_ALIAS, champion_rmse)
        return True
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "challenger", mv.version)
    logger.info("version %s kept as @challenger - champion rmse %.3f is better", mv.version, champion_rmse)
    return False


def export_for_serving(best: dict, X_train, promoted: bool) -> None:
    """Write the joblib artefact + metadata the Flask API loads at startup."""
    if not promoted and config.REGRESSION_MODEL_PATH.exists():
        logger.info("new model not promoted - keeping existing serving artefact")
        return
    joblib.dump(best["pipeline"], config.REGRESSION_MODEL_PATH, compress=3)
    meta = {
        "model_name": best["name"],
        "mlflow_run_id": best["run_id"],
        "registered_model": config.REGISTERED_MODEL_NAME,
        "metrics": {k: best[k] for k in ("rmse", "mae", "r2", "mape", "unseen_route_rmse", "unseen_route_r2")},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": list(X_train.columns),
        # used by the API to validate requests and to fill time/distance from the route
        "allowed_values": {c: sorted(X_train[c].unique().tolist()) for c in config.CATEGORICAL_FEATURES},
        "routes": {f"{f}|{t}": {"time": float(g["time"].iloc[0]), "distance": float(g["distance"].iloc[0])}
                   for (f, t), g in X_train.groupby(["from", "to"])},
    }
    config.REGRESSION_META_PATH.write_text(json.dumps(meta, indent=2))
    config.REFERENCE_STATS_PATH.write_text(json.dumps(build_reference_stats(X_train), indent=2))
    logger.info("serving artefact written to %s", config.REGRESSION_MODEL_PATH)


def _in_airflow() -> bool:
    import os
    return "AIRFLOW_CTX_DAG_ID" in os.environ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", help="subset of candidate model names to train")
    args = parser.parse_args()
    summary = train(args.models)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
