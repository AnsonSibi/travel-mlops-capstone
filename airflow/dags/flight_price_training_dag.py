"""Weekly training pipeline for the flight price model (plus the side models).

    validate_raw_data -> quality_gate -> prepare_features -> train_and_register
                                      \\-> train_gender_model                \\-> branch_on_promotion
                                      \\-> train_recommender                      |-> deploy_new_model
                                                                                  \\-> keep_current_model
                                                                     all -> pipeline_report

* Every training run is tracked in MLflow; the new model only becomes
  @champion if it beats the current champion's RMSE.
* When a new champion is promoted, deploy_new_model triggers the Jenkins
  deploy job (if JENKINS_DEPLOY_URL is configured) which builds a fresh Docker
  image and rolls it out to Kubernetes.
* The monitoring DAG can also trigger this DAG when it detects data drift.
"""
import json
import logging
import os
from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

MIN_FLIGHT_ROWS = 100_000   # refuse to train on a truncated/partial export

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}


@dag(
    dag_id="flight_price_training_pipeline",
    description="Validate travel data, retrain + register models in MLflow, deploy the champion",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["travel", "regression", "mlflow", "training"],
    doc_md=__doc__,
)
def flight_price_training_pipeline():

    @task
    def validate_raw_data() -> dict:
        """Schema + basic quality checks on all three source files."""
        from travel_ml.data import load_flights, load_hotels, load_users, validate
        return {
            "flights": validate(load_flights(), "flights"),
            "hotels": validate(load_hotels(), "hotels"),
            "users": validate(load_users(), "users"),
        }

    @task
    def quality_gate(reports: dict) -> dict:
        """Stop the pipeline early if the data looks wrong."""
        flights = reports["flights"]
        if flights["rows"] < MIN_FLIGHT_ROWS:
            raise ValueError(f"only {flights['rows']} flight rows - expected at least {MIN_FLIGHT_ROWS}")
        if flights["null_cells"] > 0.01 * flights["rows"]:
            raise ValueError(f"too many nulls in flights: {flights['null_cells']}")
        logger.info("quality gate passed: %s", reports)
        return reports

    @task
    def prepare_features(_reports: dict) -> dict:
        """Materialise the engineered feature table (useful for debugging and audits)."""
        from travel_ml import config
        from travel_ml.data import load_flights
        from travel_ml.features import add_engineered_features

        df = add_engineered_features(load_flights())
        out = config.PROCESSED_DIR / "flights_features.csv"
        df.to_csv(out, index=False)
        return {"path": str(out), "rows": len(df), "columns": len(df.columns)}

    @task
    def train_and_register(_features: dict) -> dict:
        """Train all candidates, log to MLflow, register best, promote if better."""
        from travel_ml.train_regression import train
        return train()

    @task
    def train_gender_model(_reports: dict) -> dict:
        from travel_ml.train_classifier import train
        return train()

    @task
    def train_recommender(_reports: dict) -> dict:
        from travel_ml.recommender import train
        return train()

    @task.branch
    def branch_on_promotion(summary: dict) -> str:
        return "deploy_new_model" if summary.get("promoted") else "keep_current_model"

    @task
    def deploy_new_model(summary: dict) -> str:
        """Ask Jenkins to build + deploy the new model image (CI/CD hand-off)."""
        url = os.getenv("JENKINS_DEPLOY_URL")
        if not url:
            logger.info("JENKINS_DEPLOY_URL not set - new model %s is ready in models/, "
                        "deploy with the Jenkins pipeline", summary["name"])
            return "skipped (no Jenkins URL)"
        import requests
        auth = (os.getenv("JENKINS_USER", ""), os.getenv("JENKINS_TOKEN", ""))
        resp = requests.post(url, auth=auth, params={"MODEL_RUN_ID": summary["run_id"]}, timeout=30)
        resp.raise_for_status()
        return f"jenkins triggered ({resp.status_code})"

    keep_current_model = EmptyOperator(task_id="keep_current_model")

    @task(trigger_rule="none_failed_min_one_success")
    def pipeline_report(regression: dict, gender: dict, recommender: dict) -> None:
        """Write a run summary that the Streamlit monitoring tab / reviewers can read."""
        from travel_ml import config
        report = {"generated_at": pendulum.now("UTC").isoformat(),
                  "regression": regression, "gender": gender, "recommender": recommender}
        (config.REPORTS_DIR / "last_pipeline_run.json").write_text(json.dumps(report, indent=2, default=str))
        logger.info("pipeline finished: %s", report)

    reports = quality_gate(validate_raw_data())
    summary = train_and_register(prepare_features(reports))
    gender = train_gender_model(reports)
    recs = train_recommender(reports)

    branch = branch_on_promotion(summary)
    deployed = deploy_new_model(summary)
    branch >> [deployed, keep_current_model]
    report = pipeline_report(summary, gender, recs)
    [deployed, keep_current_model] >> report


flight_price_training_pipeline()
