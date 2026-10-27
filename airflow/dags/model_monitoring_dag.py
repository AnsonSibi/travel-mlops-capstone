"""Daily monitoring of the flight price model.

    check_serving_artifacts -> run_drift_check -> branch_on_drift -> trigger_retraining
                                                                 \\-> no_drift

The API logs every prediction to logs/predictions.jsonl. This DAG compares the
live feature distribution with the training distribution (PSI per feature).
If any feature drifts beyond the threshold, the training DAG is triggered.
"""
import logging
from datetime import timedelta

import pendulum
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="flight_price_model_monitoring",
    description="Daily drift check on logged predictions; retrains when drift is found",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["travel", "monitoring", "drift"],
    doc_md=__doc__,
)
def flight_price_model_monitoring():

    @task
    def check_serving_artifacts() -> dict:
        """The serving model and its reference statistics must exist."""
        import json

        from travel_ml import config
        for path in (config.REGRESSION_MODEL_PATH, config.REFERENCE_STATS_PATH):
            if not path.exists():
                raise FileNotFoundError(path)
        meta = json.loads(config.REGRESSION_META_PATH.read_text())
        return {"model_name": meta["model_name"], "run_id": meta["mlflow_run_id"],
                "trained_at": meta["trained_at"]}

    @task
    def run_drift_check(_meta: dict) -> dict:
        from travel_ml.monitoring import run_drift_check as drift
        return drift()

    @task.branch
    def branch_on_drift(report: dict) -> str:
        if report.get("drift_detected"):
            logger.warning("drift detected in %s", report.get("drifted_features"))
            return "trigger_retraining"
        return "no_drift"

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="flight_price_training_pipeline",
        conf={"reason": "data drift"},
        wait_for_completion=False,
    )
    no_drift = EmptyOperator(task_id="no_drift")

    report = run_drift_check(check_serving_artifacts())
    branch_on_drift(report) >> [trigger_retraining, no_drift]


flight_price_model_monitoring()
