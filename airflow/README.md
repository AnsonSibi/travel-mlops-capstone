# Apache Airflow: workflow orchestration

Two DAGs in `airflow/dags/` automate the life cycle of the flight price model. Both were tested end to end with `airflow dags test` on Airflow 3.3.

## `flight_price_training_pipeline` (weekly)

```
validate_raw_data ─► quality_gate ─┬─► prepare_features ─► train_and_register ─► branch_on_promotion ─┬─► deploy_new_model ──┐
                                   ├─► train_gender_model ──────────────────────────────────────────┐  └─► keep_current_model ┤
                                   └─► train_recommender ───────────────────────────────────────────┴──────────────────────► pipeline_report
```

| Task | What it does |
|---|---|
| `validate_raw_data` | schema, nulls, duplicates, non-positive prices/distances for all 3 files |
| `quality_gate` | fails the run if flights has < 100k rows or > 1% nulls, so we never train on a broken export |
| `prepare_features` | writes the engineered feature table to `data/processed/` for auditing |
| `train_and_register` | trains 4 regressors, logs everything to MLflow, registers the best and promotes it to `@champion` only if RMSE improves |
| `train_gender_model`, `train_recommender` | retrain the side models in parallel |
| `branch_on_promotion` | `deploy_new_model` if a new champion was promoted, else `keep_current_model` |
| `deploy_new_model` | calls the Jenkins job (`JENKINS_DEPLOY_URL`) to build and roll out the new image |
| `pipeline_report` | writes `reports/last_pipeline_run.json` |

Retries (1 × 5 min), a 45 min execution timeout, `max_active_runs=1` and `catchup=False` keep runs reliable and non-overlapping.

## `flight_price_model_monitoring` (daily)

```
check_serving_artifacts ─► run_drift_check ─► branch_on_drift ─┬─► trigger_retraining  (TriggerDagRunOperator)
                                                               └─► no_drift
```
It computes the PSI of every input feature between live predictions (`logs/predictions.jsonl`) and the training distribution (`models/reference_stats.json`). If any PSI > 0.2, it triggers the training DAG.

## Run
```bash
make airflow-up                    # docker compose -f airflow/docker-compose.yaml up --build
# UI: http://localhost:8080 → unpause both DAGs → trigger flight_price_training_pipeline
```
The compose file mounts the whole repo, so DAG runs read `data/raw` and write `models/`, `mlflow.db` and `reports/` in the repo.
