# Model monitoring

## 1. Prediction logging
Every `/predict` call appends a JSON line to `logs/predictions.jsonl` with a timestamp, model version, inputs and the predicted price. On Kubernetes this is an `emptyDir` volume; in docker compose it is mounted from the host.

## 2. Operational metrics (Prometheus)
`GET /metrics` exposes:
* `api_requests_total{endpoint, status}`: traffic and error rate per endpoint
* `api_request_latency_seconds{endpoint}`: latency histogram
* `flight_predicted_price`: histogram of predicted prices (a sudden shift is an early warning)

The pod template has `prometheus.io/scrape` annotations, so a Prometheus server in the cluster discovers the pods automatically.

## 3. Data drift (PSI)
At training time `models/reference_stats.json` stores the distribution of every input feature (quantile bins for numerics, frequencies for categoricals). `python -m travel_ml.monitoring` (or the daily Airflow DAG) compares the logged live inputs with it using the **Population Stability Index**:

| PSI | Meaning | Action |
|---|---|---|
| < 0.1 | stable | none |
| 0.1–0.2 | moderate shift | watch |
| > 0.2 | significant drift | the DAG triggers retraining |

Verified behaviour:
```text
normal traffic (300 req):  distance 0.014, flightType 0.006, to 0.057  -> drift_detected: false
skewed traffic (+400 long first-class trips): distance 0.88, time 0.88, flightType 0.49 -> drift_detected: true
```

## 4. Model quality over time
* MLflow keeps the test and unseen-route metrics of every retrain, so the trend can be checked in the UI.
* The Jenkins quality gate blocks any model below R² 0.95 or above RMSE 50.
* The Streamlit **Monitoring** tab shows the serving model's metrics, the recommender's HitRate against the random baseline, the live prediction distribution and the latest drift report.
