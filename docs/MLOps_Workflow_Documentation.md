# MLOps Workflow Documentation: Flight Price Prediction System

**Project:** End-to-End MLOps Capstone (Travel & Tourism)
**Author:** Anson Sibi
**Repository:** https://github.com/ansonsibi/travel-mlops-capstone

This document explains how the flight price prediction model moves through each MLOps stage, from a trained model to a scaled, scheduled, continuously deployed and monitored service. Each section describes what the stage does, how it is implemented in the repository, and what the screenshot shows. The 📸 markers are where the screenshots go.

---

## 0. System overview

The regression model predicts a flight's price from the route (`from`, `to`), class (`flightType`), `agency`, `time`, `distance` and `date`. The trained sklearn pipeline (feature engineering + one-hot encoding + Histogram Gradient Boosting) is the single artefact that every stage below packages, serves, schedules, tracks and monitors.

| Stage | Tool | Repository location |
|---|---|---|
| Model serving | Flask REST API + gunicorn | `api/app.py` |
| User interface | Streamlit | `streamlit_app/app.py` |
| Packaging | Docker | `docker/Dockerfile.api`, `docker-compose.yml` |
| Scaling | Kubernetes | `k8s/*.yaml` |
| Scheduling | Apache Airflow | `airflow/dags/*.py` |
| CI/CD | Jenkins | `Jenkinsfile` |
| Tracking & versioning | MLflow | `src/travel_ml/train_regression.py` |

📸 *Screenshot 0: repository tree on GitHub showing the folders above.*

---

## 1. REST API (Flask)

**Purpose:** expose the model for real-time predictions so any client (website, mobile app, Streamlit) can get a price with one HTTP call.

**How it works**
1. At startup the API loads `models/flight_price_model.joblib` and its metadata (allowed values, route table, MLflow run id) once, so requests never reload the model.
2. `POST /predict` accepts one flight (JSON object) or a batch (JSON list of up to 1,000).
3. Each record is validated: required fields, values seen in training, origin ≠ destination, positive numbers. Invalid requests get **HTTP 400** with a clear message instead of a wrong price.
4. Missing `time`/`distance` are filled from the route table and a missing `date` defaults to today, so a client only needs to send the route, class and agency.
5. The validated rows go through the same sklearn pipeline used in training, so there is no training/serving skew.
6. The response includes the price, the completed input and the `model_version` (MLflow run id). The request is logged for drift monitoring and counted in Prometheus metrics.
7. `GET /health` (liveness) and `GET /ready` (readiness) are used by Docker and Kubernetes. `GET /metrics` exposes Prometheus metrics.

**Example**
```
POST /predict
{"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass", "agency": "FlyingDrops"}

200 OK
{"model_version": "1ac2ee0a", "predicted_price": 1434.38,
 "input": {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass",
           "agency": "FlyingDrops", "time": 1.76, "distance": 676.53, "date": "10/27/2026"}}
```

📸 *Screenshot 1a: Postman (or curl) single prediction with a 200 response.*
📸 *Screenshot 1b: batch prediction with a list of flights.*
📸 *Screenshot 1c: validation error (e.g. invalid agency) returning 400.*
📸 *Screenshot 1d: `/model-info` and `/metrics` responses.*

---

## 2. Streamlit web app

**Purpose:** an interactive interface for business users that shows hotel recommendations, travel insights and live model predictions.

**How it works**
* **Hotel recommendations tab:** choose a user (or a new user's age and company), a budget per night and the number of suggestions. The hybrid recommender ranks the hotels, and the page shows hotel cards, a score chart and the user's booking history.
* **Travel insights tab:** KPIs and Plotly charts (popular destinations, monthly revenue, price by class and agency, price vs distance).
* **Flight price tab:** the user picks the route, class, agency and date. If `API_URL` is set, the app calls the Flask `/predict` endpoint (in Kubernetes: `http://flight-price-api:5000`), so it is a real API client.
* **Monitoring tab:** serving model metrics from MLflow metadata, recommender HitRate vs random, live prediction histogram and the latest drift report.

📸 *Screenshot 2a: recommendations for an existing user (cards + history).*
📸 *Screenshot 2b: cold-start recommendations for a new user with a budget filter.*
📸 *Screenshot 2c: travel insights dashboard.*
📸 *Screenshot 2d: flight price prediction via the API.*
📸 *Screenshot 2e: monitoring tab.*

---

## 3. Docker deployment

**Purpose:** package the model, code and exact dependencies into an image that runs the same on a laptop, Jenkins or Kubernetes.

**How it works**
* `docker/Dockerfile.api` starts from `python:3.12-slim` and installs the pinned `requirements-api.txt` (the same scikit-learn version that trained the model) **before** copying the code, so dependency layers are cached.
* It installs the `travel_ml` package (feature engineering used by the pickled pipeline), then copies `api/` and `models/`.
* It runs as a **non-root user**, declares a **HEALTHCHECK** on `/health` and starts **gunicorn** with 2 workers × 4 threads.
* `docker-compose.yml` runs the API and Streamlit together. Streamlit reaches the API through the compose network at `http://api:5000`, and prediction logs are mounted to the host.

**Commands used**
```
docker build -f docker/Dockerfile.api -t travel-ml-api:latest .
docker run -d -p 5000:5000 --name travel-api travel-ml-api:latest
docker compose up --build
```

📸 *Screenshot 3a: `docker build` output finishing successfully.*
📸 *Screenshot 3b: `docker images` / `docker ps` showing the running containers (healthy).*
📸 *Screenshot 3c: a prediction served by the container.*

---

## 4. Kubernetes deployment

**Purpose:** run the API reliably at scale, with automatic restarts, load balancing, autoscaling and zero-downtime updates.

**How it works**
* `namespace.yaml` isolates everything in `travel-ml`. `configmap.yaml` holds the configuration.
* `deployment.yaml` runs **2 replicas** of the API with CPU/memory requests and limits.
  * **livenessProbe** `/health`: Kubernetes restarts a hung pod.
  * **readinessProbe** `/ready`: a pod gets traffic only after the model has loaded.
  * **RollingUpdate** with `maxUnavailable: 0, maxSurge: 1`: a new model version replaces pods one at a time with no downtime. `revisionHistoryLimit: 5` keeps old versions for rollback.
* `service.yaml` (NodePort 30500) load-balances requests across all healthy pods.
* `hpa.yaml` **autoscales from 2 to 6 pods** when average CPU exceeds 60%, with a 120 s stabilisation window to avoid flapping.

**Commands used**
```
kubectl apply -f k8s/
kubectl -n travel-ml get pods,svc,hpa
kubectl -n travel-ml rollout status deployment/flight-price-api
python scripts/simulate_traffic.py --url http://<minikube-ip>:30500 --n 5000 --workers 32   # load test
kubectl -n travel-ml get hpa -w
```

📸 *Screenshot 4a: `kubectl get pods,svc,hpa` with 2 running API pods.*
📸 *Screenshot 4b: prediction through the NodePort / `minikube service` URL.*
📸 *Screenshot 4c: HPA scaling up during the load test (replicas > 2).*
📸 *Screenshot 4d: `kubectl rollout history` after a new deployment.*

---

## 5. Scheduling with Apache Airflow

**Purpose:** automate the retraining and monitoring of the regression model so the model stays fresh without manual work.

**DAG 1 – `flight_price_training_pipeline` (weekly)**
1. `validate_raw_data`: schema, nulls, duplicates and value checks on all three datasets.
2. `quality_gate`: stops the run if the data looks broken (too few rows, too many nulls).
3. `prepare_features`: materialises the engineered features in `data/processed/`.
4. `train_and_register`: trains 4 candidate models, logs them to MLflow, registers the best and promotes it to `@champion` only if it improves RMSE.
5. `train_gender_model` / `train_recommender`: retrain the other models in parallel.
6. `branch_on_promotion`: if a new champion was promoted, `deploy_new_model` triggers the Jenkins job. Otherwise `keep_current_model`.
7. `pipeline_report`: writes a JSON summary of the run.

**DAG 2 – `flight_price_model_monitoring` (daily)**
`check_serving_artifacts` → `run_drift_check` (PSI on logged predictions vs training data) → `branch_on_drift` → `trigger_retraining` (starts DAG 1) or `no_drift`.

Reliability settings: retries with delay, execution timeouts, `max_active_runs=1` and `catchup=False`.

📸 *Screenshot 5a: Airflow DAG list with both DAGs.*
📸 *Screenshot 5b: graph view of the training DAG after a successful run (all green, one branch skipped).*
📸 *Screenshot 5c: task log of `train_and_register` showing model metrics and promotion.*
📸 *Screenshot 5d: monitoring DAG run where drift was detected and `trigger_retraining` ran.*

---

## 6. CI/CD pipeline with Jenkins

**Purpose:** every code or model change goes through the same automated checks before reaching production, and a bad deployment rolls back automatically.

**Pipeline stages (`Jenkinsfile`)**
1. **Checkout**: image tag = `<build number>-<git sha>`.
2. **Setup Python**: virtualenv with pinned dependencies.
3. **Lint**: flake8.
4. **Unit tests**: pytest (API, feature engineering, drift), with the JUnit report shown in Jenkins.
5. **Retrain models** (optional parameter): runs training, logged to MLflow.
6. **Model quality gate**: fails the build if R² < 0.95 or RMSE > 50.
7. **Build Docker images**: API and Streamlit.
8. **Smoke test container**: starts the new image, waits for `/ready`, sends a real `/predict` request.
9. **Push images** to the registry (or load them into minikube).
10. **Deploy to Kubernetes**: `kubectl apply`, `set image` with the new tag, records the change cause and waits for `rollout status`.
11. **Automatic rollback**: if the rollout fails, `kubectl rollout undo` restores the previous model.

Triggers: a GitHub webhook on every push, and a remote trigger from the Airflow `deploy_new_model` task after a new champion is promoted.

📸 *Screenshot 6a: Jenkins stage view with all stages green.*
📸 *Screenshot 6b: test results (JUnit) and the quality gate log.*
📸 *Screenshot 6c: console output of the deploy stage (`successfully rolled out`).*

---

## 7. MLflow tracking

**Purpose:** a record of every experiment and a controlled way to version and promote models.

**How it works**
* Each candidate model is one **run** in the `flight-price-regression` experiment, with all hyperparameters, train/test metrics (RMSE, MAE, R², MAPE), the **unseen-route** robustness metrics, training time, tags (who or what triggered it) and the model artefact with its input signature.
* The best run is registered in the **Model Registry** as `flight-price-regressor`. The **`@champion`** alias moves only when the new version beats the current champion's RMSE; otherwise the version is tagged **`@challenger`**.
* The served model's run id is stored in `models/flight_price_model_meta.json` and returned by the API as `model_version`, so every prediction can be traced to its run.

📸 *Screenshot 7a: experiment page comparing the 4 candidate runs (metrics columns).*
📸 *Screenshot 7b: run comparison chart (test_rmse vs unseen_route_r2).*
📸 *Screenshot 7c: a run's details: parameters, metrics, model artefact with signature.*
📸 *Screenshot 7d: Model Registry: versions of `flight-price-regressor` with the `@champion` alias.*

---

## 8. Monitoring

* **Operational:** Prometheus metrics at `/metrics` (request counts, errors, latency, predicted-price histogram); Kubernetes probes and restarts.
* **Data drift:** PSI per feature between live inputs and the training distribution. In testing, normal traffic gave PSI < 0.06 on every feature, while skewed traffic (only long first-class trips) gave distance PSI ≈ 0.88 and triggered retraining.
* **Model quality:** metrics of every retrain in MLflow, the Jenkins quality gate, and the Streamlit monitoring tab.

📸 *Screenshot 8a: drift report JSON with `drift_detected: true`.*
📸 *Screenshot 8b: `/metrics` output.*

---

## 9. End-to-end flow (summary)

`git push` → **Jenkins** (lint, tests, quality gate, build, smoke test, deploy) → **Kubernetes** serves the **Flask** API (scaled by the HPA) → **Streamlit** and other clients call it → predictions are logged → **Airflow** checks drift daily and retrains weekly → **MLflow** records and versions each model → a new champion triggers **Jenkins** again.
