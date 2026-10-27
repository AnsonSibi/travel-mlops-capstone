# ✈️ Travel MLOps Capstone: Flight Price Prediction, Gender Classification & Hotel Recommendation

An end-to-end machine learning system for a travel platform: three models built on linked travel datasets (users, flights, hotels) and taken all the way to production with a REST API, Docker, Kubernetes, Apache Airflow, Jenkins CI/CD, MLflow and a Streamlit web app.

| Use case | Model | Test performance | Served by |
|---|---|---|---|
| **Flight price prediction** (regression) | Histogram Gradient Boosting in an sklearn pipeline | RMSE ≈ R$0.0005, R² = 1.000 (unseen routes: R² 0.74) | Flask `POST /predict` |
| **Gender classification** | Char n-gram TF-IDF + Logistic Regression | Accuracy 0.89, macro-F1 0.89, ROC-AUC 0.96 | Flask `POST /predict/gender` |
| **Hotel recommendation** | Hybrid user-based CF + popularity | HitRate@3 0.377 (random 0.333, popularity 0.353) | Flask `GET /recommend/<user>` + Streamlit |

---

## Table of contents
1. [Motivation](#1-motivation)
2. [Data](#2-data)
3. [Architecture](#3-architecture)
4. [Repository structure](#4-repository-structure)
5. [Quick start](#5-quick-start)
6. [Components](#6-components)
7. [Model results](#7-model-results)
8. [MLOps workflow](#8-mlops-workflow)
9. [Monitoring](#9-monitoring)
10. [Future improvements](#10-future-improvements)

---

## 1. Motivation

A travel platform makes thousands of decisions a day: what price to quote for a flight, how to personalise a customer profile, which hotel to suggest. A model in a notebook can't do any of that, because it has to be **served, scaled, retrained, versioned and monitored**. This project builds the model *and* the system around it:

* **Prediction:** quote a flight price instantly through a REST API.
* **Classification:** fill in missing customer profiles (a third of the users never declared a gender).
* **Recommendation:** suggest the hotels a user is most likely to book next.
* **Automation:** retrain weekly, detect data drift daily, and deploy new champions through CI/CD with automatic rollback.

## 2. Data

Three CSV files in [`data/raw/`](data/raw), linked by `userCode` and `travelCode`:

| File | Rows | Key columns |
|---|---|---|
| `users.csv` | 1,340 | `code`, `company`, `name`, `gender`, `age` |
| `flights.csv` | 271,888 | `travelCode`, `userCode`, `from`, `to`, `flightType`, `price`, `time`, `distance`, `agency`, `date` |
| `hotels.csv` | 40,552 | `travelCode`, `userCode`, `name`, `place`, `days`, `price`, `total`, `date` |

```
users.code ──< flights.userCode        users.code ──< hotels.userCode
flights.travelCode (outbound + return) ── hotels.travelCode (≈30% of trips have a hotel stay)
```

Key facts found in the EDA:
* Flight price is a deterministic tariff: every *route × class × agency* combination always has the same price.
* A third of users have gender `none`. The same first names appear under `none` and under male/female, so `none` is treated as a missing label.
* There is one hotel per city (9 hotels), most users have stayed at 7–9 of them, and hotel choice depends on company.

## 3. Architecture

```
                    ┌──────────────────────── Apache Airflow ────────────────────────┐
  data/raw/*.csv ──►│ validate → quality gate → features → train (3 models) → branch  │
                    │                                   │                     │       │
                    │  daily: drift check (PSI) ──► trigger retraining        ▼       │
                    └───────────────────────────────────┼──────── deploy_new_model ───┘
                                                        ▼                     │ (REST)
                                   MLflow tracking + model registry           ▼
                                   (@champion / @challenger)        Jenkins CI/CD pipeline
                                                                     lint → tests → quality gate
                                                                     → docker build → smoke test
                                                                     → push → kubectl rollout
                                                                              │
          ┌────────────────────────── Kubernetes (namespace travel-ml) ───────▼──────────┐
          │  flight-price-api Deployment (2–6 pods, HPA on CPU)   ◄── NodePort :30500    │
          │     gunicorn + Flask  /predict /predict/gender /recommend /metrics           │
          │  travel-streamlit Deployment                          ◄── NodePort :30851    │
          └───────────────────────────────────────────────────────────────────────────────┘
                     │ prediction logs (JSONL)                        ▲
                     └──────────────► drift check (Airflow) ──────────┘
```

## 4. Repository structure

```
travel-mlops-capstone/
├── data/raw/                     # users.csv, flights.csv, hotels.csv
├── notebooks/                    # modelling notebooks (project template format)
│   ├── 01_Flight_Price_Regression.ipynb
│   ├── 02_Gender_Classification.ipynb
│   └── 03_Hotel_Recommendation.ipynb
├── src/travel_ml/                # shared, importable package
│   ├── config.py                 #   paths, features, MLflow settings (env-overridable)
│   ├── data.py                   #   loading + schema/quality validation
│   ├── features.py               #   feature engineering inside the sklearn pipeline
│   ├── train_regression.py       #   train/compare/register flight price models (MLflow)
│   ├── train_classifier.py       #   gender classifier training (MLflow)
│   ├── gender.py                 #   light inference helpers for the gender model
│   ├── recommender.py            #   hybrid hotel recommender + evaluation
│   └── monitoring.py             #   prediction logging + PSI drift detection
├── models/                       # serving artefacts (*.joblib + metadata json)
├── api/                          # Flask REST API  → api/README.md
├── streamlit_app/                # Streamlit web app → streamlit_app/README.md
├── docker/                       # Dockerfiles  → docker/README.md
├── docker-compose.yml            # API + Streamlit locally
├── k8s/                          # Kubernetes manifests → k8s/README.md
├── airflow/                      # DAGs + Airflow image/compose → airflow/README.md
├── Jenkinsfile                   # CI/CD pipeline → docs/JENKINS.md
├── scripts/                      # traffic simulator, CI quality gate
├── tests/                        # pytest: API, features, drift
├── docs/                         # workflow documentation, MLflow guide, video outline
├── Makefile                      # one-line commands for everything
└── requirements*.txt             # pinned dependencies (full / API / Streamlit)
```

## 5. Quick start

Prerequisites: Python 3.11+, and for the infrastructure parts Docker, minikube + kubectl and Jenkins.

```bash
git clone https://github.com/ansonsibi/travel-mlops-capstone.git
cd travel-mlops-capstone
make setup            # virtualenv + pinned dependencies + editable install of travel_ml
make train            # trains all 3 models, logs to MLflow, writes models/
make test             # 16 unit/API tests
make api              # REST API on http://localhost:5000
make streamlit        # web app on http://localhost:8501
make mlflow-ui        # MLflow UI on http://localhost:5001
```

Try the API:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops","date":"09/26/2019"}'
# {"model_version":"...","predicted_price":1434.38,"input":{... "distance":676.53,"time":1.76 ...}}
```

Run everything in containers / on Kubernetes:
```bash
make docker-build && docker compose up             # API :5000 + Streamlit :8501
minikube start && minikube addons enable metrics-server
make k8s-deploy                                    # namespace, deployments, services, HPA
minikube service flight-price-api -n travel-ml --url
make airflow-up                                    # Airflow UI on http://localhost:8080
```

All `make` targets: `make help`.

## 6. Components

| Component | What it does | Docs |
|---|---|---|
| **Notebooks** | EDA (15 charts each), hypothesis tests, feature engineering, 3 models per task with CV tuning, explainability | [`notebooks/`](notebooks) |
| **REST API (Flask)** | Validated JSON endpoints, batch prediction, readiness/liveness probes, Prometheus metrics, prediction logging | [`api/README.md`](api/README.md) |
| **Docker** | Slim non-root images with health checks for the API and the Streamlit app | [`docker/README.md`](docker/README.md) |
| **Kubernetes** | Deployment (rolling updates, zero downtime), NodePort Service, HPA 2–6 pods, ConfigMap | [`k8s/README.md`](k8s/README.md) |
| **Apache Airflow** | Weekly training DAG with a quality gate and promotion branch; daily drift DAG that triggers retraining | [`airflow/README.md`](airflow/README.md) |
| **Jenkins** | Declarative CI/CD with lint, tests, quality gate, build, smoke test, push, deploy and auto-rollback | [`docs/JENKINS.md`](docs/JENKINS.md) |
| **MLflow** | Experiment tracking, model registry, `@champion`/`@challenger` aliases | [`docs/MLFLOW.md`](docs/MLFLOW.md) |
| **Streamlit** | Hotel recommendations, travel insights, live price/gender prediction, monitoring tab | [`streamlit_app/README.md`](streamlit_app/README.md) |
| **Monitoring** | Prediction log + PSI drift report + Prometheus metrics | [`docs/MONITORING.md`](docs/MONITORING.md) |

The end-to-end workflow write-up (used for the submission document) is in [`docs/MLOps_Workflow_Documentation.md`](docs/MLOps_Workflow_Documentation.md).

## 7. Model results

### Flight price regression
| Model | Test RMSE (R$) | Test R² | Unseen-route R² |
|---|---|---|---|
| Ridge (baseline, with distance × class interactions) | 87.9 | 0.941 | **0.874** |
| Decision Tree | 1.29 | 0.99999 | 0.625 |
| Random Forest | 0.81 | 0.99999 | 0.693 |
| **Hist Gradient Boosting (tuned)** | **0.0005** | **1.000** | 0.736 |

The test set contains the same routes as training, so the trees reproduce the tariff exactly. The **unseen-route** column (GroupKFold by route) measures how a model would price a *new* route, and there the linear model generalises best. Both are logged to MLflow for every run.

### Gender classification
| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression on travel behaviour (baseline) | 0.51 | 0.50 | 0.48 |
| Naive Bayes, char n-grams (tuned) | 0.87 | 0.87 | 0.95 |
| **Logistic Regression, char n-grams (tuned)** | **0.89** | **0.89** | **0.96** |

Accuracy on first names never seen in training is 0.74. The API answers `uncertain` below 60% confidence: at that threshold 94% of users get a label, with 92% accuracy.

### Hotel recommendation (time-based leave-last-out, 1,288 users)
| Model | HitRate@3 | NDCG@3 |
|---|---|---|
| Random | 0.324 | 0.223 |
| Popularity | 0.353 | 0.255 |
| User-based k-NN | 0.378 | 0.271 |
| **Hybrid (weights tuned on validation stays)** | **0.377 ± 0.026** | **0.270** |

Users rotate almost randomly among the hotels they already know, so the ceiling for next-hotel prediction is low. The notebook explains why and what context data (flight destination) would lift it.

## 8. MLOps workflow

1. **Develop:** notebooks → reusable code in `src/travel_ml` (same feature pipeline for training and serving).
2. **Track:** every training run logs params, metrics (test + unseen-route), the model and its signature to **MLflow**. The best run is registered as `flight-price-regressor` and only takes the **`@champion`** alias if it beats the current champion's RMSE; otherwise it becomes `@challenger`.
3. **Orchestrate:** the **Airflow** DAG `flight_price_training_pipeline` (weekly) validates the data, runs a quality gate, trains all models, and branches: a promoted model calls the Jenkins deploy job, otherwise the current model is kept.
4. **CI/CD:** the **Jenkins** pipeline runs lint → pytest → (optional retrain) → model quality gate (R² ≥ 0.95, RMSE ≤ 50) → Docker build → container smoke test → push → `kubectl set image` + `rollout status`, with **automatic `rollout undo`** if the deploy fails.
5. **Serve and scale:** **Kubernetes** runs 2–6 API pods behind a Service. Readiness probes only send traffic once the model has loaded, and rolling updates with `maxUnavailable: 0` give zero-downtime model swaps.
6. **Monitor:** the API logs every prediction and exposes Prometheus metrics. The daily DAG `flight_price_model_monitoring` computes PSI drift against the training distribution and triggers retraining when drift > 0.2.

## 9. Monitoring

```bash
python scripts/simulate_traffic.py --n 300            # normal traffic   -> no drift (all PSI < 0.06)
python scripts/simulate_traffic.py --n 400 --drift    # long first-class trips only
python -m travel_ml.monitoring                        # -> drift_detected: true (distance PSI ≈ 0.88)
curl localhost:5000/metrics                           # request counts, latency, predicted price histogram
```
Details: [`docs/MONITORING.md`](docs/MONITORING.md).

## 10. Future improvements

* Add booking lead time and seat availability to support **dynamic pricing**.
* Use the flight destination as context for hotel recommendations (cross-sell to the 70% of trips without a hotel).
* Add **authentication** (API keys/JWT) and rate limiting on the API.
* Move to a managed cluster (EKS/GKE) with a remote MLflow server (Postgres + S3) and Prometheus/Grafana dashboards.
* Run canary or A/B deployments of challenger models using the MLflow aliases.

---
**Author:** Anson Sibi · Almabetter End-to-End MLOps Capstone
