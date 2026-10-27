# Common commands. Run `make help` to list them.
PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: help setup train train-regression train-classifier train-recommender test lint api streamlit \
        mlflow-ui docker-build docker-up k8s-deploy k8s-delete airflow-up simulate drift-check

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-20s %s\n", $$1, $$2}'

setup: ## create the virtualenv and install everything
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

train: train-regression train-classifier train-recommender ## train all three models

train-regression: ## train + register the flight price model in MLflow
	$(PY) -m travel_ml.train_regression

train-classifier: ## train the gender classifier
	$(PY) -m travel_ml.train_classifier

train-recommender: ## build the hotel recommender
	$(PY) -m travel_ml.recommender

test: ## run unit + API tests
	$(PY) -m pytest

lint: ## flake8
	.venv/bin/flake8 src api streamlit_app tests scripts airflow/dags

api: ## run the Flask API on :5000
	.venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 api.app:app

streamlit: ## run the Streamlit app on :8501
	.venv/bin/streamlit run streamlit_app/app.py

mlflow-ui: ## MLflow UI on :5001
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-build: ## build API + Streamlit images
	docker build -f docker/Dockerfile.api -t travel-ml-api:latest .
	docker build -f docker/Dockerfile.streamlit -t travel-ml-streamlit:latest .

docker-up: ## run API + Streamlit with docker compose
	docker compose up --build

k8s-deploy: ## deploy to minikube (loads local images first)
	minikube image load travel-ml-api:latest
	minikube image load travel-ml-streamlit:latest
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/
	kubectl -n travel-ml rollout status deployment/flight-price-api

k8s-delete: ## remove everything from the cluster
	kubectl delete namespace travel-ml

airflow-up: ## start Airflow (UI on :8080)
	docker compose -f airflow/docker-compose.yaml up --build

simulate: ## send 300 normal requests to the local API
	$(PY) scripts/simulate_traffic.py --n 300

drift-check: ## run the drift check on logged predictions
	$(PY) -m travel_ml.monitoring
