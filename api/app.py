"""Flask REST API for the travel ML models.

Endpoints
---------
GET  /                      service info
GET  /health                liveness probe (process is up)
GET  /ready                 readiness probe (models are loaded)
GET  /model-info            metadata + metrics of the serving flight-price model
GET  /options               valid cities / flight types / agencies / routes
POST /predict               flight price prediction (one object or a list)
POST /predict/gender        gender prediction from a name (or list of names)
GET  /recommend/<user_code> hotel recommendations for an existing user
POST /recommend/profile     hotel recommendations for a new user (age, company)
GET  /metrics               Prometheus metrics

Run locally:
    python api/app.py                                  # dev server on :5000
    gunicorn -w 2 -b 0.0.0.0:5000 api.app:app          # production server
"""
import json
import logging
import os
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from travel_ml import config
from travel_ml.monitoring import log_prediction
from travel_ml.gender import predict_gender

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("travel-api")

app = Flask(__name__)

# ----------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "Requests by endpoint and status", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTED_PRICE = Histogram("flight_predicted_price", "Distribution of predicted prices",
                            buckets=(300, 500, 700, 900, 1100, 1300, 1500, 1800))

# ------------------------------------------------------------------ models
MODELS = {}


def load_models():
    """Load every artefact once at startup. Missing optional models are skipped."""
    MODELS["price"] = joblib.load(config.REGRESSION_MODEL_PATH)
    MODELS["price_meta"] = json.loads(config.REGRESSION_META_PATH.read_text())
    for key, path in (("gender", config.GENDER_MODEL_PATH), ("recommender", config.RECOMMENDER_PATH)):
        if path.exists():
            MODELS[key] = joblib.load(path)
        else:
            logger.warning("%s model not found at %s - endpoint disabled", key, path)
    logger.info("models loaded: %s", [k for k in MODELS if not k.endswith("_meta")])


try:
    load_models()
except FileNotFoundError as exc:  # keep the process alive so /health works and /ready reports the problem
    logger.error("could not load flight price model: %s", exc)

MODEL_VERSION = MODELS.get("price_meta", {}).get("mlflow_run_id", "unknown")[:8]
REQUIRED_FIELDS = ("from", "to", "flightType", "agency")


class BadRequest(Exception):
    pass


@app.errorhandler(BadRequest)
def handle_bad_request(exc):
    return jsonify(error=str(exc)), 400


@app.errorhandler(Exception)
def handle_unexpected(exc):
    logger.exception("unhandled error")
    return jsonify(error="internal server error"), 500


def track(endpoint):
    """Decorator recording request count + latency for Prometheus."""
    def wrap(fn):
        def inner(*args, **kwargs):
            start = time.perf_counter()
            status = "200"
            try:
                resp = fn(*args, **kwargs)
                if isinstance(resp, tuple):
                    status = str(resp[1])
                return resp
            except BadRequest:
                status = "400"
                raise
            except Exception:
                status = "500"
                raise
            finally:
                REQUESTS.labels(endpoint, status).inc()
                LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        inner.__name__ = fn.__name__
        return inner
    return wrap


# ------------------------------------------------------------- validation
def prepare_flight(record: dict) -> dict:
    """Validate one flight request and fill time/distance from the route table if absent."""
    if not isinstance(record, dict):
        raise BadRequest("each flight must be a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise BadRequest(f"missing fields: {missing}")

    meta = MODELS["price_meta"]
    for field, allowed in meta["allowed_values"].items():
        if record[field] not in allowed:
            raise BadRequest(f"invalid {field} '{record[field]}'. allowed: {allowed}")
    if record["from"] == record["to"]:
        raise BadRequest("'from' and 'to' must be different")

    route = meta["routes"].get(f"{record['from']}|{record['to']}", {})
    clean = {f: record[f] for f in REQUIRED_FIELDS}
    for num in ("time", "distance"):
        value = record.get(num, route.get(num))
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise BadRequest(f"'{num}' must be a number (or use a known route)")
        if value <= 0:
            raise BadRequest(f"'{num}' must be positive")
        clean[num] = value
    clean["date"] = record.get("date") or pd.Timestamp.today().strftime("%m/%d/%Y")
    return clean


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return jsonify(service="travel-ml-api", model_version=MODEL_VERSION,
                   endpoints=["/health", "/ready", "/model-info", "/options", "/predict",
                              "/predict/gender", "/recommend/<user_code>", "/recommend/profile",
                              "/metrics"])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/ready")
def ready():
    if "price" not in MODELS:
        return jsonify(status="not ready", reason="flight price model not loaded"), 503
    return jsonify(status="ready", models=[k for k in MODELS if not k.endswith("_meta")])


@app.get("/model-info")
def model_info():
    meta = dict(MODELS.get("price_meta", {}))
    meta.pop("routes", None)
    return jsonify(meta)


@app.get("/options")
def options():
    meta = MODELS["price_meta"]
    return jsonify(meta["allowed_values"] | {"routes": meta["routes"]})


@app.post("/predict")
@track("predict")
def predict():
    if "price" not in MODELS:
        return jsonify(error="model not loaded"), 503
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("request body must be JSON")
    records = payload if isinstance(payload, list) else [payload]
    if not records or len(records) > 1000:
        raise BadRequest("send between 1 and 1000 flights per request")

    rows = [prepare_flight(r) for r in records]
    prices = MODELS["price"].predict(pd.DataFrame(rows))
    results = []
    for row, price in zip(rows, prices):
        price = round(float(price), 2)
        PREDICTED_PRICE.observe(price)
        log_prediction(row, price, MODEL_VERSION)
        results.append({"predicted_price": price, "input": row})

    body = {"model_version": MODEL_VERSION, "predictions": results}
    if not isinstance(payload, list):
        body = {"model_version": MODEL_VERSION, **results[0]}
    return jsonify(body)


@app.post("/predict/gender")
@track("predict_gender")
def gender():
    if "gender" not in MODELS:
        return jsonify(error="gender model not available"), 503
    payload = request.get_json(silent=True) or {}
    names = payload.get("names") or ([payload["name"]] if payload.get("name") else None)
    if not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise BadRequest("send {'name': 'Jane Doe'} or {'names': [...]}")
    return jsonify(predictions=predict_gender(MODELS["gender"], names))


@app.get("/recommend/<int:user_code>")
@track("recommend_user")
def recommend_user(user_code):
    if "recommender" not in MODELS:
        return jsonify(error="recommender not available"), 503
    k = request.args.get("k", default=3, type=int)
    max_price = request.args.get("max_price", type=float)
    rec = MODELS["recommender"]
    return jsonify(user_code=user_code, history=rec.user_history(user_code),
                   recommendations=rec.recommend(user_code, k=k, max_price=max_price))


@app.post("/recommend/profile")
@track("recommend_profile")
def recommend_profile():
    if "recommender" not in MODELS:
        return jsonify(error="recommender not available"), 503
    payload = request.get_json(silent=True) or {}
    try:
        age = float(payload["age"])
    except (KeyError, TypeError, ValueError):
        raise BadRequest("'age' is required and must be a number")
    recs = MODELS["recommender"].recommend_for_profile(
        age, payload.get("company"), k=int(payload.get("k", 3)), max_price=payload.get("max_price"))
    return jsonify(recommendations=recs)


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
