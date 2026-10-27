# REST API (Flask)

`api/app.py` serves the three models over HTTP. It loads every artefact from `models/` once at startup and runs under **gunicorn** in production (2 workers × 4 threads).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | service info + list of endpoints |
| GET | `/health` | **liveness** probe: the process is up |
| GET | `/ready` | **readiness** probe: models are loaded (503 otherwise) |
| GET | `/model-info` | serving model name, MLflow run id, metrics, training time |
| GET | `/options` | valid cities / classes / agencies and the route table |
| POST | `/predict` | flight price for one flight (object) or up to 1,000 (list) |
| POST | `/predict/gender` | `{"name": ...}` or `{"names": [...]}` → gender + confidence |
| GET | `/recommend/<user_code>?k=3&max_price=250` | hotels for an existing user |
| POST | `/recommend/profile` | `{"age": 30, "company": "4You", "max_price": 200}` → hotels for a new user |
| GET | `/metrics` | Prometheus metrics |

## Request / response flow for `/predict`

1. Parse the JSON (one object or a list), returning **400** for a missing or invalid body.
2. Validate: required fields `from`, `to`, `flightType`, `agency`; values must be ones seen in training (from `models/flight_price_model_meta.json`); `from != to`.
3. Fill `time` and `distance` from the route table if the client didn't send them; default `date` to today.
4. Build a DataFrame and call the **sklearn pipeline** (feature engineering + one-hot encoding + model), the exact code used in training.
5. Log the input + prediction to `logs/predictions.jsonl` (for drift monitoring), update the Prometheus histograms, and return JSON.

```bash
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"from":"Sao Paulo (SP)","to":"Natal (RN)","flightType":"economic","agency":"CloudFy"}'
```
```json
{"model_version": "1ac2ee0a",
 "predicted_price": 488.57,
 "input": {"from": "Sao Paulo (SP)", "to": "Natal (RN)", "flightType": "economic", "agency": "CloudFy",
           "time": 0.85, "distance": 327.55, "date": "10/27/2026"}}
```

Validation errors return 400 with a clear message:
```json
{"error": "invalid agency 'NoSuchAir'. allowed: ['CloudFy', 'FlyingDrops', 'Rainbow']"}
```

## Run

```bash
make api                                   # gunicorn on :5000
python api/app.py                          # Flask dev server
pytest tests/test_api.py                   # endpoint tests
```

Environment variables: `PORT`, `LOG_LEVEL`, `MODELS_DIR`, `LOGS_DIR`.
