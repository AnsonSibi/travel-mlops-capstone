"""Central configuration: paths, feature lists and MLflow settings.

Everything can be overridden with environment variables so the same code runs
locally, inside Docker, inside the Airflow containers and on Kubernetes.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

FLIGHTS_CSV = RAW_DIR / "flights.csv"
HOTELS_CSV = RAW_DIR / "hotels.csv"
USERS_CSV = RAW_DIR / "users.csv"

# Model artefacts used by the API and the Streamlit app
REGRESSION_MODEL_PATH = MODELS_DIR / "flight_price_model.joblib"
REGRESSION_META_PATH = MODELS_DIR / "flight_price_model_meta.json"
GENDER_MODEL_PATH = MODELS_DIR / "gender_model.joblib"
RECOMMENDER_PATH = MODELS_DIR / "hotel_recommender.joblib"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"

# MLflow - a SQLite backend is needed for the model registry
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", (PROJECT_ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "flight-price-regression")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "flight-price-regressor")
CHAMPION_ALIAS = "champion"

# Regression features (raw request fields -> pipeline handles the rest)
CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERIC_FEATURES = ["time", "distance"]
DATE_FEATURES = ["month", "day_of_week", "is_weekend"]
REGRESSION_TARGET = "price"

RANDOM_STATE = 42
TEST_SIZE = 0.2

for _d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
