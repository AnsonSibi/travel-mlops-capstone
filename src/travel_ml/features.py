"""Feature engineering for the flight price model.

The preprocessing lives inside the sklearn Pipeline, so the API can send raw
fields (from, to, flightType, agency, time, distance, date) and get the exact
same transformations that were used in training.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from travel_ml import config


FLIGHT_CLASSES = ["economic", "premium", "firstClass"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from the flight date + distance/time x class interactions.

    Price is roughly distance * (rate per km for the class), so giving the model
    distance split by class lets even a linear model learn a per-class slope.
    A missing date falls back to today.
    """
    out = df.copy()
    for cls in FLIGHT_CLASSES:
        is_cls = (out["flightType"] == cls).astype(float)
        out[f"distance_x_{cls}"] = out["distance"] * is_cls
        out[f"time_x_{cls}"] = out["time"] * is_cls
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    else:
        dates = pd.Series(pd.NaT, index=out.index)
    dates = dates.fillna(pd.Timestamp.today().normalize())
    out["month"] = dates.dt.month
    out["day_of_week"] = dates.dt.day_of_week
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


INTERACTION_FEATURES = [f"{base}_x_{cls}" for cls in FLIGHT_CLASSES for base in ("distance", "time")]


def build_preprocessor(scale_numeric: bool = False) -> Pipeline:
    """One-hot encode categoricals; optionally scale numerics (for linear models)."""
    numeric = config.NUMERIC_FEATURES + config.DATE_FEATURES + INTERACTION_FEATURES
    num_step = StandardScaler() if scale_numeric else "passthrough"
    columns = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_FEATURES),
            ("num", num_step, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features)),
        ("columns", columns),
    ])


def split_xy(df: pd.DataFrame):
    feature_cols = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES + ["date"]
    return df[feature_cols], df[config.REGRESSION_TARGET]
