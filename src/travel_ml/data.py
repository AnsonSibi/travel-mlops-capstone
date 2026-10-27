"""Loading and validating the three travel datasets."""
import logging

import pandas as pd

from travel_ml import config

logger = logging.getLogger(__name__)

# Expected schema for each file - used by the Airflow validation task
SCHEMAS = {
    "flights": ["travelCode", "userCode", "from", "to", "flightType", "price",
                "time", "distance", "agency", "date"],
    "hotels": ["travelCode", "userCode", "name", "place", "days", "price", "total", "date"],
    "users": ["code", "company", "name", "gender", "age"],
}


class DataValidationError(Exception):
    """Raised when an input file does not match what the pipeline expects."""


def load_flights(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.FLIGHTS_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def load_hotels(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.HOTELS_CSV)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def load_users(path=None) -> pd.DataFrame:
    return pd.read_csv(path or config.USERS_CSV)


def validate(df: pd.DataFrame, name: str) -> dict:
    """Basic data-quality checks. Returns a small report and raises on hard failures."""
    missing_cols = set(SCHEMAS[name]) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"{name}: missing columns {sorted(missing_cols)}")
    if df.empty:
        raise DataValidationError(f"{name}: file is empty")

    report = {
        "dataset": name,
        "rows": int(len(df)),
        "null_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if name == "flights":
        bad_price = int((df["price"] <= 0).sum())
        bad_dist = int((df["distance"] <= 0).sum())
        report.update(non_positive_price=bad_price, non_positive_distance=bad_dist)
        if bad_price or bad_dist:
            raise DataValidationError(f"flights: {bad_price} bad prices, {bad_dist} bad distances")
    if name == "users":
        bad_age = int((~df["age"].between(0, 120)).sum())
        report.update(invalid_age=bad_age)
    logger.info("validation report: %s", report)
    return report
