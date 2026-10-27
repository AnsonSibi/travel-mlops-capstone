"""Unit tests for feature engineering, data validation and drift detection."""
import pandas as pd
import pytest

from travel_ml.data import DataValidationError, validate
from travel_ml.features import add_engineered_features
from travel_ml.monitoring import build_reference_stats, compute_drift, psi


def sample_flights(n=200, distance=500.0, flight_type="economic"):
    return pd.DataFrame({
        "from": ["Recife (PE)"] * n, "to": ["Natal (RN)"] * n, "flightType": [flight_type] * n,
        "agency": ["CloudFy"] * n, "time": [1.2] * n, "distance": [distance] * n,
        "date": ["10/05/2019"] * n,
    })


def test_engineered_features():
    out = add_engineered_features(sample_flights(1))
    assert out.loc[0, "month"] == 10
    assert out.loc[0, "day_of_week"] == 5 and out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "distance_x_economic"] == 500.0
    assert out.loc[0, "distance_x_firstClass"] == 0.0


def test_missing_date_defaults_to_today():
    out = add_engineered_features(sample_flights(1).drop(columns="date"))
    assert out.loc[0, "month"] == pd.Timestamp.today().month


def test_validate_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        validate(pd.DataFrame({"price": [1]}), "flights")


def test_psi_zero_for_same_distribution():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_drift_detected_on_shifted_data():
    ref = pd.concat([sample_flights(500, d) for d in (200, 400, 600, 800)])
    stats = build_reference_stats(ref)
    assert not compute_drift(stats, ref)["drift_detected"]
    shifted = sample_flights(500, 800, "firstClass")
    report = compute_drift(stats, shifted)
    assert report["drift_detected"]
    assert "distance" in report["drifted_features"]
