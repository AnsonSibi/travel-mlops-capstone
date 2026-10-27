"""API tests - run with `pytest`. They need the trained artefacts in models/."""
import pytest

from api.app import app

FLIGHT = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass",
          "agency": "FlyingDrops", "date": "09/26/2019"}


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json["status"] == "ok"
    assert client.get("/ready").status_code == 200


def test_predict_single_known_route(client):
    resp = client.post("/predict", json=FLIGHT)
    assert resp.status_code == 200
    # actual price in flights.csv for this trip is 1434.38
    assert resp.json["predicted_price"] == pytest.approx(1434.38, abs=5)
    assert resp.json["input"]["distance"] == pytest.approx(676.53)


def test_predict_batch(client):
    economic = FLIGHT | {"flightType": "economic", "agency": "CloudFy"}
    resp = client.post("/predict", json=[FLIGHT, economic])
    preds = [p["predicted_price"] for p in resp.json["predictions"]]
    assert len(preds) == 2
    assert preds[0] > preds[1]  # first class costs more than economy


@pytest.mark.parametrize("bad, msg", [
    ({"from": "Recife (PE)"}, "missing fields"),
    (FLIGHT | {"agency": "NoSuchAir"}, "invalid agency"),
    (FLIGHT | {"to": "Recife (PE)"}, "must be different"),
    (FLIGHT | {"distance": -10}, "must be positive"),
])
def test_predict_validation(client, bad, msg):
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 400
    assert msg in resp.json["error"]


def test_predict_requires_json(client):
    assert client.post("/predict", data="not json").status_code == 400


def test_gender(client):
    resp = client.post("/predict/gender", json={"names": ["Maria Silva", "John Smith"]})
    labels = [p["gender"] for p in resp.json["predictions"]]
    assert labels == ["female", "male"]


def test_recommend(client):
    resp = client.get("/recommend/0?k=3")
    assert resp.status_code == 200
    assert len(resp.json["recommendations"]) == 3
    cold = client.post("/recommend/profile", json={"age": 30, "company": "4You", "max_price": 200})
    assert all(r["price_per_night"] <= 200 for r in cold.json["recommendations"])


def test_metrics_exposed(client):
    client.post("/predict", json=FLIGHT)
    assert b"api_requests_total" in client.get("/metrics").data
