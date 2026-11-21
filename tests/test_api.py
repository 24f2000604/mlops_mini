import pytest
from fastapi.testclient import TestClient

import app as app_module

VALID = {
    "cows": 120,
    "buffaloes": 30,
    "fodder_price_rs": 18.5,
    "temp_c": 32.0,
    "vet_visits": 2,
    "feed_kg": 240.0,
    "breed_mix": "crossbred",
    "village_cluster": "Cluster A",
}


class StubModel:
    def predict(self, frame):
        return [111.0]


@pytest.fixture
def client():
    app_module._model = StubModel()
    with TestClient(app_module.app) as test_client:
        yield test_client
    app_module._model = None


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_ok(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json() == {"prediction": 111.0}


def test_predict_rejects_high_cows(client):
    bad = dict(VALID, cows=400)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_low_feed(client):
    bad = dict(VALID, feed_kg=10.0)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "temp_c"}
    assert client.post("/predict", json=bad).status_code == 422
