import os
import pytest
from fastapi.testclient import TestClient

# Point to a temp DB so tests don't pollute the real one
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use a temporary database for every test."""
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

def test_predict_returns_uid_and_labels(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"file": ("beatles.jpeg", f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert "prediction_uid" in body
    assert "detection_count" in body
    assert isinstance(body["labels"], list)
    assert body["detection_count"] == len(body["labels"])


# ---------------------------------------------------------------------------
# GET /prediction/{uid}
# ---------------------------------------------------------------------------

def test_get_prediction_returns_session(client):
    # First create a prediction
    with open(TEST_IMAGE, "rb") as f:
        predict_resp = client.post("/predict", files={"file": ("beatles.jpeg", f, "image/jpeg")})
    uid = predict_resp.json()["prediction_uid"]

    response = client.get(f"/prediction/{uid}")
    assert response.status_code == 200
    body = response.json()
    assert body["uid"] == uid
    assert "timestamp" in body
    assert isinstance(body["detection_objects"], list)


def test_get_prediction_objects_have_expected_fields(client):
    with open(TEST_IMAGE, "rb") as f:
        predict_resp = client.post("/predict", files={"file": ("beatles.jpeg", f, "image/jpeg")})
    uid = predict_resp.json()["prediction_uid"]

    body = client.get(f"/prediction/{uid}").json()
    for obj in body["detection_objects"]:
        assert "label" in obj
        assert "score" in obj
        assert "box" in obj


def test_get_prediction_unknown_uid_returns_404(client):
    response = client.get("/prediction/non-existent-uid")
    assert response.status_code == 404
