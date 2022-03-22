import pytest
import json

from src import app

"""
   Sample test data
"""
DUMMY_id = "12345"
DUMMY_smile = "O=C(NCCNC(=O)C(c1ccccc1)c1ccccc1)c1ccco1"


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_feature_prediction(client):
    """
    Tests /api/predict API
    """
    response = client.post(
        "api/predict",
        data=json.dumps(
            {
                "id": DUMMY_id,
                "smile": DUMMY_smile,
            }
        ),
        content_type="application/json",
    )

    data = json.loads(response.data.decode())
    assert response.status_code == 200
    assert "The feature was successfully predicted" in data["msg"]
