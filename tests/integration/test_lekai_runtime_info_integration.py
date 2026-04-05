from fastapi.testclient import TestClient

from streammuse.infrastructure.inference.server_lekai import app


client = TestClient(app)


def test_runtime_info_endpoint_integration():
    response = client.get("/runtime_info")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] in {"rule_stub", "real_model"}
    assert isinstance(data["has_real_model"], bool)
    assert isinstance(data["resolved_device"], str)
    assert isinstance(data["resolved_dtype"], str)
