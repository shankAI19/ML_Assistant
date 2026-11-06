def test_unknown_api_route_is_not_frontend_html(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_configuration_files_are_never_served(client):
    for path in [
        "/back.env",
        "/front.env",
        "/backend/back.env",
        "/frontend/front.env",
        "/.runtime/sessions",
    ]:
        assert client.get(path).status_code == 404


def test_health_exposes_no_credentials(client):
    from pydantic import SecretStr

    client.app.state.settings.api_key = SecretStr("should-not-be-in-response")
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "should-not-be-in-response" not in response.text
    assert response.json()["llm_configured"] is True


def test_built_frontend_and_its_assets_are_served(client):
    import re

    import pytest

    response = client.get("/")
    if not response.headers["content-type"].startswith("text/html"):
        pytest.skip("Build the frontend before running static asset integration checks")
    assert response.status_code == 200
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', response.text)
    assert assets
    for asset in assets:
        assert client.get(asset).status_code == 200
