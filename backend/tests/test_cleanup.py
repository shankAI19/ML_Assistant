def test_end_session_deletes_files_and_invalidates_cookie(client, runner):
    client.post("/api/sessions")
    client.post("/api/uploads?kind=training", files={"file": ("data.csv", b"x,y\n1,0\n")})
    session = next(iter(client.app.state.store.sessions.values()))
    path = client.app.state.workspace.path(session.id)
    assert path.exists()
    assert client.delete("/api/sessions/current").json() == {
        "deleted": True,
        "cleanup_pending": False,
    }
    assert not path.exists()
    assert session.closed
    assert session.id in runner.cleaned
    assert client.get("/api/sessions/current").status_code == 401


def test_polling_does_not_extend_idle_retention(client):
    client.post("/api/sessions")
    session = next(iter(client.app.state.store.sessions.values()))
    touched = session.touched
    assert client.get("/api/sessions/current").status_code == 200
    assert session.touched == touched
    session.touched -= client.app.state.settings.idle_timeout + 1
    assert client.get("/api/sessions/current").status_code == 401
    assert not client.app.state.workspace.path(session.id).exists()


def test_sessions_cannot_read_each_others_artifacts(client):
    client.post("/api/sessions")
    first = next(iter(client.app.state.store.sessions.values()))
    task_id = "a" * 32
    output = client.app.state.workspace.path(first.id) / "outputs" / task_id
    output.mkdir()
    (output / "result.json").write_text('{"secret": "dataset"}')
    assert client.get(f"/api/artifacts/{task_id}/result.json").status_code == 200
    # A second cookie jar reuses the same running ASGI application without a new lifespan.
    client.cookies.clear()
    client.post("/api/sessions")
    assert client.get(f"/api/artifacts/{task_id}/result.json").status_code == 404


def test_cross_origin_mutations_rejected(client):
    assert (
        client.post("/api/sessions", headers={"origin": "https://untrusted.example"}).status_code
        == 403
    )


def test_upload_size_enforced(client):
    client.app.state.settings.max_upload_mb = 1
    client.post("/api/sessions")
    response = client.post(
        "/api/uploads?kind=training", files={"file": ("large.csv", b"a" * (1024**2 + 1))}
    )
    assert response.status_code == 413
    session = next(iter(client.app.state.store.sessions.values()))
    assert not list((client.app.state.workspace.path(session.id) / "inputs").iterdir())


def test_cleanup_failure_is_reported_and_retried(client, runner, monkeypatch):
    client.post("/api/sessions")
    session = next(iter(client.app.state.store.sessions.values()))
    original = runner.clean

    def fail(session_id=None):
        raise RuntimeError("Docker unavailable")

    monkeypatch.setattr(runner, "clean", fail)
    response = client.delete("/api/sessions/current")
    assert response.json() == {"deleted": False, "cleanup_pending": True}
    assert session.id in client.app.state.store.pending_cleanup
    monkeypatch.setattr(runner, "clean", original)
    client.app.state.store.expire()
    assert session.id not in client.app.state.store.pending_cleanup
