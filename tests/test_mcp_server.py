import mcp_server


def test_clip_passes_short_source_through():
    src = "line1\nline2\nline3"
    assert mcp_server._clip(src) == src


def test_clip_truncates_long_source():
    src = "\n".join(f"line{i}" for i in range(200))
    out = mcp_server._clip(src)
    assert "more lines" in out
    assert len(out.splitlines()) < 200


def test_search_code_reports_missing_index(monkeypatch):
    def raise_runtime(*_args, **_kwargs):
        raise RuntimeError("Nothing indexed yet. Run `python src/main.py index <repo>` first.")

    monkeypatch.setattr(mcp_server, "search", raise_runtime)
    assert "Nothing indexed" in mcp_server.search_code("q")


def test_search_code_formats_hits(monkeypatch):
    monkeypatch.setattr(mcp_server, "search", lambda q, n: [
        {"name": "foo", "kind": "function", "location": "a.py:3",
         "distance": 0.5, "score": 0.0321, "source": "def foo():\n    return 1"},
    ])
    out = mcp_server.search_code("q", n=1)
    assert "1. foo" in out
    assert "a.py:3" in out
    assert "score 0.0321" in out
    assert "distance 0.500" in out


def test_search_code_handles_name_only_hit(monkeypatch):
    monkeypatch.setattr(mcp_server, "search", lambda q, n: [
        {"name": "bar", "kind": "method", "location": "b.py:9",
         "distance": None, "score": 0.01, "source": "def bar(self): ..."},
    ])
    out = mcp_server.search_code("q", n=1)
    assert "name match" in out


def test_search_code_clamps_n(monkeypatch):
    seen = {}

    def capture(q, n):
        seen["n"] = n
        return []

    monkeypatch.setattr(mcp_server, "search", capture)
    mcp_server.search_code("q", n=999)
    assert seen["n"] == 20


def test_index_status_when_empty():
    out = mcp_server.index_status()
    assert "No codebase indexed" in out
