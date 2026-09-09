import pytest

import store
from conftest import make_symbol


@pytest.fixture
def indexed(fake_embed):
    store.store_symbols([
        make_symbol("connect_database", "def connect_database():\n    open a socket to the database\n", start_line=1),
        make_symbol("retry_request", "def retry_request():\n    loop and back off after a failure\n", start_line=10),
        make_symbol("parse_config", "def parse_config():\n    read yaml settings from disk\n", start_line=20),
    ])


def test_store_symbols_reports_counts(fake_embed):
    stats = store.store_symbols([make_symbol("a", "def a(): pass\n"), make_symbol("b", "def b(): pass\n")])
    assert stats["new"] == 2
    assert stats["total"] == 2


def test_search_result_shape(indexed):
    hits = store.search("database connection", n=2)
    assert len(hits) == 2
    assert set(hits[0]) == {"name", "kind", "location", "distance", "score", "source"}
    assert hits[0]["location"].startswith("mod.py:")
    assert all(h["score"] > 0 for h in hits)


def test_search_keyword_signal_surfaces_exact_symbol(indexed):
    # body text doesn't mention "retry"; only the symbol name does
    hits = store.search("retry_request", n=3)
    assert hits[0]["name"] == "retry_request"


def test_search_respects_n(indexed):
    assert len(store.search("anything at all", n=1)) == 1


def test_search_raises_before_any_index(fake_embed):
    with pytest.raises(RuntimeError, match="Nothing indexed"):
        store.search("query")


def test_reindex_reflects_removed_symbols(fake_embed):
    store.store_symbols([make_symbol("old_one", "def old_one(): pass\n")])
    store.store_symbols([make_symbol("new_one", "def new_one(): pass\n")])
    names = {h["name"] for h in store.search("one", n=5)}
    assert names == {"new_one"}


def test_search_uses_real_model_for_semantics():
    # no fake_embed: exercises the actual sentence-transformers model
    store.store_symbols([
        make_symbol("login", "def login(user, password):\n    verify credentials and open a session\n"),
        make_symbol("add_numbers", "def add_numbers(a, b):\n    return a + b\n"),
    ])
    hits = store.search("how does authentication work", n=1)
    assert hits[0]["name"] == "login"
