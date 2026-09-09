import pytest

import store
from conftest import make_symbol


@pytest.fixture
def counting_embed(monkeypatch):
    """Like fake_embed, but records every text it was asked to embed."""
    seen: list[str] = []

    def embed(texts):
        seen.extend(texts)
        return [[float(len(t)), 0.0, 0.0] for t in texts]

    monkeypatch.setattr(store, "embed", embed)
    return seen


def test_first_index_embeds_every_symbol(counting_embed):
    stats = store.store_symbols([
        make_symbol("a", "def a(): return 1\n"),
        make_symbol("b", "def b(): return 2\n"),
    ])
    assert stats == {"new": 2, "changed": 0, "removed": 0, "unchanged": 0, "total": 2}
    assert len(counting_embed) == 2


def test_reindexing_identical_symbols_embeds_nothing(counting_embed):
    syms = [make_symbol("a", "def a(): return 1\n"), make_symbol("b", "def b(): return 2\n")]
    store.store_symbols(syms)
    counting_embed.clear()

    stats = store.store_symbols(syms)

    assert stats == {"new": 0, "changed": 0, "removed": 0, "unchanged": 2, "total": 2}
    assert counting_embed == []


def test_only_the_changed_symbol_is_re_embedded(counting_embed):
    store.store_symbols([
        make_symbol("a", "def a(): return 1\n"),
        make_symbol("b", "def b(): return 2\n"),
    ])
    counting_embed.clear()

    stats = store.store_symbols([
        make_symbol("a", "def a(): return 1\n"),      # unchanged
        make_symbol("b", "def b(): return 999\n"),    # body changed
    ])

    assert (stats["new"], stats["changed"], stats["unchanged"]) == (0, 1, 1)
    assert counting_embed == ["def b(): return 999\n"]


def test_deleted_symbols_are_removed(counting_embed):
    store.store_symbols([make_symbol("a", "x\n"), make_symbol("b", "y\n")])

    stats = store.store_symbols([make_symbol("a", "x\n")])

    assert stats == {"new": 0, "changed": 0, "removed": 1, "unchanged": 1, "total": 1}
    names = {h["name"] for h in _all_names()}
    assert names == {"a"}


def test_new_symbol_is_added_without_touching_the_rest(counting_embed):
    store.store_symbols([make_symbol("a", "x\n")])
    counting_embed.clear()

    stats = store.store_symbols([make_symbol("a", "x\n"), make_symbol("c", "z\n")])

    assert (stats["new"], stats["unchanged"], stats["total"]) == (1, 1, 2)
    assert counting_embed == ["z\n"]


def test_moved_symbol_counts_as_new_plus_removed(counting_embed):
    # id embeds the start line, so relocating code re-embeds it (known trade-off)
    store.store_symbols([make_symbol("a", "def a(): pass\n", start_line=5)])

    stats = store.store_symbols([make_symbol("a", "def a(): pass\n", start_line=20)])

    assert (stats["new"], stats["removed"], stats["unchanged"]) == (1, 1, 0)
    assert stats["total"] == 1


def _all_names():
    got = store._get_collection().get(include=["metadatas"])
    return got["metadatas"]
