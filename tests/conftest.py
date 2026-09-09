"""Shared fixtures.

Every test runs against an in-memory ChromaDB (no disk, no cross-test bleed).
`fake_embed` swaps the real sentence-transformers model for a deterministic
bag-of-words vector so search tests are fast and stable; a couple of tests skip
it to exercise the real model.
"""
import re

import chromadb
import pytest
from chromadb.config import Settings

import store
from extract import Symbol


@pytest.fixture(autouse=True)
def ephemeral_db(monkeypatch):
    """Point store at a clean in-memory Chroma for each test.

    Chroma caches its in-memory system per settings, so an EphemeralClient is
    effectively a process singleton — reset() before each test clears any
    collections a previous test created.
    """
    client = chromadb.EphemeralClient(settings=Settings(allow_reset=True))
    client.reset()
    monkeypatch.setattr(store, "_client", client)
    return client


@pytest.fixture
def fake_embed(monkeypatch):
    """Deterministic 64-dim bag-of-words embedding — no model, no network."""
    def _vec(text: str) -> list[float]:
        v = [0.0] * 64
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[sum(ord(c) for c in tok) % 64] += 1.0
        return v

    def embed(texts):
        return [_vec(t) for t in texts]

    monkeypatch.setattr(store, "embed", embed)
    return embed


def make_symbol(name, source, *, kind="function", path="mod.py", start_line=1):
    return Symbol(
        kind=kind,
        name=name,
        path=path,
        start_line=start_line,
        end_line=start_line + source.count("\n"),
        source=source,
    )
