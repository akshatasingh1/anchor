import hashlib
import os
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

_model = SentenceTransformer("all-MiniLM-L6-v2")

# Keep the vector store in a fixed location (repo root by default) so the CLI and
# the MCP server hit the same DB no matter which directory they're launched from.
# Override with CODEBASE_RAG_DB to point at a different index.
_DB_PATH = os.environ.get("CODEBASE_RAG_DB") or str(Path(__file__).resolve().parent.parent / "chroma")
_client = chromadb.PersistentClient(path=_DB_PATH)

COLLECTION = "symbols"


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of text strings into a list of vectors."""
    return _model.encode(texts).tolist()


def _symbol_id(s) -> str:
    return f"{s.path}:{s.start_line}:{s.name}"


def _content_hash(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()


def store_symbols(symbols) -> dict:
    """Incrementally sync freshly extracted symbols into Chroma.

    Only new or body-changed symbols are embedded; symbols that no longer exist
    in the source are deleted. Returns a summary:
    ``{"new", "changed", "removed", "unchanged", "total"}``.
    """
    collection = _client.get_or_create_collection(COLLECTION)

    # What's already indexed: id -> stored content hash.
    existing = collection.get(include=["metadatas"])
    stored_hash = {
        _id: (meta or {}).get("hash")
        for _id, meta in zip(existing["ids"], existing["metadatas"])
    }

    # De-dupe the fresh symbols by id (last one wins) before diffing.
    fresh = {_symbol_id(s): s for s in symbols}

    to_upsert = []  # (id, symbol, hash) needing (re)embedding
    unchanged = 0
    for _id, s in fresh.items():
        h = _content_hash(s.source)
        if stored_hash.get(_id) == h:
            unchanged += 1
        else:
            to_upsert.append((_id, s, h))

    removed_ids = [_id for _id in stored_hash if _id not in fresh]
    new_count = sum(1 for _id, _, _ in to_upsert if _id not in stored_hash)
    changed_count = len(to_upsert) - new_count

    if removed_ids:
        collection.delete(ids=removed_ids)

    if to_upsert:
        vectors = embed([s.source for _, s, _ in to_upsert])
        collection.upsert(
            ids=[_id for _id, _, _ in to_upsert],
            embeddings=vectors,
            documents=[s.source for _, s, _ in to_upsert],
            metadatas=[
                {
                    "name": s.name,
                    "kind": s.kind,
                    "path": s.path,
                    "start_line": s.start_line,
                    "hash": h,
                }
                for _, s, h in to_upsert
            ],
        )

    return {
        "new": new_count,
        "changed": changed_count,
        "removed": len(removed_ids),
        "unchanged": unchanged,
        "total": collection.count(),
    }


def _get_collection():
    """Return the symbols collection, or None if nothing has been indexed yet."""
    try:
        return _client.get_collection(COLLECTION)
    except Exception:
        return None


def index_info() -> dict:
    """Summary of what's indexed: symbol count and the files covered."""
    collection = _get_collection()
    if collection is None:
        return {"indexed": False, "symbol_count": 0, "files": [], "db_path": _DB_PATH}
    got = collection.get(include=["metadatas"])
    files = sorted({m["path"] for m in got["metadatas"]})
    return {
        "indexed": True,
        "symbol_count": collection.count(),
        "files": files,
        "db_path": _DB_PATH,
    }


# Words too generic to carry search intent — dropped before keyword matching.
_STOPWORDS = {
    "how", "does", "do", "did", "the", "a", "an", "is", "are", "was", "were",
    "what", "when", "where", "why", "which", "who", "to", "of", "in", "on", "for",
    "and", "or", "not", "this", "that", "with", "from", "as", "at", "by", "be",
    "it", "its", "i", "we", "you", "get", "set", "use", "used", "using", "call",
    "called", "code", "function", "method", "class", "handle", "handled",
}


def _tokens(text: str) -> list[str]:
    """Split an identifier or a phrase into lowercase word tokens.

    Handles camelCase, snake_case, dotted names, and plain prose:
    'AuthManager.login' -> ['auth', 'manager', 'login'].
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", spaced) if t]


def _keyword_scores(question: str, by_id: dict) -> dict[str, float]:
    """Score each symbol by how well the query words match its qualified name.

    Exact token hit counts for more than a partial (substring) hit. Symbols with
    no overlap are left out.
    """
    q_tokens = [t for t in _tokens(question) if len(t) > 1 and t not in _STOPWORDS]
    if not q_tokens:
        return {}

    scores: dict[str, float] = {}
    for _id, entry in by_id.items():
        name_tokens = set(_tokens(entry["meta"]["name"]))
        s = 0.0
        for qt in q_tokens:
            if qt in name_tokens:
                s += 3.0
            elif len(qt) >= 4 and any(qt in nt or nt in qt for nt in name_tokens):
                s += 1.0
        if s > 0:
            scores[_id] = s
    return scores


def _rrf(*rankings: dict[str, int], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: blend several 1-based rankings into one score."""
    fused: dict[str, float] = {}
    for rank_map in rankings:
        for _id, rank in rank_map.items():
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank)
    return fused


def search(question: str, n: int = 5, candidates: int = 20):
    """Hybrid search: fuse vector similarity with symbol-name keyword matching.

    Over-fetches `candidates` from each signal, blends them with Reciprocal Rank
    Fusion, and returns the top `n` symbols.
    """
    collection = _get_collection()
    if collection is None:
        raise RuntimeError(
            "Nothing indexed yet. Run `python src/main.py index <repo>` first."
        )

    # Full catalog for keyword scoring — fine at repo scale (hundreds/low thousands
    # of symbols); revisit if indexes grow much larger.
    catalog = collection.get(include=["metadatas", "documents"])
    by_id = {
        _id: {"meta": m, "source": d}
        for _id, m, d in zip(catalog["ids"], catalog["metadatas"], catalog["documents"])
    }
    if not by_id:
        return []

    # Vector ranking.
    query_vec = embed([question])          # same model as indexing -> same space
    res = collection.query(
        query_embeddings=query_vec,
        n_results=min(candidates, len(by_id)),
    )
    vec_ids = res["ids"][0]
    vec_dist = dict(zip(vec_ids, res["distances"][0]))
    vec_rank = {_id: r for r, _id in enumerate(vec_ids, 1)}

    # Keyword ranking (symbol name only).
    kw = _keyword_scores(question, by_id)
    kw_ids = sorted(kw, key=kw.get, reverse=True)[:candidates]
    kw_rank = {_id: r for r, _id in enumerate(kw_ids, 1)}

    fused = _rrf(vec_rank, kw_rank)
    ranked = sorted(fused, key=fused.get, reverse=True)[:n]

    hits = []
    for _id in ranked:
        m = by_id[_id]["meta"]
        hits.append({
            "name": m["name"],
            "kind": m["kind"],
            "location": f'{m["path"]}:{m["start_line"]}',
            "distance": vec_dist.get(_id),
            "score": fused[_id],
            "source": by_id[_id]["source"],
        })
    return hits
