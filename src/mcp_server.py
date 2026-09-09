"""MCP server exposing the codebase RAG retrieval as tools.

An MCP-compatible client (Claude Code, etc.) can call these to ground answers in
real, current source locations from an indexed repo instead of relying on memory.

Register it once (after indexing a repo with `python src/main.py index <repo>`):

    claude mcp add codebase-rag -- python src/mcp_server.py

Tools:
    search_code(question, n)  -> ranked symbols with file:line locations + source
    index_status()            -> what repo/files are currently indexed
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from store import search, index_info

mcp = MCPServer(
    "codebase-rag",
    instructions=(
        "Semantic search over an indexed codebase. Call search_code to find the "
        "functions and classes relevant to a question before answering how the "
        "code works; every result carries an exact file:line location."
    ),
)

# Cap how much source a single hit returns so a huge function can't blow up the
# caller's context. The location is always exact, so the full body is one read away.
_MAX_SOURCE_LINES = 60


def _clip(source: str) -> str:
    lines = source.splitlines()
    if len(lines) <= _MAX_SOURCE_LINES:
        return source
    shown = "\n".join(lines[:_MAX_SOURCE_LINES])
    return f"{shown}\n... (+{len(lines) - _MAX_SOURCE_LINES} more lines; open the file at the location above)"


@mcp.tool()
def search_code(question: str, n: int = 5) -> str:
    """Find the functions/classes in the indexed codebase most relevant to a
    natural-language question. Ranking fuses semantic similarity with
    symbol-name keyword matching.

    Use this to locate real source before answering questions about how the code
    works. Semantic matching means "how are retries handled" finds the code even
    when it never says "retry"; name matching means "AuthManager.login" reliably
    surfaces that exact symbol.

    Args:
        question: What you're looking for, in plain language.
        n: How many results to return (1-20, default 5).
    """
    n = max(1, min(n, 20))
    try:
        hits = search(question, n)
    except RuntimeError as e:
        return str(e)

    if not hits:
        return "No matching symbols found."

    blocks = []
    for rank, h in enumerate(hits, 1):
        dist = f", distance {h['distance']:.3f}" if h.get("distance") is not None else ", name match"
        blocks.append(
            f"{rank}. {h['name']}  ({h['kind']})  —  {h['location']}  "
            f"[score {h['score']:.4f}{dist}]\n"
            f"```python\n{_clip(h['source'])}\n```"
        )
    return "\n\n".join(blocks)


@mcp.tool()
def index_status() -> str:
    """Report whether a codebase is indexed, and which files it covers."""
    info = index_info()
    if not info["indexed"]:
        return (
            "No codebase indexed yet.\n"
            f"DB path: {info['db_path']}\n"
            "Run: python src/main.py index <repo>"
        )
    files = "\n".join(f"  {p}" for p in info["files"])
    return (
        f"{info['symbol_count']} symbols indexed across {len(info['files'])} files "
        f"(DB: {info['db_path']}):\n{files}"
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport
