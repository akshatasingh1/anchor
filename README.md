# Codebase RAG Assistant


A CLI tool that answers natural-language questions about a codebase by retrieving the most relevant functions and classes — with exact `file:line` locations — instead of guessing or grepping.

Ask something like *"how does retry logic work"* and get back ranked, real code locations, even when the answer doesn't contain the word "retry."


## Why
Most "chat with your codebase" tools chunk source files into fixed-size blocks of text before embedding them. That splits functions in half and loses structure. This project instead parses source with **tree-sitter** and chunks by **symbol** — one function or class per chunk — so every retrieved result is a complete, meaningful unit with a precise location.


## How it works


1. **Extract** — walk a repo, parse each Python file with tree-sitter, and pull out every function, class, and method with its qualified name (e.g. `AuthManager.login`) and exact `file:line` range.
2. **Embed** — turn each symbol's source into a vector using a local `sentence-transformers` model (`all-MiniLM-L6-v2`). Runs fully offline, no API keys.
3. **Store** — persist symbols and their vectors in a local ChromaDB collection. Re-indexing is **incremental**: each symbol's body is hashed, so a repeat run only re-embeds what changed and drops symbols that disappeared.
4. **Search** — embed the user's question with the same model and rank symbols by a **hybrid** of vector similarity and symbol-name keyword matching (fused with Reciprocal Rank Fusion), so both *"how are retries handled"* and *"HTTPAdapter.send"* land the right code.
5. **MCP server** — the same search is exposed as an MCP tool, so AI coding assistants like Claude Code can call it directly to ground their answers in real source locations.


## Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # optional: adds the `codebase-rag` command
```

## Usage

Index a repository:

```bash
python src/main.py index /path/to/repo        # or: codebase-rag index /path/to/repo
```

Test files (`test_*.py`, `*_test.py`, `conftest.py`, anything under `test/` or `tests/`) are excluded by default. Pass `--tests` to include them.

Query it:

```bash
python src/main.py query "how does the library retry failed requests"
```

Each result shows the symbol's kind, qualified name, fused relevance score, vector distance, `file:line` location, and a source preview.


## MCP server

Exposes the same retrieval to MCP-compatible clients like Claude Code:

- `search_code(question, n)` — ranked symbols with `file:line` locations and source
- `index_status()` — which repo/files are currently indexed

```bash
# after installing requirements.txt and indexing a repo.
# use absolute paths (and the venv's python) so it works from any client cwd:
claude mcp add codebase-rag -- /abs/path/.venv/bin/python /abs/path/src/mcp_server.py
```

Once registered, an assistant with access to these tools can answer questions about the indexed repo using real, current source locations instead of relying on memory.

The vector store lives at `<repo>/chroma` by default so the CLI and the server share one index regardless of working directory. Point at a different index with the `CODEBASE_RAG_DB` environment variable.

## Project structure

```
src/
  extract.py      # tree-sitter symbol extraction
  store.py        # embedding + ChromaDB storage/search
  main.py         # Typer CLI (index, query)
  mcp_server.py   # MCP server exposing search_code / index_status
tests/            # pytest suite
pyproject.toml    # packaging + pytest config
requirements.txt  # runtime dependencies
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs against an in-memory ChromaDB and stubs the embedding model, so it's fast; a couple of tests exercise the real `sentence-transformers` model for a semantic-ranking sanity check.

## Status

Working end to end on real repositories (tested on `requests`, 800+ symbols extracted and searchable). Currently supports Python only.

## Roadmap

- [ ] Multi-language support (JavaScript/TypeScript via tree-sitter grammars)
- [ ] Dependency/call-graph aware retrieval
- [x] `--no-tests` flag to exclude test files from results (default; `--tests` to include)
- [ ] `--watch` mode: re-index on file save
- [x] Incremental indexing (re-embed only changed symbols)
- [x] Installable pip package
- [x] Hybrid search (embeddings + keyword/symbol-name matching)

## Stack

Python, tree-sitter, ChromaDB, sentence-transformers, Typer, Rich, MCP
