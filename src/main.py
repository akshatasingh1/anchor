import typer
from extract import extract_repo
from store import store_symbols, search

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

app = typer.Typer(help="RAG assistant for navigating a codebase.")


@app.command()
def index(repo: str):
    """Index a repo: extract symbols, embed only what changed, store them."""
    symbols = extract_repo(repo)
    print(f"Extracted {len(symbols)} symbols. Syncing index...")
    s = store_symbols(symbols)
    print(
        f"Done: +{s['new']} new, ~{s['changed']} changed, -{s['removed']} removed, "
        f"{s['unchanged']} unchanged  ({s['total']} total in Chroma)."
    )


@app.command()
def query(text: str, n: int = 5):
    """Search the indexed codebase for a task or question."""
    try:
        hits = search(text, n)
    except RuntimeError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    console.print(f"\n[bold]Top {len(hits)} matches for:[/bold] [cyan]{text}[/cyan]\n")
    for rank, h in enumerate(hits, 1):
        dist = f" · dist {h['distance']:.3f}" if h.get("distance") is not None else " · name-only"
        meta = f"[dim]({h['kind']}) · score {h['score']:.4f}{dist}[/dim]"
        header = f"[bold green]{rank}. {h['name']}[/bold green]  {meta}  →  [yellow]{h['location']}[/yellow]"
        console.print(header)
        # show the matched symbol as a preview
        preview = "\n".join(h["source"].splitlines()[:8])
        console.print(Syntax(preview, "python", theme="ansi_dark", line_numbers=False))
        console.print()

def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()

