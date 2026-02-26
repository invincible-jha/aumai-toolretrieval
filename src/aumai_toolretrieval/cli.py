"""CLI entry point for aumai-toolretrieval."""

from __future__ import annotations

import json
from pathlib import Path

import click

from aumai_toolretrieval.core import ToolIndex
from aumai_toolretrieval.models import SearchQuery, ToolRecord

_DEFAULT_STATE_DIR = Path.home() / ".aumai" / "toolretrieval"
_DEFAULT_INDEX_FILE = _DEFAULT_STATE_DIR / "index.json"


def _load_index() -> ToolIndex:
    """Load a ToolIndex from persisted state."""
    index = ToolIndex()
    if _DEFAULT_INDEX_FILE.exists():
        raw: list[dict[str, object]] = json.loads(
            _DEFAULT_INDEX_FILE.read_text(encoding="utf-8")
        )
        for entry in raw:
            tool = ToolRecord.model_validate(entry)
            index.add_tool(tool)
        index.build_index()
    return index


def _save_index(index: ToolIndex) -> None:
    """Persist the tool index to disk."""
    _DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = [tool.model_dump(mode="json") for tool in index.get_all_tools()]
    _DEFAULT_INDEX_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@click.group()
@click.version_option()
def main() -> None:
    """AumAI Toolretrieval — vector-indexed semantic tool search."""


@main.command("index")
@click.option(
    "--tools-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing tool definition JSON files.",
)
def index_cmd(tools_dir: str) -> None:
    """Build or update the tool index from JSON definition files.

    Each JSON file in TOOLS_DIR must represent a ToolRecord (tool_id, name,
    description, tags, capabilities).
    """
    tools_path = Path(tools_dir)
    tool_files = list(tools_path.glob("*.json"))

    if not tool_files:
        click.echo(f"No JSON files found in {tools_dir}.", err=True)
        return

    index = ToolIndex()
    loaded_count = 0

    for tool_file in tool_files:
        try:
            raw = json.loads(tool_file.read_text(encoding="utf-8"))
            tool = ToolRecord.model_validate(raw)
            index.add_tool(tool)
            loaded_count += 1
        except Exception as exc:  # noqa: BLE001
            click.echo(f"  Skipping {tool_file.name}: {exc}", err=True)

    index.build_index()
    _save_index(index)
    click.echo(f"Indexed {loaded_count} tool(s) from {tools_dir}.")


@main.command("search")
@click.option("--query", required=True, help="Natural-language search query.")
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    type=int,
    help="Maximum number of results to return.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Filter results to tools with this tag (repeatable).",
)
@click.option(
    "--output-format",
    default="text",
    type=click.Choice(["text", "json"]),
    show_default=True,
)
def search_cmd(
    query: str, top_k: int, tags: tuple[str, ...], output_format: str
) -> None:
    """Search the tool index for tools matching a natural-language query."""
    index = _load_index()

    search_query = SearchQuery(
        query_text=query,
        tags_filter=list(tags) if tags else None,
        top_k=top_k,
    )
    results = index.search(search_query)

    if not results:
        click.echo("No matching tools found.")
        return

    if output_format == "json":
        data = [
            {
                "rank": r.rank,
                "tool_id": r.tool.tool_id,
                "name": r.tool.name,
                "score": round(r.score, 4),
                "description": r.tool.description,
            }
            for r in results
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        for result in results:
            click.echo(
                f"  [{result.rank}] {result.tool.name} "
                f"(score={result.score:.4f})\n"
                f"      {result.tool.description}"
            )


if __name__ == "__main__":
    main()
