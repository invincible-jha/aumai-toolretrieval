"""Comprehensive CLI tests for aumai-toolretrieval."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_toolretrieval.cli import main
from aumai_toolretrieval.models import ToolRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tool_json(directory: Path, tool: dict) -> Path:
    """Write a tool dict as a JSON file in *directory*."""
    path = directory / f"{tool['tool_id']}.json"
    path.write_text(json.dumps(tool), encoding="utf-8")
    return path


SAMPLE_TOOL = {
    "tool_id": "test-001",
    "name": "Test Tool",
    "description": "A tool for testing search and retrieval operations",
    "tags": ["test", "search", "retrieval"],
    "capabilities": ["query", "filter"],
}

SAMPLE_TOOL_2 = {
    "tool_id": "test-002",
    "name": "Email Sender",
    "description": "Send email messages to recipients",
    "tags": ["email", "communication"],
    "capabilities": ["send", "attach"],
}


# ---------------------------------------------------------------------------
# Tests for --version
# ---------------------------------------------------------------------------


class TestCliVersion:
    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "toolretrieval" in result.output.lower() or "AumAI" in result.output


# ---------------------------------------------------------------------------
# Tests for `index` command
# ---------------------------------------------------------------------------


class TestIndexCommand:
    def test_index_no_tools_dir_required(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["index"])
        assert result.exit_code != 0

    def test_index_empty_directory(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("tools").mkdir()
            result = runner.invoke(main, ["index", "--tools-dir", "tools"])
            assert "No JSON files found" in result.output

    def test_index_single_tool(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            tools_dir = Path("tools")
            tools_dir.mkdir()
            _write_tool_json(tools_dir, SAMPLE_TOOL)
            result = runner.invoke(main, ["index", "--tools-dir", "tools"])
            assert result.exit_code == 0
            assert "Indexed 1 tool" in result.output

    def test_index_multiple_tools(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            tools_dir = Path("tools")
            tools_dir.mkdir()
            _write_tool_json(tools_dir, SAMPLE_TOOL)
            _write_tool_json(tools_dir, SAMPLE_TOOL_2)
            result = runner.invoke(main, ["index", "--tools-dir", "tools"])
            assert result.exit_code == 0
            assert "Indexed 2 tool" in result.output

    def test_index_skips_invalid_json(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            tools_dir = Path("tools")
            tools_dir.mkdir()
            _write_tool_json(tools_dir, SAMPLE_TOOL)
            (tools_dir / "invalid.json").write_text("not valid json", encoding="utf-8")
            result = runner.invoke(main, ["index", "--tools-dir", "tools"])
            assert result.exit_code == 0
            # One valid tool indexed; invalid one skipped
            assert "Indexed 1 tool" in result.output

    def test_index_skips_missing_required_fields(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            tools_dir = Path("tools")
            tools_dir.mkdir()
            # Missing required 'name'
            bad_tool = {"tool_id": "bad", "description": "missing name"}
            _write_tool_json(tools_dir, bad_tool)
            _write_tool_json(tools_dir, SAMPLE_TOOL)
            result = runner.invoke(main, ["index", "--tools-dir", "tools"])
            assert result.exit_code == 0
            assert "Indexed 1 tool" in result.output


# ---------------------------------------------------------------------------
# Tests for `search` command
# ---------------------------------------------------------------------------


class TestSearchCommand:
    def _setup_index(self, runner: CliRunner) -> None:
        """Build an index with two tools in the isolated filesystem."""
        tools_dir = Path("tools")
        tools_dir.mkdir()
        _write_tool_json(tools_dir, SAMPLE_TOOL)
        _write_tool_json(tools_dir, SAMPLE_TOOL_2)
        runner.invoke(main, ["index", "--tools-dir", "tools"])

    def test_search_no_query_required(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["search"])
        assert result.exit_code != 0

    def test_search_returns_results_text_format(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main, ["search", "--query", "search retrieval", "--output-format", "text"]
            )
            assert result.exit_code == 0

    def test_search_returns_results_json_format(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main, ["search", "--query", "search retrieval", "--output-format", "json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)

    def test_search_json_result_structure(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main,
                ["search", "--query", "testing search", "--output-format", "json", "--top-k", "1"],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) >= 1
            item = data[0]
            assert "rank" in item
            assert "tool_id" in item
            assert "name" in item
            assert "score" in item
            assert "description" in item

    def test_search_top_k_limits_results(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main,
                ["search", "--query", "operations", "--top-k", "1", "--output-format", "json"],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) <= 1

    def test_search_with_tag_filter(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main,
                [
                    "search",
                    "--query",
                    "send messages",
                    "--tag",
                    "email",
                    "--output-format",
                    "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            # All results must have the 'email' tag
            for item in data:
                tool_id = item["tool_id"]
                assert tool_id == "test-002"

    def test_search_no_results_message(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            # Use a tag that does not exist in any indexed tool
            result = runner.invoke(
                main,
                ["search", "--query", "find tools", "--tag", "zzz-nonexistent-tag-xyz"],
            )
            assert result.exit_code == 0
            assert "No matching tools found" in result.output

    def test_search_multiple_tag_filters(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main,
                [
                    "search",
                    "--query",
                    "operations",
                    "--tag",
                    "search",
                    "--tag",
                    "email",
                    "--output-format",
                    "json",
                ],
            )
            assert result.exit_code == 0

    def test_search_text_output_has_rank_and_score(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            self._setup_index(runner)
            result = runner.invoke(
                main, ["search", "--query", "search testing tool", "--output-format", "text"]
            )
            assert result.exit_code == 0
            assert "score=" in result.output


# ---------------------------------------------------------------------------
# Tests for models
# ---------------------------------------------------------------------------


class TestModels:
    def test_tool_record_valid(self) -> None:
        tool = ToolRecord(**SAMPLE_TOOL)
        assert tool.tool_id == "test-001"
        assert tool.embedding is None

    def test_tool_record_default_tags_empty(self) -> None:
        tool = ToolRecord(tool_id="t", name="T", description="D")
        assert tool.tags == []
        assert tool.capabilities == []

    def test_search_query_default_top_k(self) -> None:
        from aumai_toolretrieval.models import SearchQuery

        q = SearchQuery(query_text="hello")
        assert q.top_k == 10

    def test_search_query_top_k_min_one(self) -> None:
        from aumai_toolretrieval.models import SearchQuery
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SearchQuery(query_text="hello", top_k=0)

    def test_search_result_rank_and_score(self) -> None:
        from aumai_toolretrieval.models import SearchResult

        tool = ToolRecord(**SAMPLE_TOOL)
        result = SearchResult(tool=tool, score=0.85, rank=1)
        assert result.rank == 1
        assert result.score == 0.85
        assert result.tool.tool_id == "test-001"
