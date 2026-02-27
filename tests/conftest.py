"""Shared test fixtures for aumai-toolretrieval."""
from __future__ import annotations

import pytest

from aumai_toolretrieval.core import CosineSimilarity, SimpleEmbedder, ToolIndex
from aumai_toolretrieval.models import SearchQuery, SearchResult, ToolRecord


# ---------------------------------------------------------------------------
# Reusable tool records
# ---------------------------------------------------------------------------


@pytest.fixture()
def web_search_tool() -> ToolRecord:
    return ToolRecord(
        tool_id="web-search-001",
        name="Web Search",
        description="Search the web for current information on any topic",
        tags=["search", "web", "retrieval"],
        capabilities=["query", "filter", "paginate"],
    )


@pytest.fixture()
def email_tool() -> ToolRecord:
    return ToolRecord(
        tool_id="email-001",
        name="Send Email",
        description="Send email messages to one or more recipients",
        tags=["email", "communication", "messaging"],
        capabilities=["send", "attach", "cc", "bcc"],
    )


@pytest.fixture()
def database_tool() -> ToolRecord:
    return ToolRecord(
        tool_id="db-001",
        name="Database Query",
        description="Execute SQL queries against the database",
        tags=["database", "sql", "data"],
        capabilities=["select", "insert", "update", "delete"],
    )


@pytest.fixture()
def file_tool() -> ToolRecord:
    return ToolRecord(
        tool_id="file-001",
        name="File Operations",
        description="Read, write, list, and delete files on the filesystem",
        tags=["file", "filesystem", "io"],
        capabilities=["read", "write", "list", "delete"],
    )


@pytest.fixture()
def populated_index(
    web_search_tool: ToolRecord,
    email_tool: ToolRecord,
    database_tool: ToolRecord,
    file_tool: ToolRecord,
) -> ToolIndex:
    """Return a ToolIndex populated with 4 tools and the index built."""
    index = ToolIndex()
    for tool in [web_search_tool, email_tool, database_tool, file_tool]:
        index.add_tool(tool)
    index.build_index()
    return index


@pytest.fixture()
def embedder_fitted() -> SimpleEmbedder:
    """Return a SimpleEmbedder fitted on a small corpus."""
    embedder = SimpleEmbedder()
    embedder.fit(
        [
            "search the web for information",
            "send email to recipients",
            "execute database queries",
            "file read write operations",
        ]
    )
    return embedder
