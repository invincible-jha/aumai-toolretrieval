"""Pydantic models for aumai-toolretrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "ToolRecord",
    "SearchQuery",
    "SearchResult",
]


class ToolRecord(BaseModel):
    """A tool registered in the retrieval index."""

    tool_id: str = Field(..., description="Globally unique identifier for the tool")
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Natural-language description of what the tool does")
    tags: list[str] = Field(default_factory=list, description="Categorical tags for tag-based filtering")
    capabilities: list[str] = Field(
        default_factory=list, description="List of capability strings the tool exposes"
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Dense embedding vector used for cosine similarity search. "
        "Set automatically by the index when a tool is added.",
    )


class SearchQuery(BaseModel):
    """Parameters for a similarity search against the tool index."""

    query_text: str = Field(..., description="Natural-language query")
    tags_filter: list[str] | None = Field(
        default=None,
        description="When set, only tools that share at least one tag are considered",
    )
    top_k: int = Field(default=10, ge=1, description="Number of results to return")


class SearchResult(BaseModel):
    """A single result from a similarity search."""

    tool: ToolRecord = Field(..., description="The matching tool record")
    score: float = Field(..., description="Cosine similarity score in [0, 1]")
    rank: int = Field(..., description="1-based rank within the result set")
