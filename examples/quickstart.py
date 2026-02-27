"""Quickstart examples for aumai-toolretrieval.

Demonstrates building a vector-indexed tool registry, semantic search,
tag-based filtering, and direct tool lookup — all without any external
services or API keys.

Run this file directly to verify your installation:

    python examples/quickstart.py

The TF-IDF embedder is fitted in-memory, so results improve with larger
tool corpora.
"""

from aumai_toolretrieval.core import CosineSimilarity, SimpleEmbedder, ToolIndex
from aumai_toolretrieval.models import SearchQuery, SearchResult, ToolRecord


# ---------------------------------------------------------------------------
# Shared fixture: a small but varied tool registry
# ---------------------------------------------------------------------------


def _build_registry() -> ToolIndex:
    """Create and index a sample set of tools covering multiple domains."""
    index = ToolIndex()

    tools = [
        ToolRecord(
            tool_id="web-search-001",
            name="Web Search",
            description="Search the internet for up-to-date information given a query string.",
            tags=["search", "web", "retrieval"],
            capabilities=["query_web", "return_ranked_urls"],
        ),
        ToolRecord(
            tool_id="code-exec-001",
            name="Python Code Executor",
            description="Execute a Python code snippet in a sandboxed environment and return stdout.",
            tags=["code", "execution", "python"],
            capabilities=["run_python", "capture_output"],
        ),
        ToolRecord(
            tool_id="db-query-001",
            name="SQL Query Runner",
            description="Run parameterised SQL queries against a relational database and return rows.",
            tags=["database", "sql", "query"],
            capabilities=["execute_sql", "fetch_rows"],
        ),
        ToolRecord(
            tool_id="file-read-001",
            name="File Reader",
            description="Read the contents of a file from the local filesystem given an absolute path.",
            tags=["file", "filesystem", "io"],
            capabilities=["read_file", "detect_encoding"],
        ),
        ToolRecord(
            tool_id="email-send-001",
            name="Email Sender",
            description="Send an email message with optional attachments via SMTP.",
            tags=["email", "communication", "smtp"],
            capabilities=["send_email", "attach_files"],
        ),
        ToolRecord(
            tool_id="vector-search-001",
            name="Vector Similarity Search",
            description="Search a vector database for documents similar to a query embedding.",
            tags=["search", "vector", "retrieval", "embeddings"],
            capabilities=["embed_query", "cosine_search"],
        ),
        ToolRecord(
            tool_id="http-get-001",
            name="HTTP GET",
            description="Make an HTTP GET request to a URL and return the response body.",
            tags=["http", "web", "api"],
            capabilities=["http_request", "parse_json"],
        ),
        ToolRecord(
            tool_id="image-ocr-001",
            name="Image OCR",
            description="Extract text from an image file using optical character recognition.",
            tags=["ocr", "image", "vision"],
            capabilities=["extract_text", "detect_language"],
        ),
    ]

    for tool in tools:
        index.add_tool(tool)

    # build_index() fits the TF-IDF embedder and computes all embeddings.
    # Call this again after any add_tool() calls to keep embeddings current.
    index.build_index()
    return index


# ---------------------------------------------------------------------------
# Demo 1: Semantic search
# ---------------------------------------------------------------------------


def demo_semantic_search(index: ToolIndex) -> None:
    """Search for tools using a natural language query."""
    print("\n--- Demo 1: Semantic Search ---")

    query = SearchQuery(query_text="run python code and capture output", top_k=3)
    results: list[SearchResult] = index.search(query)

    print(f"Query : '{query.query_text}'  (top_k={query.top_k})")
    for result in results:
        print(
            f"  Rank {result.rank}: [{result.score:.4f}] "
            f"{result.tool.name} (id={result.tool.tool_id})"
        )

    # Second query tests a different part of the vocabulary
    query2 = SearchQuery(query_text="search web retrieve urls", top_k=3)
    results2 = index.search(query2)
    print(f"\nQuery : '{query2.query_text}'  (top_k={query2.top_k})")
    for result in results2:
        print(
            f"  Rank {result.rank}: [{result.score:.4f}] "
            f"{result.tool.name} (id={result.tool.tool_id})"
        )


# ---------------------------------------------------------------------------
# Demo 2: Tag-based pre-filtering combined with semantic scoring
# ---------------------------------------------------------------------------


def demo_tag_filtered_search(index: ToolIndex) -> None:
    """Narrow the candidate pool by tag before scoring."""
    print("\n--- Demo 2: Tag-Filtered Semantic Search ---")

    # Only consider tools tagged 'retrieval', then rank by semantic similarity
    query = SearchQuery(
        query_text="find similar documents",
        tags_filter=["retrieval"],
        top_k=5,
    )
    results = index.search(query)

    print(f"Query: '{query.query_text}'  tags_filter={query.tags_filter}")
    if results:
        for result in results:
            print(
                f"  Rank {result.rank}: [{result.score:.4f}] "
                f"{result.tool.name} | tags={result.tool.tags}"
            )
    else:
        print("  No results matched the tag filter.")


# ---------------------------------------------------------------------------
# Demo 3: Tag-only lookup (no embeddings involved)
# ---------------------------------------------------------------------------


def demo_tag_only_lookup(index: ToolIndex) -> None:
    """Return every tool that carries at least one of the given tags."""
    print("\n--- Demo 3: Tag-Only Lookup ---")

    tags = ["web", "http"]
    matching_tools: list[ToolRecord] = index.search_by_tags(tags)

    print(f"Tools with any tag in {tags}:")
    for tool in matching_tools:
        print(f"  {tool.name} (id={tool.tool_id}) | tags={tool.tags}")


# ---------------------------------------------------------------------------
# Demo 4: Direct tool retrieval and get_all_tools()
# ---------------------------------------------------------------------------


def demo_direct_retrieval(index: ToolIndex) -> None:
    """Retrieve a single tool by ID and list the full registry."""
    print("\n--- Demo 4: Direct Retrieval ---")

    tool = index.get_tool("email-send-001")
    if tool:
        print(f"get_tool('email-send-001'): {tool.name}")
        print(f"  Capabilities       : {tool.capabilities}")
        print(f"  Embedding dimensions: {len(tool.embedding or [])}")

    # Missing ID returns None — no exception raised
    missing = index.get_tool("does-not-exist")
    print(f"get_tool('does-not-exist'): {missing!r}")

    all_tools = index.get_all_tools()
    print(f"\nTotal tools in index: {len(all_tools)}")
    for tool in all_tools:
        print(f"  {tool.tool_id:<22} {tool.name}")


# ---------------------------------------------------------------------------
# Demo 5: SimpleEmbedder and CosineSimilarity standalone
# ---------------------------------------------------------------------------


def demo_embedder_and_similarity() -> None:
    """Use SimpleEmbedder and CosineSimilarity directly, without ToolIndex."""
    print("\n--- Demo 5: Embedder and Cosine Similarity ---")

    corpus = [
        "search the web for information",
        "execute python code in a sandbox",
        "send an email via smtp",
    ]

    embedder = SimpleEmbedder()
    embedder.fit(corpus)

    vec_search = embedder.embed("web search retrieval")
    vec_code = embedder.embed("python execution sandbox")

    print(f"Vocabulary size after fit : {len(embedder._vocab)}")
    print(f"Non-zero dims (web search): {sum(1 for v in vec_search if v != 0)}")

    similarity = CosineSimilarity()
    score_cross = similarity.compute(vec_search, vec_code)
    score_self = similarity.compute(vec_search, vec_search)

    print(f"Cosine(web search, python code) = {score_cross:.4f}  (expected: low)")
    print(f"Cosine(web search, itself)      = {score_self:.4f}  (expected: 1.0)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all aumai-toolretrieval quickstart demos."""
    print("=== aumai-toolretrieval Quickstart ===")

    index = _build_registry()
    print(f"Index built with {len(index.get_all_tools())} tools.")

    demo_semantic_search(index)
    demo_tag_filtered_search(index)
    demo_tag_only_lookup(index)
    demo_direct_retrieval(index)
    demo_embedder_and_similarity()

    print("\nAll demos completed successfully.")


if __name__ == "__main__":
    main()
