# API Reference — aumai-toolretrieval

Complete documentation for all public classes, functions, and Pydantic models.

---

## Module: `aumai_toolretrieval.models`

Pydantic v2 models that represent the data structures flowing through the library.

---

### `ToolRecord`

A tool registered in the retrieval index.

```python
class ToolRecord(BaseModel):
    tool_id: str
    name: str
    description: str
    tags: list[str]
    capabilities: list[str]
    embedding: list[float] | None
```

**Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_id` | `str` | yes | Globally unique identifier for the tool. Used as the dictionary key inside `ToolIndex`. |
| `name` | `str` | yes | Human-readable display name (e.g. `"Web Search"`). |
| `description` | `str` | yes | Natural-language description of what the tool does. This is the primary source of semantic signal for search. |
| `tags` | `list[str]` | no | Categorical labels for tag-based pre-filtering (e.g. `["search", "web"]`). Case-insensitive comparisons. Defaults to `[]`. |
| `capabilities` | `list[str]` | no | Specific capability strings the tool exposes (e.g. `["keyword search", "news lookup"]`). Also included in the tool document for embedding. Defaults to `[]`. |
| `embedding` | `list[float] \| None` | no | Dense TF-IDF embedding vector. Set automatically by `ToolIndex.build_index()`. Do not set manually. Defaults to `None`. |

**Example:**

```python
from aumai_toolretrieval import ToolRecord

tool = ToolRecord(
    tool_id="web_search",
    name="Web Search",
    description="Search the internet for current information on any topic",
    tags=["search", "web", "retrieval"],
    capabilities=["keyword search", "news lookup", "fact checking"],
)
print(tool.tool_id)        # "web_search"
print(tool.embedding)      # None — set after build_index()
```

**Serialisation:**

```python
data = tool.model_dump(mode="json")   # dict with JSON-serialisable values
json_str = tool.model_dump_json()     # JSON string
restored = ToolRecord.model_validate(data)
```

---

### `SearchQuery`

Parameters for a similarity search against the tool index.

```python
class SearchQuery(BaseModel):
    query_text: str
    tags_filter: list[str] | None
    top_k: int
```

**Fields:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `query_text` | `str` | yes | — | Natural-language query string (e.g. `"find information online"`). |
| `tags_filter` | `list[str] \| None` | no | — | When set, only tools that share at least one tag with this list are included as candidates. Case-insensitive. Defaults to `None` (no filtering). |
| `top_k` | `int` | no | `>= 1` | Number of top results to return. Defaults to `10`. |

**Example:**

```python
from aumai_toolretrieval import SearchQuery

# Basic query
q1 = SearchQuery(query_text="send a message to the user")

# Filtered query, top 3 results
q2 = SearchQuery(
    query_text="notify the team",
    tags_filter=["communication", "email"],
    top_k=3,
)
```

---

### `SearchResult`

A single result from a similarity search.

```python
class SearchResult(BaseModel):
    tool: ToolRecord
    score: float
    rank: int
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `tool` | `ToolRecord` | The full tool record that matched. |
| `score` | `float` | Cosine similarity score in `[-1, 1]`; higher is more similar. In practice, TF-IDF unit vectors yield scores in `[0, 1]` for typical queries. |
| `rank` | `int` | 1-based rank within the result set. The highest-scoring result has `rank=1`. |

**Example:**

```python
from aumai_toolretrieval import ToolIndex, ToolRecord, SearchQuery

index = ToolIndex()
index.add_tool(ToolRecord(
    tool_id="web_search",
    name="Web Search",
    description="Search the internet for current information",
    tags=["search"],
    capabilities=["keyword search"],
))
index.build_index()

results = index.search(SearchQuery(query_text="find news online"))
for r in results:
    print(f"rank={r.rank}  score={r.score:.4f}  tool={r.tool.name}")
```

---

## Module: `aumai_toolretrieval.core`

Core business logic: embedding, similarity, and the tool index.

---

### `SimpleEmbedder`

Bag-of-words TF-IDF embedder with no external dependencies.

The embedder builds a vocabulary from all documents seen via `fit()` and encodes text as sparse TF-IDF vectors normalised to unit length. The vocabulary is fixed after `fit()` is called; call `fit()` again with the full corpus after adding new documents.

```python
class SimpleEmbedder:
    def fit(self, documents: list[str]) -> None: ...
    def embed(self, text: str) -> list[float]: ...
```

#### `SimpleEmbedder.__init__()`

No parameters.

```python
from aumai_toolretrieval import SimpleEmbedder
embedder = SimpleEmbedder()
```

---

#### `SimpleEmbedder.fit(documents)`

Build the vocabulary and IDF weights from a corpus of documents.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `documents` | `list[str]` | Corpus of text strings. Each string is tokenised independently. |

**Returns:** `None`

**Side effects:** Updates `_vocab`, `_vocab_index`, `_idf`, and `_num_docs` in-place.

**Notes:**
- Tokenisation: lower-case, split on non-alphanumeric boundaries (`re.findall(r"[a-z0-9]+", ...)`).
- IDF formula: `log((N+1) / (df+1)) + 1` (scikit-learn smooth IDF variant).
- Calling `fit()` a second time fully replaces the vocabulary.

**Example:**

```python
embedder = SimpleEmbedder()
embedder.fit([
    "search the web for information",
    "send an email to the user",
    "query the database for records",
])
```

---

#### `SimpleEmbedder.embed(text)`

Encode text as a unit-normalised TF-IDF vector.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Text to encode. Tokenised the same way as during `fit()`. |

**Returns:** `list[float]` — a dense float vector with dimensionality equal to vocabulary size. Returns an empty list `[]` if `fit()` has not been called. Returns a zero vector if none of the tokens in `text` appear in the vocabulary.

**Notes:**
- The returned vector is L2-normalised (unit magnitude) unless all values are zero.
- Out-of-vocabulary tokens contribute nothing to the vector.

**Example:**

```python
vec = embedder.embed("search for news online")
print(f"Vector dimension: {len(vec)}")
print(f"L2 norm: {sum(v*v for v in vec) ** 0.5:.4f}")  # ~1.0
```

---

### `CosineSimilarity`

Compute cosine similarity between two equal-length float vectors using numpy.

```python
class CosineSimilarity:
    @staticmethod
    def compute(vec_a: list[float], vec_b: list[float]) -> float: ...
```

#### `CosineSimilarity.compute(vec_a, vec_b)` (static method)

Return the cosine similarity between two vectors.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `vec_a` | `list[float]` | First vector. |
| `vec_b` | `list[float]` | Second vector. Must have the same length as `vec_a`. |

**Returns:** `float` — cosine similarity in `[-1, 1]`. Returns `0.0` for empty vectors or zero-magnitude vectors.

**Raises:**
- `ValueError` — if `len(vec_a) != len(vec_b)`.

**Notes:**
- Uses `numpy.dot` and `numpy.linalg.norm` for efficient computation.
- Because `SimpleEmbedder` produces L2-normalised vectors, the cosine similarity between two embedded texts is equivalent to their dot product.

**Example:**

```python
from aumai_toolretrieval import SimpleEmbedder, CosineSimilarity

embedder = SimpleEmbedder()
embedder.fit(["search the web", "send an email", "run a database query"])

vec_a = embedder.embed("search online for news")
vec_b = embedder.embed("web lookup for information")
vec_c = embedder.embed("send email notification")

print(CosineSimilarity.compute(vec_a, vec_b))  # high similarity ~0.6-0.9
print(CosineSimilarity.compute(vec_a, vec_c))  # lower similarity ~0.0-0.3
```

---

### `ToolIndex`

Vector-indexed tool registry supporting semantic and tag-based search.

Call `build_index()` after adding all tools (or after each batch of additions) to refit the embedder and recompute all tool embeddings.

```python
class ToolIndex:
    def add_tool(self, tool: ToolRecord) -> None: ...
    def build_index(self) -> None: ...
    def search(self, query: SearchQuery) -> list[SearchResult]: ...
    def search_by_tags(self, tags: list[str]) -> list[ToolRecord]: ...
    def get_tool(self, tool_id: str) -> ToolRecord | None: ...
    def get_all_tools(self) -> list[ToolRecord]: ...
```

#### `ToolIndex.__init__()`

No parameters. Initialises an empty in-memory registry.

```python
from aumai_toolretrieval import ToolIndex
index = ToolIndex()
```

---

#### `ToolIndex.add_tool(tool)`

Add or replace a tool in the registry.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tool` | `ToolRecord` | The tool record to register. If a tool with the same `tool_id` already exists it is replaced. |

**Returns:** `None`

**Important:** You must call `build_index()` after adding tools. Until `build_index()` is called, the new tool has `embedding=None` and will be excluded from scored search results.

---

#### `ToolIndex.build_index()`

Refit the embedder on all tool documents and update all stored embeddings.

**Parameters:** None

**Returns:** `None`

**Side effects:** Each `ToolRecord` in the registry is updated with a fresh embedding vector. The embedder's vocabulary is rebuilt from the current corpus.

**Notes:**
- A no-op if the registry is empty.
- Safe to call multiple times; each call fully replaces the previous index state.

---

#### `ToolIndex.search(query)`

Perform semantic similarity search against the tool index.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `query` | `SearchQuery` | Search parameters: query text, optional tag filter, top-k count. |

**Returns:** `list[SearchResult]` — up to `query.top_k` results sorted by descending cosine similarity score. Returns an empty list if the index is empty, the query embedding is empty, or no candidates survive the tag filter.

**Algorithm:**
1. Embed `query.query_text` using the fitted `SimpleEmbedder`.
2. If `query.tags_filter` is set, reduce candidates to tools with at least one matching tag (case-insensitive).
3. Compute cosine similarity between query vector and each candidate's embedding. Tools with `embedding=None` are skipped.
4. Sort by score descending; take the top `query.top_k`.
5. Wrap each result in a `SearchResult`.

**Example:**

```python
from aumai_toolretrieval import SearchQuery

results = index.search(SearchQuery(query_text="query the database", top_k=3))
for r in results:
    print(r.rank, r.tool.name, f"score={r.score:.4f}")
```

---

#### `ToolIndex.search_by_tags(tags)`

Return all tools that have at least one of the given tags.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tags` | `list[str]` | Tag strings to match. Comparison is case-insensitive. |

**Returns:** `list[ToolRecord]` — all matching tool records. Order is not guaranteed. Returns an empty list if no tools match.

**Notes:**
- Does not require `build_index()` to have been called.
- Uses set intersection: `filter_set.intersection(t.lower() for t in tool.tags)`.

**Example:**

```python
comm_tools = index.search_by_tags(["communication", "messaging"])
for tool in comm_tools:
    print(tool.tool_id)
```

---

#### `ToolIndex.get_tool(tool_id)`

Retrieve a tool by its unique identifier.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tool_id` | `str` | The tool's `tool_id`. Case-sensitive. |

**Returns:** `ToolRecord | None` — the matching record, or `None` if not found.

---

#### `ToolIndex.get_all_tools()`

Return all tools currently in the registry.

**Parameters:** None

**Returns:** `list[ToolRecord]` — a snapshot of all registered tools. Mutations to the returned list do not affect the index.

---

## Module: `aumai_toolretrieval.cli`

CLI entry point accessed via the `aumai-toolretrieval` command. Commands are documented in [CLI Reference](../README.md#cli-reference).

### Internal functions (non-public)

These are not part of the public API but are documented here for contributors.

| Function | Description |
|---|---|
| `_load_index()` | Deserialises `~/.aumai/toolretrieval/index.json` into a `ToolIndex`, calls `build_index()`. |
| `_save_index(index)` | Serialises a `ToolIndex` to `~/.aumai/toolretrieval/index.json`. |
| `_tokenize(text)` | Tokenisation helper: `re.findall(r"[a-z0-9]+", text.lower())`. Used by `SimpleEmbedder`. |

---

## Top-level `__init__.py` exports

```python
from aumai_toolretrieval import (
    CosineSimilarity,   # class
    SimpleEmbedder,     # class
    ToolIndex,          # class
    SearchQuery,        # Pydantic model
    SearchResult,       # Pydantic model
    ToolRecord,         # Pydantic model
)

__version__  # str, e.g. "0.1.0"
```

All six symbols are importable directly from `aumai_toolretrieval`.

---

## Type aliases and internal helpers

| Symbol | Type | Location | Description |
|---|---|---|---|
| `_DEFAULT_STATE_DIR` | `Path` | `cli.py` | `~/.aumai/toolretrieval` |
| `_DEFAULT_INDEX_FILE` | `Path` | `cli.py` | `~/.aumai/toolretrieval/index.json` |
| `_tokenize` | `function` | `core.py` | Lower-case word tokeniser; not exported |
