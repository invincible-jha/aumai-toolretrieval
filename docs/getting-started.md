# Getting Started with aumai-toolretrieval

This guide takes you from zero to a working semantic tool search in about 15 minutes.

---

## Prerequisites

- Python 3.11 or higher
- `pip` (comes with Python)
- Basic familiarity with the command line

No GPU, API key, or external service is required.

---

## Installation

### From PyPI (recommended)

```bash
pip install aumai-toolretrieval
```

Verify the installation:

```bash
aumai-toolretrieval --version
# aumai-toolretrieval, version 0.1.0
```

### From source

```bash
git clone https://github.com/aumai/aumai-toolretrieval.git
cd aumai-toolretrieval
pip install -e .
```

### Development mode (with test dependencies)

```bash
git clone https://github.com/aumai/aumai-toolretrieval.git
cd aumai-toolretrieval
pip install -e ".[dev]"
make lint   # ruff + mypy
make test   # pytest
```

---

## Your First Semantic Tool Search

This tutorial builds a small tool registry and demonstrates retrieval end-to-end.

### Step 1 — Create your tool definitions

Each tool is a JSON file. Create a directory and add some tools:

```bash
mkdir ~/my-tools
```

`~/my-tools/search.json`:

```json
{
  "tool_id": "web_search",
  "name": "Web Search",
  "description": "Search the internet for current information on any topic",
  "tags": ["search", "web", "retrieval", "information"],
  "capabilities": ["keyword search", "news lookup", "fact checking", "real-time data"]
}
```

`~/my-tools/email.json`:

```json
{
  "tool_id": "send_email",
  "name": "Send Email",
  "description": "Compose and send an email message to one or more recipients",
  "tags": ["communication", "email", "messaging", "notification"],
  "capabilities": ["send message", "attach files", "cc and bcc", "html content"]
}
```

`~/my-tools/database.json`:

```json
{
  "tool_id": "execute_sql",
  "name": "Database Query",
  "description": "Run parameterised SQL queries against a relational database",
  "tags": ["database", "sql", "data", "storage"],
  "capabilities": ["select records", "insert data", "update records", "aggregate functions"]
}
```

`~/my-tools/files.json`:

```json
{
  "tool_id": "file_operation",
  "name": "File Operations",
  "description": "Read, write, list, and delete files on the local filesystem",
  "tags": ["filesystem", "storage", "files", "io"],
  "capabilities": ["read file", "write file", "list directory", "delete file"]
}
```

### Step 2 — Build the index

```bash
aumai-toolretrieval index --tools-dir ~/my-tools
# Indexed 4 tool(s) from /home/user/my-tools.
```

The index is persisted to `~/.aumai/toolretrieval/index.json`. You only need to rebuild when tools change.

### Step 3 — Run your first search

```bash
aumai-toolretrieval search --query "I need to look up some facts online"
#   [1] Web Search (score=0.7231)
#       Search the internet for current information on any topic
#   [2] Database Query (score=0.1847)
#       Run parameterised SQL queries against a relational database
```

Try different queries to see how similarity scoring works:

```bash
aumai-toolretrieval search --query "write a file to disk"
aumai-toolretrieval search --query "notify the user"
aumai-toolretrieval search --query "how many records are in the users table"
```

### Step 4 — Filter by tags

Use `--tag` to restrict candidates before scoring:

```bash
aumai-toolretrieval search --query "reach out to someone" --tag communication
#   [1] Send Email (score=0.8103)
#       Compose and send an email message to one or more recipients
```

### Step 5 — Machine-readable output

For integrating into a pipeline or an agent loop:

```bash
aumai-toolretrieval search \
  --query "store a record" \
  --top-k 2 \
  --output-format json
```

Output:

```json
[
  {
    "rank": 1,
    "tool_id": "execute_sql",
    "name": "Database Query",
    "score": 0.6812,
    "description": "Run parameterised SQL queries against a relational database"
  },
  {
    "rank": 2,
    "tool_id": "file_operation",
    "name": "File Operations",
    "score": 0.3401,
    "description": "Read, write, list, and delete files on the local filesystem"
  }
]
```

---

## Common Patterns

### Pattern 1 — Dynamic tool injection into LLM prompts

The core use-case: retrieve the relevant tools for the current task and inject only those into the system prompt.

```python
from aumai_toolretrieval import ToolIndex, ToolRecord, SearchQuery
import json

# Load your registered tools from disk (or build the index in memory)
index = ToolIndex()
# ... populate index from JSON files or add_tool() calls ...
index.build_index()

def get_tools_for_task(task: str, top_k: int = 5) -> list[dict]:
    """Return the top-k most relevant tool descriptions for a task string."""
    results = index.search(SearchQuery(query_text=task, top_k=top_k))
    return [
        {
            "type": "function",
            "function": {
                "name": r.tool.tool_id,
                "description": r.tool.description,
                "parameters": {},
            },
        }
        for r in results
    ]

# In your agent loop:
task = "Search the web for the latest AI research papers"
tools = get_tools_for_task(task)
# Pass `tools` to your LLM API call's tools parameter
```

---

### Pattern 2 — Bulk loading from a directory

Register a large catalogue at startup:

```python
import json
from pathlib import Path
from aumai_toolretrieval import ToolIndex, ToolRecord

def load_tool_registry(tools_dir: str) -> ToolIndex:
    """Build a ToolIndex from all JSON files in a directory."""
    index = ToolIndex()
    path = Path(tools_dir)
    loaded = 0
    errors = 0
    for json_file in path.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            tool = ToolRecord.model_validate(data)
            index.add_tool(tool)
            loaded += 1
        except Exception as exc:
            print(f"Skipping {json_file.name}: {exc}")
            errors += 1
    index.build_index()
    print(f"Loaded {loaded} tools ({errors} errors)")
    return index

registry = load_tool_registry("./tool-definitions")
```

---

### Pattern 3 — Tag-based domain routing

Use tags to implement domain-aware routing before similarity search:

```python
from aumai_toolretrieval import SearchQuery

def search_in_domain(index, task: str, domain: str, top_k: int = 3):
    """Search only within a specific domain (tag category)."""
    domain_tags = {
        "communication": ["email", "messaging", "notification"],
        "data":          ["database", "sql", "storage"],
        "research":      ["search", "web", "retrieval"],
        "files":         ["filesystem", "files", "io"],
    }
    tags = domain_tags.get(domain, [])
    query = SearchQuery(query_text=task, tags_filter=tags or None, top_k=top_k)
    return index.search(query)

results = search_in_domain(index, "send a report to the team", domain="communication")
```

---

### Pattern 4 — Incremental index updates

Add new tools without rebuilding from scratch if you track additions:

```python
from aumai_toolretrieval import ToolRecord

new_tool = ToolRecord(
    tool_id="calendar_event",
    name="Create Calendar Event",
    description="Schedule a new event on a user's calendar",
    tags=["calendar", "scheduling", "productivity"],
    capabilities=["create event", "set reminder", "invite attendees"],
)

index.add_tool(new_tool)
index.build_index()  # Refits the embedder on the full updated corpus
```

---

### Pattern 5 — Serialise and restore the index

Persist the index to JSON for use across processes:

```python
import json

# Serialise
tools_data = [tool.model_dump(mode="json") for tool in index.get_all_tools()]
with open("tool_registry.json", "w") as fh:
    json.dump(tools_data, fh, indent=2)

# Restore in another process
from aumai_toolretrieval import ToolIndex, ToolRecord

index2 = ToolIndex()
with open("tool_registry.json") as fh:
    for entry in json.load(fh):
        index2.add_tool(ToolRecord.model_validate(entry))
index2.build_index()
```

---

## Troubleshooting FAQ

### The index command says "No JSON files found"

Check that the path you pass to `--tools-dir` contains `.json` files at the top level (not in subdirectories):

```bash
ls ~/my-tools/*.json
```

The `index` command uses `Path(tools_dir).glob("*.json")` — it does not recurse into subdirectories.

---

### Search returns no results

Two possible causes:

1. **Empty query embedding** — if the query contains no tokens that appear in the vocabulary, the embedder returns an empty vector and the search short-circuits. Try a query with words that appear in at least one tool description.

2. **Index not built** — if you added tools programmatically without calling `build_index()`, the stored embeddings will be `None`. Always call `build_index()` after `add_tool()`.

---

### Scores are all very low (below 0.1)

Low scores are normal when the query words have low overlap with any tool's vocabulary. The TF-IDF embedder is a bag-of-words model; it works best when the query and tool description share lexical terms. Try using domain-specific words that actually appear in your tool descriptions.

---

### After adding new tools, old results are unchanged

Call `build_index()` after every batch of `add_tool()` calls. The CLI `index` command does this automatically. In Python, the pattern is:

```python
index.add_tool(new_tool_a)
index.add_tool(new_tool_b)
index.build_index()  # required — refits the embedder on the full corpus
```

---

### The CLI persists state but my Python code doesn't see it

The CLI saves state to `~/.aumai/toolretrieval/index.json`. Python code using `ToolIndex` directly does not read this file automatically. To use the CLI-persisted state in Python, load it manually:

```python
import json
from pathlib import Path
from aumai_toolretrieval import ToolIndex, ToolRecord

index_file = Path.home() / ".aumai" / "toolretrieval" / "index.json"
index = ToolIndex()
if index_file.exists():
    for entry in json.loads(index_file.read_text()):
        index.add_tool(ToolRecord.model_validate(entry))
    index.build_index()
```

---

### mypy reports a type error on `SearchResult.tool`

`SearchResult.tool` is typed as `ToolRecord`. If mypy reports an issue, ensure you have `pydantic` installed and that `mypy` is configured to use the pydantic plugin or stub packages:

```toml
# pyproject.toml
[tool.mypy]
plugins = ["pydantic.mypy"]
```

---

## Next Steps

- [API Reference](api-reference.md) — complete documentation for every class and function.
- [Examples](../examples/quickstart.py) — runnable Python code demonstrating all features.
- [Contributing](../CONTRIBUTING.md) — how to add tools, fix bugs, or improve the embedder.
