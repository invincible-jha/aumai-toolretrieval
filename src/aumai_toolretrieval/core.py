"""Core logic for aumai-toolretrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from aumai_toolretrieval.models import SearchQuery, SearchResult, ToolRecord

__all__ = ["SimpleEmbedder", "CosineSimilarity", "ToolIndex"]


def _tokenize(text: str) -> list[str]:
    """Lower-case and split *text* into word tokens, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleEmbedder:
    """Bag-of-words TF-IDF style embedder with no external dependencies.

    The embedder builds a vocabulary from all documents seen via :meth:`fit` and
    encodes text as sparse TF-IDF vectors normalised to unit length.

    The vocabulary is fixed after :meth:`fit` is called; call :meth:`fit` again
    (with the full corpus) after adding new documents if you want the IDF
    weights to reflect the new data.
    """

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._num_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        """Build the vocabulary and IDF weights from *documents*.

        Args:
            documents: Corpus of text strings.
        """
        self._num_docs = len(documents)
        doc_freq: Counter[str] = Counter()

        for doc in documents:
            tokens = set(_tokenize(doc))
            doc_freq.update(tokens)

        vocab = sorted(doc_freq.keys())
        self._vocab = vocab
        self._vocab_index = {word: idx for idx, word in enumerate(vocab)}

        # Compute smooth IDF: log((N+1) / (df+1)) + 1
        for word, df in doc_freq.items():
            self._idf[word] = math.log((self._num_docs + 1) / (df + 1)) + 1.0

    def embed(self, text: str) -> list[float]:
        """Encode *text* as a unit-normalised TF-IDF vector.

        Args:
            text: Text to encode.

        Returns:
            A dense float vector with dimensionality equal to the vocabulary size.
            Returns a zero vector when the vocabulary has not been fitted yet.
        """
        if not self._vocab:
            return []

        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * len(self._vocab)

        tf: Counter[str] = Counter(tokens)
        vector: list[float] = [0.0] * len(self._vocab)

        for token, count in tf.items():
            idx = self._vocab_index.get(token)
            if idx is not None:
                idf_weight = self._idf.get(token, 1.0)
                vector[idx] = (count / len(tokens)) * idf_weight

        # L2 normalise
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector


class CosineSimilarity:
    """Compute cosine similarity between two equal-length float vectors using numpy."""

    @staticmethod
    def compute(vec_a: list[float], vec_b: list[float]) -> float:
        """Return the cosine similarity between *vec_a* and *vec_b*.

        Uses numpy for efficient dot product and norm computation.

        Args:
            vec_a: First vector.
            vec_b: Second vector.  Must have the same length as *vec_a*.

        Returns:
            A value in ``[-1, 1]``.  Returns 0.0 for zero-length or empty vectors.

        Raises:
            ValueError: When the vectors have different lengths.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
            )
        if not vec_a:
            return 0.0

        arr_a: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(vec_a, dtype=np.float64)
        arr_b: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(vec_b, dtype=np.float64)

        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


class ToolIndex:
    """Vector-indexed tool registry supporting semantic and tag-based search.

    Call :meth:`build_index` after adding all tools (or adding new batches) to
    refit the embedder and recompute all tool embeddings.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolRecord] = {}
        self._embedder: SimpleEmbedder = SimpleEmbedder()
        self._similarity: CosineSimilarity = CosineSimilarity()

    def add_tool(self, tool: ToolRecord) -> None:
        """Add or replace a tool in the index.

        Args:
            tool: The tool record to register.

        Note:
            You must call :meth:`build_index` after adding tools to update
            the embeddings; otherwise searches may use stale vectors.
        """
        self._tools[tool.tool_id] = tool

    def build_index(self) -> None:
        """Refit the embedder on all tool documents and update embeddings.

        Call this method after adding or removing tools to ensure all
        similarity scores reflect the current corpus.
        """
        tools = list(self._tools.values())
        if not tools:
            return

        documents = [self._tool_document(tool) for tool in tools]
        self._embedder.fit(documents)

        # Recompute embeddings for all tools
        for tool in tools:
            doc = self._tool_document(tool)
            embedding = self._embedder.embed(doc)
            updated = tool.model_copy(update={"embedding": embedding})
            self._tools[tool.tool_id] = updated

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Perform semantic similarity search against the tool index.

        Args:
            query: Search parameters.

        Returns:
            Up to ``query.top_k`` :class:`~aumai_toolretrieval.models.SearchResult`
            objects sorted by descending similarity score.
        """
        query_embedding = self._embedder.embed(query.query_text)
        if not query_embedding:
            return []

        candidates = list(self._tools.values())

        if query.tags_filter:
            filter_set = {tag.lower() for tag in query.tags_filter}
            candidates = [
                tool
                for tool in candidates
                if filter_set.intersection(t.lower() for t in tool.tags)
            ]

        scored: list[tuple[float, ToolRecord]] = []
        for tool in candidates:
            if tool.embedding is None:
                continue
            score = self._similarity.compute(query_embedding, tool.embedding)
            scored.append((score, tool))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, tool) in enumerate(scored[: query.top_k], start=1):
            results.append(SearchResult(tool=tool, score=score, rank=rank))

        return results

    def search_by_tags(self, tags: list[str]) -> list[ToolRecord]:
        """Return all tools that have at least one of the given tags.

        Args:
            tags: Tag strings to match (case-insensitive).

        Returns:
            Matching :class:`~aumai_toolretrieval.models.ToolRecord` objects.
        """
        filter_set = {tag.lower() for tag in tags}
        return [
            tool
            for tool in self._tools.values()
            if filter_set.intersection(t.lower() for t in tool.tags)
        ]

    def get_tool(self, tool_id: str) -> ToolRecord | None:
        """Retrieve a tool by its ID.

        Args:
            tool_id: The tool's unique identifier.

        Returns:
            The :class:`~aumai_toolretrieval.models.ToolRecord`, or *None*.
        """
        return self._tools.get(tool_id)

    def get_all_tools(self) -> list[ToolRecord]:
        """Return all tools currently in the index.

        Returns:
            A list of :class:`~aumai_toolretrieval.models.ToolRecord` objects.
        """
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_document(self, tool: ToolRecord) -> str:
        """Combine all searchable text fields of a tool into a single string."""
        parts = [tool.name, tool.description]
        parts.extend(tool.tags)
        parts.extend(tool.capabilities)
        return " ".join(parts)
