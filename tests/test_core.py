"""Comprehensive tests for aumai_toolretrieval core module."""
from __future__ import annotations

import math

import pytest

from aumai_toolretrieval.core import CosineSimilarity, SimpleEmbedder, ToolIndex, _tokenize
from aumai_toolretrieval.models import SearchQuery, SearchResult, ToolRecord


# ---------------------------------------------------------------------------
# Tests for _tokenize helper
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_words(self) -> None:
        tokens = _tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_lowercases_text(self) -> None:
        tokens = _tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        tokens = _tokenize("hello, world!")
        assert tokens == ["hello", "world"]

    def test_alphanumeric_tokens(self) -> None:
        tokens = _tokenize("gpt-4o model123")
        assert tokens == ["gpt", "4o", "model123"]

    def test_empty_string(self) -> None:
        tokens = _tokenize("")
        assert tokens == []

    def test_punctuation_only(self) -> None:
        tokens = _tokenize("... --- !!!")
        assert tokens == []

    def test_numbers(self) -> None:
        tokens = _tokenize("version 3 14159")
        assert tokens == ["version", "3", "14159"]

    def test_mixed_separators(self) -> None:
        tokens = _tokenize("search_web.query.execute")
        assert "search" in tokens
        assert "web" in tokens


# ---------------------------------------------------------------------------
# Tests for SimpleEmbedder
# ---------------------------------------------------------------------------


class TestSimpleEmbedder:
    def test_embed_before_fit_returns_empty(self) -> None:
        embedder = SimpleEmbedder()
        result = embedder.embed("hello world")
        assert result == []

    def test_fit_builds_vocabulary(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["hello world", "foo bar"])
        assert len(embedder._vocab) > 0

    def test_embed_returns_correct_dimension(self, embedder_fitted: SimpleEmbedder) -> None:
        vec = embedder_fitted.embed("search the web")
        assert len(vec) == len(embedder_fitted._vocab)

    def test_embed_is_unit_normalised(self, embedder_fitted: SimpleEmbedder) -> None:
        vec = embedder_fitted.embed("search the web")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-9

    def test_embed_empty_string_returns_zeros(self, embedder_fitted: SimpleEmbedder) -> None:
        vec = embedder_fitted.embed("")
        assert vec == [0.0] * len(embedder_fitted._vocab)

    def test_embed_unknown_tokens_returns_zeros(self, embedder_fitted: SimpleEmbedder) -> None:
        vec = embedder_fitted.embed("xyzzy quux blorb")
        assert all(v == 0.0 for v in vec)
        assert len(vec) == len(embedder_fitted._vocab)

    def test_fit_single_document(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["single document text"])
        vec = embedder.embed("single document")
        assert len(vec) > 0

    def test_fit_empty_corpus(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit([])
        # No vocab -> embed returns []
        assert embedder.embed("anything") == []

    def test_idf_weights_computed(self, embedder_fitted: SimpleEmbedder) -> None:
        # IDF weights must be > 0 for all vocab words
        for word, idf in embedder_fitted._idf.items():
            assert idf > 0, f"IDF for '{word}' should be positive"

    def test_fit_resets_vocabulary(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["apple banana"])
        first_vocab = set(embedder._vocab)
        embedder.fit(["cherry date elderberry"])
        second_vocab = set(embedder._vocab)
        assert first_vocab != second_vocab

    def test_embed_known_word_nonzero(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["search query web"])
        vec = embedder.embed("search")
        assert any(v != 0.0 for v in vec)

    def test_duplicate_documents_handled(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["hello world", "hello world", "different text"])
        # Should not raise; vocabulary built from 3 docs
        assert embedder._num_docs == 3

    def test_idf_formula_smooth(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["word", "word", "other"])
        # 'word' appears in 2 of 3 docs -> IDF = log((3+1)/(2+1)) + 1
        expected_idf = math.log((3 + 1) / (2 + 1)) + 1.0
        assert abs(embedder._idf["word"] - expected_idf) < 1e-9

    def test_vocab_is_sorted(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["zebra apple mango"])
        assert embedder._vocab == sorted(embedder._vocab)

    def test_embed_only_known_vocab(self) -> None:
        embedder = SimpleEmbedder()
        embedder.fit(["cat dog"])
        # 'cat' and 'dog' are in vocab; 'xray' is not
        vec_known = embedder.embed("cat")
        vec_unknown = embedder.embed("xray")
        assert any(v != 0.0 for v in vec_known)
        assert all(v == 0.0 for v in vec_unknown)


# ---------------------------------------------------------------------------
# Tests for CosineSimilarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [1.0, 0.5, 0.25]
        score = CosineSimilarity.compute(vec, vec)
        assert abs(score - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        score = CosineSimilarity.compute([1.0, 0.0], [0.0, 1.0])
        assert abs(score - 0.0) < 1e-9

    def test_opposite_vectors(self) -> None:
        score = CosineSimilarity.compute([1.0, 0.0], [-1.0, 0.0])
        assert abs(score - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self) -> None:
        score = CosineSimilarity.compute([0.0, 0.0], [1.0, 1.0])
        assert score == 0.0

    def test_both_zero_vectors(self) -> None:
        score = CosineSimilarity.compute([0.0, 0.0], [0.0, 0.0])
        assert score == 0.0

    def test_empty_vectors_returns_zero(self) -> None:
        score = CosineSimilarity.compute([], [])
        assert score == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            CosineSimilarity.compute([1.0, 2.0], [1.0])

    def test_symmetry(self) -> None:
        a = [0.3, 0.7, 0.1]
        b = [0.5, 0.2, 0.8]
        assert CosineSimilarity.compute(a, b) == CosineSimilarity.compute(b, a)

    def test_score_in_range(self) -> None:
        a = [0.1, 0.5, 0.9]
        b = [0.4, 0.2, 0.6]
        score = CosineSimilarity.compute(a, b)
        assert -1.0 <= score <= 1.0

    def test_single_element_vectors(self) -> None:
        score = CosineSimilarity.compute([3.0], [4.0])
        assert abs(score - 1.0) < 1e-9

    def test_unit_vectors(self) -> None:
        # Two unit vectors at 60 degrees: cos(60) = 0.5
        import math

        score = CosineSimilarity.compute(
            [math.cos(0), math.sin(0)],
            [math.cos(math.pi / 3), math.sin(math.pi / 3)],
        )
        assert abs(score - 0.5) < 1e-9

    def test_large_values_normalised(self) -> None:
        score = CosineSimilarity.compute([1000.0, 0.0], [1000.0, 0.0])
        assert abs(score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests for ToolIndex
# ---------------------------------------------------------------------------


class TestToolIndex:
    def test_add_tool_stored(self, web_search_tool: ToolRecord) -> None:
        index = ToolIndex()
        index.add_tool(web_search_tool)
        assert index.get_tool("web-search-001") == web_search_tool

    def test_get_tool_missing_returns_none(self) -> None:
        index = ToolIndex()
        assert index.get_tool("nonexistent") is None

    def test_get_all_tools_empty(self) -> None:
        index = ToolIndex()
        assert index.get_all_tools() == []

    def test_get_all_tools_returns_all(self, populated_index: ToolIndex) -> None:
        tools = populated_index.get_all_tools()
        assert len(tools) == 4

    def test_add_tool_replaces_existing(self, web_search_tool: ToolRecord) -> None:
        index = ToolIndex()
        index.add_tool(web_search_tool)
        updated = web_search_tool.model_copy(update={"name": "Updated Search"})
        index.add_tool(updated)
        assert index.get_tool("web-search-001").name == "Updated Search"
        assert len(index.get_all_tools()) == 1

    def test_build_index_sets_embeddings(self, populated_index: ToolIndex) -> None:
        for tool in populated_index.get_all_tools():
            assert tool.embedding is not None
            assert len(tool.embedding) > 0

    def test_build_index_empty_does_not_raise(self) -> None:
        index = ToolIndex()
        index.build_index()  # Should not raise

    def test_search_returns_list(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="search the web", top_k=3)
        results = populated_index.search(query)
        assert isinstance(results, list)

    def test_search_respects_top_k(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="information retrieval", top_k=2)
        results = populated_index.search(query)
        assert len(results) <= 2

    def test_search_results_sorted_by_score(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="send email messages", top_k=10)
        results = populated_index.search(query)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_results_have_correct_rank(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="database queries", top_k=4)
        results = populated_index.search(query)
        for i, result in enumerate(results, start=1):
            assert result.rank == i

    def test_search_with_tag_filter(
        self, populated_index: ToolIndex, email_tool: ToolRecord
    ) -> None:
        query = SearchQuery(
            query_text="send message",
            tags_filter=["email"],
            top_k=10,
        )
        results = populated_index.search(query)
        # All results must have the 'email' tag
        for result in results:
            tags_lower = {t.lower() for t in result.tool.tags}
            assert "email" in tags_lower

    def test_search_tag_filter_case_insensitive(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(
            query_text="database",
            tags_filter=["SQL"],  # uppercase
            top_k=10,
        )
        results = populated_index.search(query)
        for result in results:
            tags_lower = {t.lower() for t in result.tool.tags}
            assert "sql" in tags_lower

    def test_search_tag_filter_no_matches_returns_empty(
        self, populated_index: ToolIndex
    ) -> None:
        query = SearchQuery(
            query_text="search",
            tags_filter=["nonexistent-tag"],
            top_k=10,
        )
        results = populated_index.search(query)
        assert results == []

    def test_search_before_build_returns_empty(self, web_search_tool: ToolRecord) -> None:
        index = ToolIndex()
        index.add_tool(web_search_tool)
        # No build_index called — embeddings are None
        query = SearchQuery(query_text="search web", top_k=5)
        results = index.search(query)
        # Tools without embeddings are skipped
        assert results == []

    def test_search_by_tags_returns_matching(self, populated_index: ToolIndex) -> None:
        results = populated_index.search_by_tags(["database", "sql"])
        assert len(results) >= 1
        for tool in results:
            tags_lower = {t.lower() for t in tool.tags}
            assert tags_lower & {"database", "sql"}

    def test_search_by_tags_case_insensitive(self, populated_index: ToolIndex) -> None:
        results_lower = populated_index.search_by_tags(["search"])
        results_upper = populated_index.search_by_tags(["SEARCH"])
        assert len(results_lower) == len(results_upper)

    def test_search_by_tags_no_match_returns_empty(self, populated_index: ToolIndex) -> None:
        results = populated_index.search_by_tags(["nothing"])
        assert results == []

    def test_search_empty_query_text_returns_empty(self, populated_index: ToolIndex) -> None:
        # Empty query text produces an all-zero vector -> similarity 0
        query = SearchQuery(query_text="", top_k=5)
        results = populated_index.search(query)
        # No error; may return empty (zero-vector embedding)
        assert isinstance(results, list)

    def test_search_returns_search_result_instances(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="web search", top_k=2)
        results = populated_index.search(query)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_tool_document_combines_fields(self) -> None:
        tool = ToolRecord(
            tool_id="t1",
            name="MyTool",
            description="Does amazing things",
            tags=["alpha", "beta"],
            capabilities=["run", "stop"],
        )
        index = ToolIndex()
        doc = index._tool_document(tool)
        assert "MyTool" in doc
        assert "amazing" in doc
        assert "alpha" in doc
        assert "run" in doc

    def test_search_top_k_default_ten(self, populated_index: ToolIndex) -> None:
        query = SearchQuery(query_text="tool")
        assert query.top_k == 10

    def test_multiple_builds_consistent(self, populated_index: ToolIndex) -> None:
        # Building index twice should yield same embedding dimensions
        dims_before = [len(t.embedding) for t in populated_index.get_all_tools() if t.embedding]
        populated_index.build_index()
        dims_after = [len(t.embedding) for t in populated_index.get_all_tools() if t.embedding]
        assert dims_before == dims_after

    @pytest.mark.parametrize(
        "top_k,expected_max",
        [(1, 1), (2, 2), (3, 3), (100, 4)],
    )
    def test_search_top_k_parametrized(
        self, populated_index: ToolIndex, top_k: int, expected_max: int
    ) -> None:
        query = SearchQuery(query_text="tool operations", top_k=top_k)
        results = populated_index.search(query)
        assert len(results) <= expected_max
