"""
End-to-end workshop flow tests.

Validates the entire 4-phase workshop progression:
- Phase 1: NaiveContextEngine (baseline, returns all chunks)
- Phase 2: Prompt engineering (system prompt template)
- Phase 3: SimilarityContextEngine (cosine similarity + top-k)
- Phase 4: RAGContextEngine (Qdrant ANN + BM25 re-ranking)

Also verifies:
- All exercise stubs raise NotImplementedError
- All solution implementations produce correct output
- Toggle switching correctly swaps between exercises and solutions
"""

import importlib
import os
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
import pytest

from workshop import LLMConfig
from workshop.chat import WhatsappMessage, load_whatsapp_chat
from workshop.rag.chunkers import MessageCountChunker, MessageCountParams
from workshop.rag.engines.types import ChunkObject

EXAMPLE_CHAT_PATH = Path(__file__).parent.parent / "chats" / "example_chat.txt"

HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_GOOGLE_KEY = bool(os.environ.get("GOOGLE_API_KEY"))
HAS_ANY_LLM_KEY = HAS_OPENAI_KEY or HAS_ANTHROPIC_KEY or HAS_GOOGLE_KEY

requires_llm_key = pytest.mark.skipif(
    not HAS_ANY_LLM_KEY,
    reason="No LLM API key found (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)",
)


def set_toggles(
    *,
    prompting: bool = True,
    similarity: bool = True,
    reranking: bool = True,
) -> None:
    """
    Set exercise toggles and reload all consuming modules.

    Toggles are read at module import time, so changing them requires
    reloading the modules that consume them. This function:
    1. Sets toggle values directly on the exercise_toggles module
    2. Reloads all consuming modules in dependency order

    Args:
        prompting: Whether to use the prompting solution
        similarity: Whether to use the similarity solution
        reranking: Whether to use the reranking solution
    """
    import workshop.exercise_toggles as toggles

    toggles.USE_PROMPTING_SOLUTION = prompting
    toggles.USE_SIMILARITY_SOLUTION = similarity
    toggles.USE_RERANKING_SOLUTION = reranking

    import workshop.llm

    importlib.reload(workshop.llm)

    import workshop.rag.engines.similarity

    importlib.reload(workshop.rag.engines.similarity)

    import workshop.rag.engines.qdrant

    importlib.reload(workshop.rag.engines.qdrant)

    import workshop.rag.engines

    importlib.reload(workshop.rag.engines)


@pytest.fixture(autouse=True)
def restore_toggles():
    """
    Save and restore exercise toggle state after each test.

    This is critical because toggle switching modifies module-level state
    that persists across tests.
    """
    import workshop.exercise_toggles as toggles

    original_prompting = toggles.USE_PROMPTING_SOLUTION
    original_similarity = toggles.USE_SIMILARITY_SOLUTION
    original_reranking = toggles.USE_RERANKING_SOLUTION

    yield

    set_toggles(
        prompting=original_prompting,
        similarity=original_similarity,
        reranking=original_reranking,
    )


@pytest.fixture
def example_messages() -> Sequence[WhatsappMessage]:
    """Load messages from the example chat file."""
    return load_whatsapp_chat(EXAMPLE_CHAT_PATH)


@pytest.fixture
def example_chunks(example_messages: Sequence[WhatsappMessage]) -> List[ChunkObject]:
    """Create chunks from the example chat with default params."""
    chunker: Any = MessageCountChunker(
        params=MessageCountParams(chunk_length=5, chunk_overlap=2),
    )
    return chunker.chunk_messages(example_messages)


@pytest.fixture
def mock_embedder() -> Embedder:
    """Mock embedder that returns deterministic embeddings without API calls."""
    return Embedder(TestEmbeddingModel())


def _make_test_chunks(n: int = 5) -> List[ChunkObject]:
    """
    Create simple test chunks for unit-level tests.

    Args:
        n: Number of chunks to create

    Returns:
        List of ChunkObject with distinct text content
    """
    texts = [
        "Alice said let's plan a trip to Paris next month",
        "Bob suggested we should book flights early for better prices",
        "Charlie mentioned the hotel near the Eiffel Tower was affordable",
        "Alice asked about restaurant recommendations in Paris",
        "Bob shared a link to a travel guide for France",
        "Charlie proposed visiting museums on the first day",
        "Alice wanted to discuss the budget for the trip",
        "Bob calculated the total cost including flights and hotels",
    ]
    chunks = []
    for i in range(min(n, len(texts))):
        chunks.append(
            ChunkObject(
                id=f"chunk_{i}",
                text=texts[i],
                message_ids=[i],
                metadata={
                    "start_idx": i,
                    "end_idx": i + 1,
                    "start_time": f"2024-01-01T10:0{i}:00",
                    "end_time": f"2024-01-01T10:0{i}:30",
                    "speakers": ["Alice", "Bob", "Charlie"][i % 3 : i % 3 + 1],
                },
            )
        )
    return chunks


class TestExercisesRaiseNotImplemented:
    """Verify all exercise stubs raise NotImplementedError."""

    def test_prompting_exercise_raises(self) -> None:
        """Exercise get_system_prompt() raises NotImplementedError."""
        from workshop.rag.exercises.prompting import get_system_prompt

        with pytest.raises(NotImplementedError):
            get_system_prompt()

    def test_similarity_cosine_raises(self) -> None:
        """Exercise cosine_similarity() raises NotImplementedError."""
        from workshop.rag.exercises.similarity import cosine_similarity

        query = np.array([1.0, 0.0, 0.0])
        chunks = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(NotImplementedError):
            cosine_similarity(query, chunks)

    def test_similarity_topk_raises(self) -> None:
        """Exercise get_top_k() raises NotImplementedError."""
        from workshop.rag.exercises.similarity import get_top_k

        sims = np.array([0.9, 0.5, 0.1])

        with pytest.raises(NotImplementedError):
            get_top_k(sims, threshold=0.0, k=2)

    def test_reranking_exercise_raises(self) -> None:
        """Exercise rerank() raises NotImplementedError."""
        from workshop.rag.exercises.reranking import rerank

        chunks = _make_test_chunks(3)

        with pytest.raises(NotImplementedError):
            rerank("Paris trip", chunks, top_k=2)


class TestSolutionsWork:
    """Verify all solution implementations produce correct output."""

    def test_prompting_solution_returns_template(self) -> None:
        """Solution get_system_prompt() returns a string with {context} placeholder."""
        from workshop.rag.solutions.prompting import get_system_prompt

        prompt = get_system_prompt()

        assert isinstance(prompt, str)
        assert "{context}" in prompt
        assert len(prompt) > 50

    def test_prompting_solution_template_is_formattable(self) -> None:
        """Solution prompt template can be formatted with context."""
        from workshop.rag.solutions.prompting import get_system_prompt

        prompt = get_system_prompt()
        formatted = prompt.format(context="Test context here")

        assert "Test context here" in formatted
        assert "{context}" not in formatted

    def test_similarity_solution_cosine(self) -> None:
        """Solution cosine_similarity() returns correct values."""
        from workshop.rag.solutions.similarity import cosine_similarity

        query = np.array([1.0, 0.0, 0.0])
        chunks = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )

        sims = cosine_similarity(query, chunks)

        assert sims.shape == (3,)
        assert pytest.approx(sims[0], abs=1e-6) == 1.0
        assert pytest.approx(sims[1], abs=1e-6) == 0.0
        assert pytest.approx(sims[2], abs=1e-6) == -1.0

    def test_similarity_solution_topk(self) -> None:
        """Solution get_top_k() returns correct indices in descending order."""
        from workshop.rag.solutions.similarity import get_top_k

        sims = np.array([0.9, 0.3, 0.7, 0.1, 0.8])
        indices = get_top_k(sims, threshold=0.5, k=2)

        assert len(indices) == 2
        assert indices[0] == 0
        assert indices[1] == 4

    def test_similarity_solution_topk_threshold_filter(self) -> None:
        """Solution get_top_k() respects threshold filtering."""
        from workshop.rag.solutions.similarity import get_top_k

        sims = np.array([0.9, 0.3, 0.7, 0.1, 0.8])
        indices = get_top_k(sims, threshold=0.5, k=10)

        assert len(indices) == 3
        assert all(sims[i] >= 0.5 for i in indices)

    def test_reranking_solution_reranks(self) -> None:
        """Solution rerank() reorders chunks by BM25 relevance."""
        from workshop.rag.solutions.reranking import rerank

        chunks = _make_test_chunks(5)
        result = rerank("Paris restaurant recommendations", chunks, top_k=3)

        assert len(result) == 3
        assert all(isinstance(c, ChunkObject) for c in result)

    def test_reranking_solution_prefers_relevant(self) -> None:
        """Solution rerank() scores chunks with query terms higher."""
        from workshop.rag.solutions.reranking import rerank

        chunks = _make_test_chunks(5)
        result = rerank("Paris", chunks, top_k=5)

        top_texts = [c.text for c in result[:2]]
        assert any("Paris" in t for t in top_texts)

    def test_reranking_solution_empty_chunks(self) -> None:
        """Solution rerank() handles empty input."""
        from workshop.rag.solutions.reranking import rerank

        result = rerank("test query", [], top_k=5)
        assert result == []


class TestToggleSwitching:
    """Verify that set_toggles() correctly swaps between exercises and solutions."""

    def test_toggle_prompting_solution_on(self) -> None:
        """With prompting=True, llm module uses solution prompt."""
        set_toggles(prompting=True)

        from workshop.llm import get_system_prompt

        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert "{context}" in prompt

    def test_toggle_prompting_exercise_on(self) -> None:
        """With prompting=False, llm module uses exercise prompt (raises)."""
        set_toggles(prompting=False)

        from workshop.llm import get_system_prompt

        with pytest.raises(NotImplementedError):
            get_system_prompt()

    def test_toggle_similarity_solution(self, example_chunks: List[ChunkObject], mock_embedder: Embedder) -> None:
        """With similarity=True, SimilarityContextEngine retrieval works."""
        set_toggles(similarity=True)

        from workshop.rag.engines.similarity import SimilarityContextEngine

        engine = SimilarityContextEngine(
            embedder=mock_embedder,
            similarity_threshold=0.0,
            top_k=3,
        )
        engine.add_context(example_chunks)
        results = engine.get_relevant_context("chunking strategies")

        assert len(results) > 0
        assert len(results) <= 3

    def test_toggle_similarity_exercise(self, example_chunks: List[ChunkObject], mock_embedder: Embedder) -> None:
        """With similarity=False, SimilarityContextEngine raises NotImplementedError."""
        set_toggles(similarity=False)

        from workshop.rag.engines.similarity import SimilarityContextEngine

        engine = SimilarityContextEngine(
            embedder=mock_embedder,
            similarity_threshold=0.0,
            top_k=3,
        )
        engine.add_context(example_chunks)

        with pytest.raises(NotImplementedError):
            engine.get_relevant_context("chunking strategies")

    def test_toggle_reranking_solution(
        self, example_chunks: List[ChunkObject], mock_embedder: Embedder, tmp_path: Path
    ) -> None:
        """With reranking=True, RAGContextEngine uses BM25 solution."""
        set_toggles(reranking=True)

        from workshop.rag.engines.qdrant import RAGContextEngine

        db_path = str(tmp_path / "qdrant_solution")
        engine = RAGContextEngine(
            embedder=mock_embedder,
            db_path=db_path,
            top_k=3,
            rerank_candidates=10,
        )
        try:
            engine.add_context(example_chunks)
            results = engine.get_relevant_context("chunking strategies")

            assert len(results) > 0
            assert len(results) <= 3
        finally:
            engine.delete_storage()

    def test_toggle_reranking_exercise(
        self, example_chunks: List[ChunkObject], mock_embedder: Embedder, tmp_path: Path
    ) -> None:
        """With reranking=False, RAGContextEngine raises NotImplementedError."""
        set_toggles(reranking=False)

        from workshop.rag.engines.qdrant import RAGContextEngine

        db_path = str(tmp_path / "qdrant_exercise")
        engine = RAGContextEngine(
            embedder=mock_embedder,
            db_path=db_path,
            top_k=3,
            rerank_candidates=10,
        )
        try:
            engine.add_context(example_chunks)

            with pytest.raises(NotImplementedError):
                engine.get_relevant_context("chunking strategies")
        finally:
            engine.delete_storage()


class TestPhaseProgression:
    """End-to-end walk through all 4 workshop phases."""

    def test_full_workshop_flow(
        self,
        example_messages: Sequence[WhatsappMessage],
        example_chunks: List[ChunkObject],
        mock_embedder: Embedder,
        tmp_path: Path,
    ) -> None:
        """
        Walk through P1 -> P2 -> P3 -> P4 sequentially.

        Verifies each phase produces expected behavior with all solutions enabled.
        """
        set_toggles(prompting=True, similarity=True, reranking=True)

        # -- Phase 1: NaiveContextEngine returns ALL chunks --
        from workshop.rag.engines.naive import NaiveContextEngine

        naive = NaiveContextEngine()
        naive.add_context(example_chunks)

        naive_results = naive.get_relevant_context("any query", top_k=3)
        assert len(naive_results) == len(
            example_chunks
        ), "NaiveContextEngine should return ALL chunks regardless of query"
        naive.clear()
        assert naive.context_count == 0

        # -- Phase 2: Prompt template is valid --
        from workshop.llm import get_system_prompt

        prompt = get_system_prompt()
        assert "{context}" in prompt
        formatted = prompt.format(context="Some retrieved context")
        assert "Some retrieved context" in formatted

        # -- Phase 3: SimilarityContextEngine returns relevant subset --
        from workshop.rag.engines.similarity import SimilarityContextEngine

        sim_engine = SimilarityContextEngine(
            embedder=mock_embedder,
            similarity_threshold=0.0,
            top_k=3,
        )
        sim_engine.add_context(example_chunks)

        sim_results = sim_engine.get_relevant_context("chunking strategies")
        assert 0 < len(sim_results) <= 3, "SimilarityContextEngine should return a filtered subset"
        sim_engine.clear()
        assert sim_engine.context_count == 0

        # -- Phase 4: RAGContextEngine with Qdrant + BM25 re-ranking --
        from workshop.rag.engines.qdrant import RAGContextEngine

        db_path = str(tmp_path / "qdrant_flow")
        rag_engine = RAGContextEngine(
            embedder=mock_embedder,
            db_path=db_path,
            top_k=3,
            rerank_candidates=10,
        )
        try:
            rag_engine.add_context(example_chunks)
            assert rag_engine.context_count == len(example_chunks)

            rag_results = rag_engine.get_relevant_context("chunking strategies")
            assert 0 < len(rag_results) <= 3, "RAGContextEngine should return re-ranked subset"

            for chunk in rag_results:
                assert chunk.text
                assert chunk.message_ids
                assert "start_time" in chunk.metadata
                assert "end_time" in chunk.metadata

            rag_engine.clear()
            assert rag_engine.context_count == 0
        finally:
            rag_engine.delete_storage()


@pytest.mark.integration
class TestIntegrationWithLLM:
    """
    Real LLM integration tests.

    These tests require API keys and make real API calls.
    Run with: uv run pytest tests/test_workshop_flow.py -m integration
    """

    @staticmethod
    def _get_llm_config() -> LLMConfig:
        """Pick an available LLM based on environment variables."""
        if HAS_OPENAI_KEY:
            return {
                "model_name": "openai:gpt-4o-mini",
                "kwargs": {"temperature": 0.3},
                "structured_output_type": None,
            }
        if HAS_ANTHROPIC_KEY:
            return {
                "model_name": "anthropic:claude-3-haiku-20240307",
                "kwargs": {"temperature": 0.3},
                "structured_output_type": None,
            }
        if HAS_GOOGLE_KEY:
            return {
                "model_name": "google-gla:gemini-2.0-flash",
                "kwargs": {"temperature": 0.3},
                "structured_output_type": None,
            }
        pytest.skip("No LLM API key available")

    @requires_llm_key
    def test_real_llm_with_naive_engine(self, example_chunks: List[ChunkObject]) -> None:
        """
        Full end-to-end: load chat, chunk, query NaiveContextEngine, call real LLM.

        Verifies the LLM produces a structured RAGResponse with all expected fields.
        """
        set_toggles(prompting=True)

        from workshop.llm import get_pydantic_agent
        from workshop.rag.engines.naive import NaiveContextEngine
        from workshop.structured_types import RAGResponse

        config: LLMConfig = self._get_llm_config()
        agent = get_pydantic_agent(config, structured_output_type=RAGResponse)

        engine = NaiveContextEngine()
        engine.add_context(example_chunks[:3])

        context_text = "\n\n---\n\n".join(chunk.text for chunk in engine.get_relevant_context("RAG workshop"))

        result = agent.run_sync(
            "What topics were discussed?",
            deps={"context": context_text},
        )

        assert isinstance(result.output, RAGResponse)
        assert result.output.output
        assert result.output.confidence in ("high", "medium", "low")
        assert len(result.output.reasoning_steps) >= 1
