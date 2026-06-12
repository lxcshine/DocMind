# -*- coding: utf-8 -*-
"""
Reranker — Cross-Encoder reranking for Lost-in-the-Middle mitigation.

Industrial-grade approach:
  1. Retrieve candidate chunks via vector search (LightRAG)
  2. Rerank with Cross-Encoder (Cohere Rerank API) for precise relevance scoring
  3. Place highest-scoring chunks at context edges (sandwich structure)

This is the key difference between "heuristic reordering" and
"industrial-grade reordering": we use a learned model to score relevance
rather than guessing from position.

Architecture:
  ┌─────────────────────────────────────────────┐
  │  Retrieved chunks (from LightRAG / Agent)    │
  │    ↓                                         │
  │  Reranker.rerank(query, chunks)              │
  │    ↓                                         │
  │  Scored chunks: [(text, score), ...]         │
  │    ↓                                         │
  │  Sandwich reorder: high-score at edges,       │
  │  low-score in middle                         │
  └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


def _detect_device(device_hint: str = "") -> str:
    """
    Detect the best available device for model inference.

    Priority:
      1. Explicit device_hint (from RERANKER_DEVICE env var)
      2. Auto-detect: CUDA → MPS → CPU

    Returns:
      Device string: "cuda", "cuda:0", "mps", or "cpu"
    """
    if device_hint:
        logger.info(f"[Reranker:Device] Using configured device: {device_hint}")
        return device_hint

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(
                f"[Reranker:Device] Auto-detected: {device} ({gpu_name}, "
                f"{gpu_mem:.1f} GB VRAM)"
            )
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[Reranker:Device] Auto-detected: mps (Apple Silicon)")
            return "mps"
        else:
            logger.info("[Reranker:Device] Auto-detected: cpu (no GPU available)")
            return "cpu"
    except ImportError:
        logger.info("[Reranker:Device] PyTorch not installed, falling back to CPU")
        return "cpu"


@dataclass
class ScoredChunk:
    """A text chunk with a relevance score."""
    text: str
    score: float
    index: int  # original position

    def __lt__(self, other: "ScoredChunk") -> bool:
        return self.score < other.score


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 0,
    ) -> List[ScoredChunk]:
        """
        Rerank chunks by relevance to the query.

        Args:
            query: The user's query.
            chunks: List of text chunks to rerank.
            top_n: Keep only top-N results (0 = keep all).

        Returns:
            List of ScoredChunk sorted by relevance (highest first).
        """
        ...


class CohereReranker(BaseReranker):
    """
    Cohere Rerank API — industrial-grade Cross-Encoder reranking.

    Uses the Cohere Rerank API which runs a cross-encoder model
    (rerank-v3.5) to score each chunk's relevance to the query.
    This is significantly more accurate than heuristic or BM25-based
    reranking for Lost-in-the-Middle mitigation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
        top_n: int = 0,
    ):
        self._api_key = api_key
        self._model = model
        self._top_n = top_n
        self._client = None

    def _get_client(self):
        """Lazily initialize the Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.ClientV2(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "cohere package is required for Cohere Reranker. "
                    "Install it with: pip install cohere"
                )
        return self._client

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 0,
    ) -> List[ScoredChunk]:
        """Rerank chunks using Cohere Rerank API."""
        if not chunks:
            logger.debug("[Reranker:Cohere] No chunks to rerank, returning empty")
            return []

        n = top_n or self._top_n or len(chunks)
        logger.info(
            f"[Reranker:Cohere] Reranking {len(chunks)} chunks "
            f"(top_n={n}, model={self._model}), "
            f"query={query[:80]!r}..."
        )

        try:
            client = self._get_client()
            response = client.rerank(
                model=self._model,
                query=query,
                documents=chunks,
                top_n=n,
            )

            results = []
            for r in response.results:
                results.append(ScoredChunk(
                    text=chunks[r.index],
                    score=r.relevance_score,
                    index=r.index,
                ))

            # Log the reranking result: original_index → score
            score_summary = ", ".join(
                f"#{r.index}→{r.score:.4f}" for r in results
            )
            logger.info(
                f"[Reranker:Cohere] Reranking complete. "
                f"Scores: [{score_summary}]"
            )

            return results

        except Exception as e:
            logger.warning(f"[Reranker:Cohere] API call failed, falling back to heuristic: {e}")
            return HeuristicReranker().rerank(query, chunks, top_n)


class JinaRerankerV3(BaseReranker):
    """
    Jina Reranker v3 — listwise document reranker (local deployment).

    Unlike pairwise cross-encoders, Jina Reranker v3 processes the query
    and ALL candidate documents in a single forward pass (listwise),
    enabling cross-document attention for better ranking.

    Architecture: Qwen3-0.6B backbone + MLP projector (0.6B params)
    - Supports up to 64 documents per query
    - 131K token context window
    - 100+ languages, strong Chinese & multilingual performance
    - BEIR: 61.94 nDCG@10 (SOTA for 0.6B class)

    Model auto-downloads from HuggingFace on first use (~1.2GB).
    Requires: pip install transformers torch

    Usage:
      reranker = JinaRerankerV3()
      results = reranker.rerank("query", ["doc1", "doc2", ...])
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        device: str = "",
    ):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _get_model(self):
        """Lazily load the Jina Reranker v3 model."""
        if self._model is None:
            try:
                from transformers import AutoModel
                device = _detect_device(self._device)
                logger.info(
                    f"[Reranker:JinaV3] Loading model {self._model_name} "
                    f"(device={device})... "
                    f"First run will download ~1.2GB from HuggingFace."
                )
                kwargs = {
                    "trust_remote_code": True,
                }
                if device:
                    kwargs["device_map"] = device
                self._model = AutoModel.from_pretrained(
                    self._model_name,
                    **kwargs,
                )
                self._model.eval()
                logger.info(f"[Reranker:JinaV3] Model loaded successfully on {device}")
            except ImportError:
                raise ImportError(
                    "transformers and torch are required for JinaRerankerV3. "
                    "Install them with: pip install transformers torch"
                )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 0,
    ) -> List[ScoredChunk]:
        """Rerank chunks using Jina Reranker v3 (listwise)."""
        if not chunks:
            logger.debug("[Reranker:JinaV3] No chunks to rerank, returning empty")
            return []

        logger.info(
            f"[Reranker:JinaV3] Reranking {len(chunks)} chunks "
            f"(model={self._model_name}), "
            f"query={query[:80]!r}..."
        )

        try:
            model = self._get_model()

            # Jina v3 supports up to 64 documents per call
            batch_size = 64
            all_results = []

            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                results = model.rerank(query, batch)

                for r in results:
                    # Adjust index for batch offset
                    all_results.append(ScoredChunk(
                        text=r["document"],
                        score=r["relevance_score"],
                        index=r["index"] + start,
                    ))

            # Sort by score descending
            all_results.sort(key=lambda x: x.score, reverse=True)

            # Log results
            score_summary = ", ".join(
                f"#{r.index}→{r.score:.4f}" for r in all_results[:min(10, len(all_results))]
            )
            logger.info(
                f"[Reranker:JinaV3] Reranking complete. "
                f"Scores: [{score_summary}]"
            )

            n = top_n or len(chunks)
            if n < len(all_results):
                logger.info(f"[Reranker:JinaV3] Trimming to top_n={n}")
            return all_results[:n]

        except Exception as e:
            logger.warning(f"[Reranker:JinaV3] Failed, falling back to heuristic: {e}")
            return HeuristicReranker().rerank(query, chunks, top_n)


class LocalCrossEncoderReranker(BaseReranker):
    """
    Local Cross-Encoder reranker using sentence-transformers / FlagEmbedding.

    Runs a bge-reranker model locally for relevance scoring.
    No external API required — model is downloaded on first use (~560MB for base).

    Supported models:
      - BAAI/bge-reranker-base       (~560MB, good balance)
      - BAAI/bge-reranker-large      (~1.3GB, higher quality)
      - BAAI/bge-reranker-v2-m3      (~560MB, multilingual)

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "",
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._model = None

    def _get_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                device = _detect_device(self._device)
                logger.info(
                    f"[Reranker:Local] Loading model {self._model_name} "
                    f"(device={device}, max_length={self._max_length})..."
                )
                self._model = CrossEncoder(
                    self._model_name,
                    device=device if device else None,
                    max_length=self._max_length,
                )
                logger.info(f"[Reranker:Local] Model loaded successfully on {device}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for LocalCrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 0,
    ) -> List[ScoredChunk]:
        """Rerank chunks using local Cross-Encoder model."""
        if not chunks:
            logger.debug("[Reranker:Local] No chunks to rerank, returning empty")
            return []

        logger.info(
            f"[Reranker:Local] Reranking {len(chunks)} chunks "
            f"(model={self._model_name}), "
            f"query={query[:80]!r}..."
        )

        try:
            model = self._get_model()

            # Build query-document pairs
            pairs = [(query, chunk) for chunk in chunks]

            # Score all pairs
            scores = model.predict(pairs)

            # Build scored chunks
            import math
            scored = []
            for i, (score, chunk) in enumerate(zip(scores, chunks)):
                # Normalize score to [0, 1] range (Cross-Encoder outputs raw logits)
                # sigmoid normalization for consistency with Cohere's [0,1] range
                normalized = 1.0 / (1.0 + math.exp(-float(score)))
                scored.append(ScoredChunk(
                    text=chunk,
                    score=normalized,
                    index=i,
                ))

            # Sort by score descending
            scored.sort(key=lambda x: x.score, reverse=True)

            # Log results
            score_summary = ", ".join(
                f"#{s.index}→{s.score:.4f}(raw={float(scores[s.index]):.4f})"
                for s in scored[:min(10, len(scored))]
            )
            logger.info(
                f"[Reranker:Local] Reranking complete. "
                f"Scores: [{score_summary}]"
            )

            n = top_n or len(chunks)
            if n < len(scored):
                logger.info(f"[Reranker:Local] Trimming to top_n={n}")
            return scored[:n]

        except Exception as e:
            logger.warning(f"[Reranker:Local] Failed, falling back to heuristic: {e}")
            return HeuristicReranker().rerank(query, chunks, top_n)


class HeuristicReranker(BaseReranker):
    """
    Fallback heuristic reranker — no external API required.

    Uses a combination of:
      1. Keyword overlap (BM25-style TF scoring)
      2. Position bias (first/last chunks get slight boost)
      3. Length normalization

    This is a reasonable fallback when Cohere API is unavailable,
    but significantly less accurate than a cross-encoder.
    """

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: int = 0,
    ) -> List[ScoredChunk]:
        """Rerank chunks using heuristic scoring."""
        if not chunks:
            logger.debug("[Reranker:Heuristic] No chunks to rerank, returning empty")
            return []

        logger.info(
            f"[Reranker:Heuristic] Reranking {len(chunks)} chunks, "
            f"query={query[:80]!r}..."
        )

        query_terms = set(query.lower().split())
        scored = []

        for i, chunk in enumerate(chunks):
            score = self._compute_score(query_terms, chunk, i, len(chunks))
            scored.append(ScoredChunk(text=chunk, score=score, index=i))

        scored.sort(key=lambda x: x.score, reverse=True)

        # Log top scores and position changes
        score_summary = ", ".join(
            f"#{s.index}→{s.score:.4f}" for s in scored[:min(10, len(scored))]
        )
        logger.info(
            f"[Reranker:Heuristic] Scoring complete. "
            f"Top scores: [{score_summary}]"
        )

        # Log position changes (original → new)
        position_changes = ", ".join(
            f"orig#{s.index}→rank{i}" for i, s in enumerate(scored)
        )
        logger.debug(f"[Reranker:Heuristic] Position mapping: [{position_changes}]")

        n = top_n or len(chunks)
        if n < len(scored):
            logger.info(f"[Reranker:Heuristic] Trimming to top_n={n} (from {len(scored)})")
        return scored[:n]

    def _compute_score(
        self,
        query_terms: set,
        chunk: str,
        position: int,
        total: int,
    ) -> float:
        """Compute heuristic relevance score."""
        chunk_lower = chunk.lower()
        chunk_terms = set(chunk_lower.split())

        # 1. Keyword overlap (Jaccard-like)
        overlap = len(query_terms & chunk_terms)
        union = len(query_terms | chunk_terms)
        jaccard = overlap / max(union, 1)

        # 2. Term frequency (BM25-style)
        tf = sum(chunk_lower.count(t) for t in query_terms)
        # Normalize by chunk length
        tf_norm = tf / max(len(chunk_terms), 1)

        # 3. Position bias: slight boost for first/last chunks
        # (they tend to contain overview/conclusion)
        if total <= 2:
            pos_bias = 0.0
        elif position == 0 or position == total - 1:
            pos_bias = 0.1
        else:
            pos_bias = 0.0

        # Combined score
        return jaccard * 0.5 + tf_norm * 0.4 + pos_bias


# ---- Sandwich Reordering ----

def sandwich_reorder(scored_chunks: List[ScoredChunk]) -> List[ScoredChunk]:
    """
    Reorder scored chunks into sandwich structure for Lost-in-the-Middle mitigation.

    Strategy:
      - Highest-scoring chunks → beginning (high attention)
      - Lowest-scoring chunks → middle (lower attention)
      - Second-highest chunks → end (high attention, just before current message)

    This ensures the LLM sees the most relevant information at the
    positions where it pays the most attention.

    Example with 6 chunks scored [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
      Beginning: [0.9, 0.8]  (top 2)
      Middle:    [0.5, 0.4]  (bottom 2)
      End:       [0.7, 0.6]  (middle 2)
    """
    if len(scored_chunks) <= 2:
        logger.debug(
            f"[Sandwich] Only {len(scored_chunks)} chunks, no reorder needed"
        )
        return scored_chunks

    # Split into thirds — ensure NO chunks are dropped
    n = len(scored_chunks)
    third = max(1, n // 3)
    remainder = n - 3 * third  # chunks left over after equal thirds

    # Distribute remainder to avoid dropping chunks:
    # beginning gets extra (most important, high attention position)
    beg_end = third + (1 if remainder >= 1 else 0)
    mid_start = 2 * third + (1 if remainder >= 2 else 0)

    # Top → beginning (high attention)
    beginning = scored_chunks[:beg_end]

    # Bottom → middle (lower attention)
    middle = scored_chunks[mid_start:]

    # Middle-scoring → end (high attention, just before user message)
    end = scored_chunks[beg_end:mid_start]

    # Sandwich: beginning + middle + end
    result = beginning + middle + end

    # Safety check: ensure no chunks were dropped
    if len(result) != n:
        logger.warning(
            f"[Sandwich] Chunk count mismatch: input={n}, output={len(result)}. "
            f"Falling back to original order."
        )
        return scored_chunks

    # Log the sandwich structure
    beg_scores = [f"#{c.index}({c.score:.4f})" for c in beginning]
    mid_scores = [f"#{c.index}({c.score:.4f})" for c in middle]
    end_scores = [f"#{c.index}({c.score:.4f})" for c in end]

    logger.info(
        f"[Sandwich] Reordered {n} chunks into sandwich "
        f"(1/3={third}): "
        f"BEGIN[{', '.join(beg_scores)}] "
        f"MID[{', '.join(mid_scores)}] "
        f"END[{', '.join(end_scores)}]"
    )

    # Log final position mapping: original_index → new_position
    final_order = [f"orig#{c.index}→pos{i}" for i, c in enumerate(result)]
    logger.debug(f"[Sandwich] Final order: [{', '.join(final_order)}]")

    return result


def sandwich_reorder_texts(
    query: str,
    chunks: List[str],
    reranker: Optional[BaseReranker] = None,
) -> List[str]:
    """
    Convenience function: rerank + sandwich reorder, return text strings.

    This is the main entry point for ContextManager and RAG pipeline.
    """
    if not chunks:
        return []

    reranker_name = type(reranker).__name__ if reranker else "HeuristicReranker"
    logger.info(
        f"[Reranker] sandwich_reorder_texts called: "
        f"{len(chunks)} chunks, reranker={reranker_name}, "
        f"query={query[:80]!r}..."
    )

    # Rerank
    if reranker is not None:
        scored = reranker.rerank(query, chunks)
    else:
        scored = HeuristicReranker().rerank(query, chunks)

    # Sandwich reorder
    reordered = sandwich_reorder(scored)

    result = [chunk.text for chunk in reordered]
    logger.info(
        f"[Reranker] sandwich_reorder_texts complete: "
        f"input_order={[f'chunk{i}' for i in range(len(chunks))]} → "
        f"output_order=[{', '.join(f'orig#{c.index}' for c in reordered)}]"
    )

    return result


# ---- Singleton ----

_reranker: Optional[BaseReranker] = None


def get_reranker() -> Optional[BaseReranker]:
    """Get or create the global reranker instance.

    Priority chain:
      1. Cohere Rerank API (if COHERE_API_KEY is set)
      2. Jina Reranker v3 (default local model, listwise, auto-download)
      3. Local Cross-Encoder (if RERANKER_MODEL is set to a non-Jina model)
      4. Heuristic (keyword-based, always available as fallback)
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    # 1. Try Cohere Rerank API
    if settings.COHERE_API_KEY:
        logger.info("[Reranker:Init] Attempting Cohere Rerank API...")
        try:
            reranker = CohereReranker(
                api_key=settings.COHERE_API_KEY,
                model=settings.COHERE_RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
            )
            # Validate API key by initializing the client
            reranker._get_client()
            _reranker = reranker
            logger.info(
                f"[Reranker:Init] Cohere Rerank API initialized successfully "
                f"(model={settings.COHERE_RERANK_MODEL})"
            )
            return _reranker
        except Exception as e:
            logger.warning(
                f"[Reranker:Init] Cohere Rerank API FAILED: {type(e).__name__}: {e}. "
                f"Falling back to next option."
            )
    else:
        logger.info("[Reranker:Init] COHERE_API_KEY not set, skipping Cohere")

    # 2. Try local reranker model
    model_name = getattr(settings, "RERANKER_MODEL", "")
    if model_name:
        # Jina Reranker v3 (listwise architecture)
        if "jina-reranker-v3" in model_name or model_name == "jinaai/jina-reranker-v3":
            logger.info(f"[Reranker:Init] Attempting Jina Reranker v3 (model={model_name})...")
            try:
                reranker = JinaRerankerV3(
                    model_name=model_name,
                    device=getattr(settings, "RERANKER_DEVICE", ""),
                )
                # Validate by loading the model (will download on first use)
                reranker._get_model()
                _reranker = reranker
                logger.info(
                    f"[Reranker:Init] Jina Reranker v3 initialized successfully "
                    f"(model={model_name})"
                )
                return _reranker
            except Exception as e:
                logger.warning(
                    f"[Reranker:Init] Jina Reranker v3 FAILED: {type(e).__name__}: {e}. "
                    f"Falling back to next option."
                )
        else:
            # Other cross-encoder models (BGE, etc.)
            logger.info(f"[Reranker:Init] Attempting local Cross-Encoder (model={model_name})...")
            try:
                reranker = LocalCrossEncoderReranker(
                    model_name=model_name,
                    device=getattr(settings, "RERANKER_DEVICE", ""),
                    max_length=getattr(settings, "RERANKER_MAX_LENGTH", 512),
                )
                # Validate by loading the model
                reranker._get_model()
                _reranker = reranker
                logger.info(
                    f"[Reranker:Init] Local Cross-Encoder initialized successfully "
                    f"(model={model_name})"
                )
                return _reranker
            except Exception as e:
                logger.warning(
                    f"[Reranker:Init] Local Cross-Encoder FAILED: {type(e).__name__}: {e}. "
                    f"Falling back to heuristic."
                )
    else:
        logger.info("[Reranker:Init] RERANKER_MODEL not set, skipping local model")

    # 3. Fallback to heuristic
    _reranker = HeuristicReranker()
    logger.info("[Reranker:Init] Using heuristic reranker (no API key, no local model)")
    return _reranker
