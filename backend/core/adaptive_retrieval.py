# -*- coding: utf-8 -*-
"""
Adaptive Context Retrieval (RF-Mem inspired)

Implements familiarity-aware adaptive retrieval with recollection path:
  1. Probe retrieval + familiarity signal assessment
  2. Adaptive switching between familiarity/recollection paths
  3. Multi-turn chain recollection (clustering + alpha-mix + iterative expansion)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FamiliaritySignal:
    """Familiarity assessment result."""
    mean_score: float
    entropy: float
    familiarity: float
    is_familiar: bool


class AdaptiveRetrieval:
    """
    RF-Mem inspired adaptive retrieval system.

    Uses probe retrieval to assess familiarity, then adaptively switches
    between fast familiarity path and deep recollection path.
    """

    def __init__(
        self,
        familiarity_threshold: float = 0.65,
        entropy_threshold: float = 1.5,
        probe_top_k: int = 5,
        max_recollection_iters: int = 3,
        alpha_mix: float = 0.6,
        n_clusters: int = 3,
    ):
        self.familiarity_threshold = familiarity_threshold
        self.entropy_threshold = entropy_threshold
        self.probe_top_k = probe_top_k
        self.max_recollection_iters = max_recollection_iters
        self.alpha_mix = alpha_mix
        self.n_clusters = n_clusters
        self._embedding_func = None
        self._rag_instance = None

    def set_rag_instance(self, rag):
        """Set the RAGAnything instance for retrieval."""
        self._rag_instance = rag
        if hasattr(rag, 'lightrag') and hasattr(rag.lightrag, 'embedding_func'):
            self._embedding_func = rag.lightrag.embedding_func

    async def compute_familiarity(self, query: str) -> FamiliaritySignal:
        """
        Step 1: Probe retrieval + familiarity assessment.

        Performs a low-cost probe retrieval and computes:
        - Mean similarity score
        - Information entropy (uncertainty measure)
        - Combined familiarity score
        """
        if self._rag_instance is None:
            logger.warning("RAG instance not set, returning default unfamiliar signal")
            return FamiliaritySignal(
                mean_score=0.0,
                entropy=2.0,
                familiarity=0.0,
                is_familiar=False,
            )

        try:
            # Probe retrieval
            probe_results = await self._rag_instance.lightrag.chunk_retrieval(
                query, top_k=self.probe_top_k
            )

            if not probe_results or len(probe_results) == 0:
                return FamiliaritySignal(
                    mean_score=0.0,
                    entropy=2.0,
                    familiarity=0.0,
                    is_familiar=False,
                )

            # Extract scores
            scores = []
            for result in probe_results:
                if hasattr(result, 'score'):
                    scores.append(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    scores.append(result['score'])
                else:
                    scores.append(0.5)  # default

            if not scores:
                return FamiliaritySignal(
                    mean_score=0.0,
                    entropy=2.0,
                    familiarity=0.0,
                    is_familiar=False,
                )

            # Normalize scores to probabilities
            scores_np = np.array(scores)
            scores_np = scores_np - scores_np.min()
            score_sum = scores_np.sum()
            if score_sum == 0:
                probs = np.ones(len(scores_np)) / len(scores_np)
            else:
                probs = scores_np / score_sum

            # Compute metrics
            mean_score = float(np.mean(scores))
            entropy = float(-np.sum([p * np.log(p + 1e-10) for p in probs]))

            # Combined familiarity: higher mean + lower entropy = more familiar
            max_entropy = np.log(len(scores))
            normalized_entropy = entropy / (max_entropy + 1e-10)
            familiarity = mean_score * (1 - normalized_entropy * 0.5)

            is_familiar = (
                mean_score >= self.familiarity_threshold
                and entropy <= self.entropy_threshold
            )

            logger.info(
                f"[Familiarity] mean={mean_score:.3f}, entropy={entropy:.3f}, "
                f"familiarity={familiarity:.3f}, is_familiar={is_familiar}"
            )

            return FamiliaritySignal(
                mean_score=mean_score,
                entropy=entropy,
                familiarity=familiarity,
                is_familiar=is_familiar,
            )

        except Exception as e:
            logger.warning(f"[Familiarity] Probe retrieval failed: {e}")
            return FamiliaritySignal(
                mean_score=0.0,
                entropy=2.0,
                familiarity=0.0,
                is_familiar=False,
            )

    async def adaptive_query(
        self,
        query: str,
        history: List[Dict],
        mode: str = "mix",
    ) -> str:
        """
        Step 2: Adaptive dual-path switching.

        Based on familiarity signal, switches between:
        - High familiarity: Fast direct retrieval
        - Low familiarity: Deep recollection retrieval
        """
        signal = await self.compute_familiarity(query)

        if signal.is_familiar:
            logger.info("[Adaptive] High familiarity → using fast path")
            return await self._familiarity_path(query, mode)
        else:
            logger.info("[Adaptive] Low familiarity → using recollection path")
            return await self._recollection_path(query, history, mode)

    async def _familiarity_path(self, query: str, mode: str) -> str:
        """Fast path: direct retrieval with minimal context."""
        try:
            result = await self._rag_instance.aquery(query, mode=mode)
            return result
        except Exception as e:
            logger.error(f"[Familiarity Path] Query failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def _recollection_path(
        self,
        query: str,
        history: List[Dict],
        mode: str,
    ) -> str:
        """
        Step 3: Multi-turn chain recollection retrieval.

        Simulates human "following the thread" thinking process:
        1. Cluster candidate memories
        2. Alpha-mix query with cluster centroids
        3. Iterative expansion
        """
        if not history or len(history) < 2:
            # Not enough history, fallback to direct query with context
            logger.info("[Recollection] Not enough history, using fallback")
            return await self._fallback_with_history(query, history, mode)

        try:
            # Get history embeddings
            history_embeddings, history_texts = await self._encode_history(history)

            if history_embeddings is None or len(history_embeddings) < 2:
                return await self._fallback_with_history(query, history, mode)

            # Iterative recollection
            expanded_contexts = []
            current_query_embed = await self._encode_text(query)

            if current_query_embed is None:
                return await self._fallback_with_history(query, history, mode)

            for iteration in range(self.max_recollection_iters):
                logger.info(f"[Recollection] Iteration {iteration + 1}/{self.max_recollection_iters}")

                # Cluster
                n_clusters = min(self.n_clusters, len(history_embeddings))
                if n_clusters < 2:
                    break

                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(history_embeddings)
                centroids = kmeans.cluster_centers_

                # Alpha-mix: blend query with centroids
                recollected_queries = []
                for centroid in centroids:
                    mixed = self.alpha_mix * current_query_embed + (1 - self.alpha_mix) * centroid
                    recollected_queries.append(mixed)

                # Retrieve with mixed queries
                iteration_results = []
                for mixed_query in recollected_queries:
                    # Find nearest history texts to use as context
                    similarities = mixed_query @ history_embeddings.T
                    top_indices = np.argsort(similarities)[-3:][::-1]
                    for idx in top_indices:
                        iteration_results.append(history_texts[idx])

                expanded_contexts.extend(iteration_results)

                # Update current query embedding for next iteration
                if iteration_results:
                    result_embeds = await self._encode_texts(iteration_results[:3])
                    if result_embeds:
                        current_query_embed = np.mean(result_embeds, axis=0)

            # Build enhanced query with expanded context
            unique_contexts = list(dict.fromkeys(expanded_contexts))[:10]
            enhanced_query = self._build_enhanced_query(query, unique_contexts)

            logger.info(f"[Recollection] Enhanced query with {len(unique_contexts)} context items")
            result = await self._rag_instance.aquery(enhanced_query, mode=mode)
            return result

        except Exception as e:
            logger.error(f"[Recollection] Failed: {e}")
            return await self._fallback_with_history(query, history, mode)

    async def _encode_history(self, history: List[Dict]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Encode conversation history into embeddings."""
        texts = []
        for msg in history[-20:]:  # Keep last 20 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                texts.append(f"{role}: {content}")

        if not texts:
            return None, []

        embeddings = await self._encode_texts(texts)
        if embeddings is None:
            return None, []

        return np.array(embeddings), texts

    async def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode a single text into embedding."""
        if self._embedding_func is None:
            return None
        try:
            result = await self._embedding_func([text])
            if result and len(result) > 0:
                return np.array(result[0])
        except Exception as e:
            logger.warning(f"[Embedding] Failed to encode text: {e}")
        return None

    async def _encode_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Encode multiple texts into embeddings."""
        if self._embedding_func is None:
            return None
        try:
            result = await self._embedding_func(texts)
            if result:
                return result
        except Exception as e:
            logger.warning(f"[Embedding] Failed to encode texts: {e}")
        return None

    def _build_enhanced_query(self, query: str, contexts: List[str]) -> str:
        """Build enhanced query with retrieved context."""
        if not contexts:
            return query

        context_text = "\n".join([f"- {ctx}" for ctx in contexts[:5]])
        enhanced = (
            f"Based on our conversation context:\n{context_text}\n\n"
            f"Please answer the following question considering the above context:\n{query}"
        )
        return enhanced

    async def _fallback_with_history(
        self,
        query: str,
        history: List[Dict],
        mode: str,
    ) -> str:
        """Fallback: simple history concatenation."""
        if not history:
            return await self._rag_instance.aquery(query, mode=mode)

        history_context = "\n".join(
            [f"{m['role']}: {m['content']}" for m in history[-6:]]
        )
        enhanced = (
            f"Conversation history:\n{history_context}\n\n"
            f"Current question: {query}"
        )
        return await self._rag_instance.aquery(enhanced, mode=mode)


# Global instance
_adaptive_retrieval = AdaptiveRetrieval()


def get_adaptive_retrieval() -> AdaptiveRetrieval:
    """Get the global adaptive retrieval instance."""
    return _adaptive_retrieval
