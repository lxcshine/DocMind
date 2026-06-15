# -*- coding: utf-8 -*-
"""
Batch Embedding Service

Reduces API calls by batching multiple texts into a single embedding request.
OpenAI supports up to 2048 texts per batch.

Benefits:
  - Reduces API calls by 90%+ (1000 chunks → 1 batch instead of 1000 calls)
  - Faster processing (parallel embedding)
  - Lower cost (some providers offer batch discounts)
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

# OpenAI batch limit
MAX_BATCH_SIZE = 2048
# Conservative default (some models have lower limits)
DEFAULT_BATCH_SIZE = 512


class BatchEmbeddingService:
    """
    Batches multiple embedding requests into a single API call.
    
    Usage:
        service = BatchEmbeddingService()
        embeddings = await service.embed_batch(["text1", "text2", ...])
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.base_url = base_url or settings.EMBEDDING_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)
        self._client = None

    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=60.0,
            )
        return self._client

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            
        Returns:
            List of embedding vectors (numpy arrays)
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        client = self._get_client()
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(
            f"[BatchEmbedding] Processing {len(texts)} texts in "
            f"{total_batches} batch(es) (batch_size={batch_size})"
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                logger.debug(
                    f"[BatchEmbedding] Batch {batch_num}/{total_batches}: "
                    f"{len(batch)} texts"
                )
                
                response = await client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                
                # Extract embeddings in order
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in sorted(response.data, key=lambda x: x.index)
                ]
                
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(
                    f"[BatchEmbedding] Batch {batch_num}/{total_batches} complete: "
                    f"{len(batch_embeddings)} embeddings"
                )
                
            except Exception as e:
                logger.error(
                    f"[BatchEmbedding] Batch {batch_num}/{total_batches} failed: {e}"
                )
                # Fallback: embed individually
                fallback_embeddings = await self._embed_individually(batch)
                all_embeddings.extend(fallback_embeddings)

        logger.info(
            f"[BatchEmbedding] Complete: {len(all_embeddings)} embeddings generated"
        )
        
        return all_embeddings

    async def _embed_individually(self, texts: List[str]) -> List[np.ndarray]:
        """Fallback: embed texts one by one."""
        client = self._get_client()
        embeddings = []
        
        for text in texts:
            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=[text],
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"[BatchEmbedding] Individual embedding failed: {e}")
                # Return zero vector as last resort
                embeddings.append(np.zeros(settings.EMBEDDING_MAX_LENGTH, dtype=np.float32))
        
        return embeddings


# Singleton instance
_batch_embedding_service: Optional[BatchEmbeddingService] = None


def get_batch_embedding_service() -> BatchEmbeddingService:
    """Get or create the global BatchEmbeddingService."""
    global _batch_embedding_service
    if _batch_embedding_service is None:
        _batch_embedding_service = BatchEmbeddingService()
    return _batch_embedding_service


async def embed_batch(texts: List[str]) -> List[np.ndarray]:
    """
    Convenience function: embed multiple texts in batches.
    
    Example:
        embeddings = await embed_batch(["text1", "text2", ...])
    """
    service = get_batch_embedding_service()
    return await service.embed_batch(texts)
