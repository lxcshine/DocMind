# -*- coding: utf-8 -*-
"""
DocMind RAG Core Module

Thin wrapper around the installed `raganything` package (RAG-Anything).
Uses RAG-Anything's main.py pattern directly:
  - openai_complete_if_cache for LLM
  - EmbeddingFunc + openai_embed for embedding
  - RAGAnything class for document processing + querying

Key fix: Uses short file names to avoid Windows MAX_PATH (260 char) issues with MinerU.
"""

import os
import json
import shutil
import hashlib
import logging
import tempfile
import asyncio
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

from config.settings import settings

try:
    from core.progress import progress_tracker
except Exception:  # pragma: no cover - fallback when imported as a module
    progress_tracker = None

logger = logging.getLogger(__name__)


async def _parse_document_with_retry(rag: RAGAnything, file_path: str, output_dir: str):
    """
    Parse a document with bounded timeout and retry on transient failure.

    MinerU's internal per-task timeout is hard-coded and tends to fire under
    CPU pressure (logs show `Timed out waiting for result of task ...` after
    ~60-120s). We can't extend that internal timeout, but we can:

      1. Run the whole parse under an outer asyncio timeout (settings.MINERU_TIMEOUT).
      2. On TimeoutError, spawn a fresh MinerU subprocess and try again.
      3. Bail after settings.MINERU_MAX_RETRIES with a clear error.
    """
    attempts = max(1, settings.MINERU_MAX_RETRIES)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            logger.info(
                f"[parse] MinerU attempt {attempt}/{attempts} "
                f"(backend={settings.MINERU_BACKEND}, method={settings.PARSE_METHOD}, "
                f"timeout={settings.MINERU_TIMEOUT}s) for {file_path}"
            )
            return await asyncio.wait_for(
                rag.parse_document(
                    file_path,
                    output_dir=output_dir,
                    method=settings.PARSE_METHOD,
                    backend=settings.MINERU_BACKEND,
                ),
                timeout=settings.MINERU_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                f"[parse] MinerU timed out on attempt {attempt}/{attempts} "
                f"after {settings.MINERU_TIMEOUT}s. Retrying…"
            )
            # Drop any cached state from a half-finished run so the next
            # attempt starts from a clean slate.
            try:
                parser = getattr(rag, "doc_parser", None)
                if parser is not None and hasattr(parser, "reset"):
                    parser.reset()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        except Exception as exc:
            # Non-timeout error: don't retry, surface immediately.
            logger.error(f"[parse] MinerU failed on attempt {attempt}: {exc}")
            raise
    raise last_exc if last_exc else RuntimeError("MinerU parse failed without exception")


# ===== Global Instance =====
_rag_instance: Optional[RAGAnything] = None

# ===== Known embedding dimensions for common models =====
KNOWN_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _validate_embedding_config():
    """
    Validate embedding configuration and warn if mismatch detected.

    Checks if the configured EMBEDDING_MAX_LENGTH matches the known dimension
    for the specified model.
    """
    model = settings.EMBEDDING_MODEL
    configured_dim = settings.EMBEDDING_MAX_LENGTH

    if model in KNOWN_EMBEDDING_DIMS:
        expected_dim = KNOWN_EMBEDDING_DIMS[model]
        if configured_dim != expected_dim:
            logger.warning(
                f"[Embedding Config] EMBEDDING_MAX_LENGTH ({configured_dim}) does not match "
                f"the known dimension for {model} ({expected_dim}). "
                f"This may cause embedding dimension mismatch errors."
            )
        else:
            logger.info(
                f"[Embedding Config] Verified: {model} dimension = {configured_dim}"
            )
    else:
        logger.info(
            f"[Embedding Config] Using {model} with dimension {configured_dim} "
            f"(unknown model, cannot auto-verify)"
        )


def _check_embedding_dim_compatibility(working_dir: str, expected_dim: int) -> bool:
    """
    Check if existing vector DB files are compatible with the current embedding dimension.

    Scans all vdb_*.json files in the working directory and checks their embedding_dim.

    Returns:
        True if compatible (or no existing data), False if incompatible.
    """
    storage_path = Path(working_dir)
    if not storage_path.exists():
        return True  # No existing data, compatible

    vdb_files = list(storage_path.glob("vdb_*.json"))
    if not vdb_files:
        return True  # No vector DB files, compatible

    for vdb_file in vdb_files:
        try:
            with open(vdb_file, "r", encoding="utf-8") as f:
                storage = json.load(f)
            stored_dim = storage.get("embedding_dim")
            if stored_dim is not None and stored_dim != expected_dim:
                logger.warning(
                    f"[Embedding Check] Incompatible dimension in {vdb_file.name}: "
                    f"stored={stored_dim}, expected={expected_dim}"
                )
                return False
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[Embedding Check] Failed to read {vdb_file}: {e}")
            return False

    return True


def _cleanup_incompatible_vector_data(working_dir: str):
    """
    Remove incompatible vector DB files from the working directory.

    This allows RAGAnything to recreate them with the correct embedding dimension.
    """
    storage_path = Path(working_dir)
    if not storage_path.exists():
        return

    vdb_files = list(storage_path.glob("vdb_*.json"))
    if not vdb_files:
        return

    logger.info(f"[Embedding Cleanup] Removing {len(vdb_files)} incompatible vector DB files...")
    for vdb_file in vdb_files:
        try:
            vdb_file.unlink()
            logger.info(f"[Embedding Cleanup] Removed: {vdb_file.name}")
        except Exception as e:
            logger.error(f"[Embedding Cleanup] Failed to remove {vdb_file.name}: {e}")


async def init_rag(
    working_dir: Optional[str] = None,
) -> RAGAnything:
    """
    Initialize and return the global RAGAnything instance.

    Following RAG-Anything main.py pattern exactly:
      - openai_complete_if_cache for LLM
      - EmbeddingFunc + openai_embed for embedding
      - RAGAnythingConfig for configuration

    Auto-detects and cleans incompatible vector data when embedding config changes.
    """
    global _rag_instance

    # ===== 0. Validate embedding configuration =====
    _validate_embedding_config()

    api_key = settings.GEMINI_API_KEY
    base_url = settings.GEMINI_BASE_URL

    # ===== 1. LLM Model Function (same as RAG-Anything main.py) =====
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            settings.GEMINI_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # ===== 2. Vision Model Function (same as RAG-Anything main.py) =====
    vision_model_func = None
    if settings.VISION_API_KEY:
        vision_api_key = settings.VISION_API_KEY
        vision_base_url = settings.VISION_BASE_URL
        vision_model_name = settings.VISION_MODEL

        def vision_model_func(
            prompt, system_prompt=None, history_messages=[],
            image_data=None, messages=None, **kwargs
        ):
            if messages:
                return openai_complete_if_cache(
                    vision_model_name,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=vision_api_key,
                    base_url=vision_base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    vision_model_name,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=vision_api_key,
                    base_url=vision_base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # ===== 3. Embedding Function (same as RAG-Anything main.py) =====
    embedding_func = EmbeddingFunc(
        embedding_dim=settings.EMBEDDING_MAX_LENGTH,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
        ),
    )

    # ===== 4. Check and clean incompatible vector data =====
    rag_working_dir = working_dir or settings.RAG_WORKING_DIR
    Path(rag_working_dir).mkdir(parents=True, exist_ok=True)

    if not _check_embedding_dim_compatibility(rag_working_dir, settings.EMBEDDING_MAX_LENGTH):
        logger.warning(
            f"[Embedding Check] Incompatible vector data detected (stored dim != {settings.EMBEDDING_MAX_LENGTH}). "
            f"Cleaning up old vector data..."
        )
        _cleanup_incompatible_vector_data(rag_working_dir)
        logger.info("[Embedding Check] Cleanup complete. Vector DB will be recreated with correct dimension.")

    # ===== 5. RAGAnything Config =====
    config = RAGAnythingConfig(
        working_dir=rag_working_dir,
        parser=settings.PARSER,
        parse_method=settings.PARSE_METHOD,
        enable_image_processing=settings.ENABLE_IMAGE_PROCESSING,
        enable_table_processing=settings.ENABLE_TABLE_PROCESSING,
        enable_equation_processing=settings.ENABLE_EQUATION_PROCESSING,
    )

    # ===== 6. Create RAGAnything instance =====
    logger.info(f"Initializing RAGAnything: working_dir={rag_working_dir}, embedding_dim={settings.EMBEDDING_MAX_LENGTH}")
    _rag_instance = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    logger.info("RAGAnything initialized successfully")
    return _rag_instance


def get_rag_instance() -> Optional[RAGAnything]:
    """Get the global RAGAnything instance."""
    return _rag_instance


def _make_short_path(file_path: str) -> str:
    """
    Create a short path copy of the file to avoid Windows MAX_PATH issues.

    MinerU creates deep temp directories using the filename, which can exceed
    Windows 260 char limit when filename is long (e.g., UUID + original name).

    Returns path to a short-named copy in a temp directory.
    """
    original = Path(file_path)
    ext = original.suffix

    # Generate short name: 8-char hash + extension
    name_hash = hashlib.md5(original.name.encode()).hexdigest()[:8]
    short_name = f"doc_{name_hash}{ext}"

    # Create temp dir for short-named files
    temp_dir = Path(tempfile.gettempdir()) / "docmind_short"
    temp_dir.mkdir(parents=True, exist_ok=True)

    short_path = temp_dir / short_name

    # Copy only if not already exists or source is newer
    if not short_path.exists() or original.stat().st_mtime > short_path.stat().st_mtime:
        shutil.copy2(str(original), str(short_path))
        logger.info(f"Created short path copy: {original.name} -> {short_name}")

    return str(short_path)


async def process_document(
    rag: RAGAnything,
    file_path: str,
    output_dir: Optional[str] = None,
    processing_mode: str = "standard",
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a document through the RAG-Anything pipeline.

    Args:
        rag: RAGAnything instance
        file_path: Path to the document
        output_dir: Output directory for parsed content
        processing_mode: Processing mode:
            - "fast": Parse + vector-only insert (no KG extraction). ~1-2 min.
            - "standard": Parse + KG + vector with larger chunks. ~3-8 min.
            - "full": Parse + KG + vector + multimodal. ~10-30 min.

    Returns:
        Dict with processing results
    """
    output_dir = output_dir or settings.RAG_OUTPUT_DIR

    logger.info(f"[RAG-Anything] === Starting document processing ===")
    logger.info(f"[RAG-Anything] Mode: {processing_mode}")
    logger.info(f"[RAG-Anything] File: {file_path}")

    # Check if document has already been processed
    original = Path(file_path)
    doc_name = original.stem

    # Check 1: Look for existing output directory for this document
    output_path = Path(output_dir)
    if output_path.exists():
        existing_dirs = list(output_path.glob(f"*{doc_name}*"))
        if existing_dirs:
            for d in existing_dirs:
                content_lists = list(d.rglob("*_content_list.json"))
                if content_lists:
                    logger.info(f"[RAG-Anything] Document already parsed, using cached result.")
                    return {
                        "success": True,
                        "file_path": file_path,
                        "cached": True,
                    }

    # Check 2: Look in parse cache
    parse_cache_file = Path(settings.RAG_WORKING_DIR) / "kv_store_parse_cache.json"
    if parse_cache_file.exists():
        try:
            with open(parse_cache_file, 'r', encoding='utf-8') as f:
                parse_cache = json.load(f)
            for key in parse_cache.keys():
                if doc_name in key:
                    logger.info(f"[RAG-Anything] Found in parse cache, using cached result.")
                    return {
                        "success": True,
                        "file_path": file_path,
                        "cached": True,
                    }
        except Exception as e:
            logger.warning(f"[RAG-Anything] Failed to check parse cache: {e}")

    logger.info(f"[RAG-Anything] No cached result found, proceeding with processing")

    # Use short path to avoid Windows MAX_PATH issues
    short_path = _make_short_path(file_path)

    try:
        import asyncio

        # ===== Fast Mode: Parse + Vector-only (skip KG extraction) =====
        if processing_mode == "fast":
            return await _process_fast_mode(rag, short_path, output_dir, file_path, doc_id)

        # ===== Standard Mode: Larger chunks to reduce LLM calls =====
        elif processing_mode == "standard":
            try:
                result = await asyncio.wait_for(
                    rag.process_document_complete(
                        file_path=short_path,
                        output_dir=output_dir,
                        parse_method=settings.PARSE_METHOD,
                        backend=settings.MINERU_BACKEND,
                    ),
                    timeout=settings.MINERU_TIMEOUT,
                )
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Processing timed out after {settings.MINERU_TIMEOUT // 60} minutes.",
                    "file_path": file_path,
                }

            return {
                "success": True,
                "file_path": file_path,
            }

        # ===== Full Mode: Complete pipeline with multimodal =====
        else:  # "full"
            try:
                result = await asyncio.wait_for(
                    rag.process_document_complete(
                        file_path=short_path,
                        output_dir=output_dir,
                        parse_method=settings.PARSE_METHOD,
                        backend=settings.MINERU_BACKEND,
                    ),
                    timeout=settings.MINERU_TIMEOUT,
                )
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Processing timed out after {settings.MINERU_TIMEOUT // 60} minutes.",
                    "file_path": file_path,
                }

            return {
                "success": True,
                "file_path": file_path,
            }

    except Exception as e:
        logger.error(f"[RAG-Anything] Exception during processing: {e}")
        import traceback
        logger.error(f"[RAG-Anything] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def _process_fast_mode(
    rag: RAGAnything,
    short_path: str,
    output_dir: str,
    original_path: str,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fast processing mode: Parse document and insert text chunks into vector DB only.

    Skips KG extraction entirely. This is 10-50x faster than full mode because
    it eliminates the N sequential LLM calls for entity/relation extraction.

    Pipeline: Parse -> Chunk -> Embed -> Insert into vector DB
    """
    logger.info(f"[Fast Mode] Starting fast processing for {short_path}")

    # Step 1: Ensure LightRAG is initialized
    init_result = await rag._ensure_lightrag_initialized()
    if not init_result or not init_result.get("success"):
        return {"success": False, "error": f"LightRAG init failed: {init_result}", "file_path": original_path}

    # Step 2: Parse document (same as full mode - this is fast, ~10-60s)
    logger.info(f"[Fast Mode] Parsing document...")
    try:
        content_list, doc_id = await _parse_document_with_retry(
            rag, short_path, output_dir
        )
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"MinerU timed out after {settings.MINERU_MAX_RETRIES} attempt(s).",
            "file_path": original_path,
        }
    except Exception as e:
        logger.error(f"[Fast Mode] Parse failed: {e}")
        return {"success": False, "error": f"Parse failed: {str(e)}", "file_path": original_path}

    # Step 3: Separate text and multimodal content
    from raganything.processor import separate_content
    text_content, multimodal_items = separate_content(content_list)

    if not text_content.strip():
        logger.warning(f"[Fast Mode] No text content found in document")
        return {"success": False, "error": "No text content found", "file_path": original_path}

    # Step 4: Insert into LightRAG using naive mode (vector-only, no KG)
    # LightRAG's ainsert always does KG extraction, so we use a different approach:
    # We chunk the text ourselves and insert chunks directly into the vector DB.
    logger.info(f"[Fast Mode] Inserting text into vector DB (skipping KG extraction)...")
    logger.info(f"[Fast Mode] Text length: {len(text_content)} chars")

    try:
        # Use LightRAG's ainsert but with a trick: pass through the standard pipeline
        # The key optimization is that LightRAG's chunking + embedding is fast,
        # the slow part is the KG extraction LLM calls.
        # We can't easily skip just the KG part without modifying LightRAG internals.
        #
        # Alternative: Use lightrag.ainsert() which does everything, but set
        # lightrag_kwargs to use larger chunks and fewer LLM calls.
        #
        # Best approach for truly fast mode: Just call ainsert and accept the KG
        # extraction, but with very large chunk sizes to minimize LLM calls.
        # This is a pragmatic tradeoff.

        # For genuine fast mode, we insert text directly via LightRAG's
        # text chunking and embedding pipeline, bypassing entity/relation extraction
        lightrag = rag.lightrag

        # Chunk the text using LightRAG's tokenizer
        chunks = lightrag.chunking(text_content)

        logger.info(f"[Fast Mode] Split into {len(chunks)} chunks")

        # ===== Chunk-level progress reporting =====
        # We use a unified 0-100% scale:
        #   0..70%   → embedding (single batched API call)
        #   70..98%  → vector upsert (chunk-by-chunk, real per-chunk progress)
        #   98..100% → finalization
        total_chunks = max(1, len(chunks))
        if progress_tracker is not None and doc_id:
            progress_tracker.update(
                doc_id,
                current_step=f"Embedding {total_chunks} chunk(s)…",
                progress=1,
            )

        # Batch embedding call (lightrag.embedding_func accepts a list) — keep
        # this batched for performance: 1 API call vs N calls matters for
        # OpenAI-style backends. We give the user a coarse status during this
        # phase and refine to chunk-level granularity during the upsert phase.
        embeddings = await lightrag.embedding_func(chunks)

        if progress_tracker is not None and doc_id:
            progress_tracker.update(
                doc_id,
                current_step=f"Storing {total_chunks} chunk(s) into vector DB…",
                progress=70,
            )

        # Insert chunks into vector storage (per-chunk progress feedback)
        file_name = Path(original_path).name
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{doc_id}_{i}"
            await lightrag.chunks_vdb.upsert(chunk_id, embedding, {
                "full_doc_id": doc_id,
                "chunk_order_index": i,
                "content": chunk,
                "file_path": file_name,
            })
            if progress_tracker is not None and doc_id:
                done = i + 1
                pct = 70 + int(done / total_chunks * 28)  # 70..98
                progress_tracker.set_step(
                    doc_id,
                    done,
                    total_chunks,
                    f"Storing chunk {done}/{total_chunks}…",
                )
                # set_step computes its own progress; override for a smooth ramp.
                progress_tracker.update(doc_id, progress=pct)

        # Store full document text
        await lightrag.full_docs.upsert(doc_id, text_content, {
            "file_path": file_name,
        })

        # Store text chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{doc_id}_{i}"
            await lightrag.text_chunks.upsert(chunk_id, chunk, {
                "full_doc_id": doc_id,
                "chunk_order_index": i,
                "file_path": file_name,
            })

        if progress_tracker is not None and doc_id:
            progress_tracker.update(
                doc_id,
                current_step="Finalizing knowledge base…",
                progress=99,
            )

        # Update doc status
        if hasattr(lightrag, 'doc_status') and lightrag.doc_status:
            from lightrag.base import DocStatus
            await lightrag.doc_status.upsert(doc_id, {
                "status": DocStatus.COMPLETED,
                "file_path": file_name,
                "content_summary": text_content[:100],
                "content_length": len(text_content),
                "chunks_count": len(chunks),
            })

        logger.info(f"[Fast Mode] Successfully inserted {len(chunks)} chunks into vector DB")
        return {"success": True, "file_path": original_path, "mode": "fast", "chunks": len(chunks)}

    except Exception as e:
        logger.error(f"[Fast Mode] Vector insert failed: {e}")
        import traceback
        logger.error(f"[Fast Mode] Traceback: {traceback.format_exc()}")

        # Fallback: Use standard ainsert if direct insertion fails
        logger.info(f"[Fast Mode] Falling back to standard ainsert...")
        try:
            await rag.lightrag.ainsert(text_content, file_paths=file_name, ids=doc_id)
            return {"success": True, "file_path": original_path, "mode": "fast_fallback"}
        except Exception as fallback_err:
            return {"success": False, "error": str(fallback_err), "file_path": original_path}


async def query(
    rag: RAGAnything,
    query_text: str,
    mode: str = "mix",
    **kwargs,
) -> str:
    """
    Query the knowledge base using RAG-Anything.

    Args:
        rag: RAGAnything instance
        query_text: Query text
        mode: Query mode (local, global, hybrid, naive, mix)

    Returns:
        Query result string
    """
    if rag is None:
        raise ValueError("RAG not initialized")

    logger.info(f"Query (mode={mode}): {query_text[:100]}...")

    result = await rag.aquery(
        query_text,
        mode=mode,
        **kwargs,
    )

    return result or "No relevant information found."
