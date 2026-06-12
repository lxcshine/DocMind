# -*- coding: utf-8 -*-
"""
DocMind MCP Server — Model Context Protocol integration

Exposes DocMind's core capabilities as an MCP Server using the official
Python SDK (FastMCP). Any MCP-compatible client (Claude Desktop, Cursor,
VS Code Copilot, etc.) can discover and invoke these tools at runtime.

Architecture:
  MCP Client (Claude/Cursor/...)
       │  Streamable HTTP (JSON-RPC 2.0)
       ▼
  FastAPI /mcp  ←── mount mcp.streamable_http_app()
       │
  DocMind MCP Server
   ├── Tools (7):  search_knowledge_base, get_document, get_document_structure,
   │               get_page_content, search_web, memory_add, memory_search
   ├── Resources (2): document://list, document://{doc_id}/structure
   └── Prompts (1): knowledge_qa

Transport: Streamable HTTP (spec 2025-11-25)
  - Single /mcp endpoint, POST for JSON-RPC, GET/SSE for streaming
  - Stateless mode for horizontal scaling behind load balancers
  - Compatible with reverse proxies (nginx, cloud LBs)

Usage (Claude Desktop settings.json):
  {
    "mcpServers": {
      "docmind": {
        "url": "http://localhost:8000/mcp"
      }
    }
  }
"""

import asyncio
import json
import logging
import contextlib
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP instance — stateless_http=True for production deployment behind LBs
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "DocMind",
    stateless_http=True,
)


# ===========================================================================
# MCP Tools — functions the AI can call
# ===========================================================================

@mcp.tool()
async def search_knowledge_base(
    query: str,
    mode: str = "hybrid",
) -> str:
    """
    Search the DocMind knowledge base using RAG (Retrieval-Augmented Generation).

    Supports multiple query modes for different retrieval strategies:
    - "local": Vector similarity search for precise, local matches
    - "global": Knowledge graph traversal for broad, conceptual answers
    - "hybrid": Combines local + global for balanced results
    - "mix": Full pipeline — retrieval + LLM answer generation (recommended)
    - "naive": Simple vector search without graph

    Args:
        query: The search query text
        mode: Retrieval mode — "local", "global", "hybrid", "mix", "naive". Default "hybrid"

    Returns:
        Search result as text. For "mix" mode, returns a complete LLM-generated answer.
        For other modes, returns raw retrieval context.
    """
    try:
        from core.raganything import get_rag_instance, query as rag_query

        rag = get_rag_instance()
        if rag is None:
            return "Error: Knowledge base not initialized. Please upload documents first."

        result = await rag_query(rag, query, mode=mode)
        return result
    except Exception as e:
        logger.error(f"MCP search_knowledge_base error: {e}", exc_info=True)
        return f"Error searching knowledge base: {e}"


@mcp.tool()
async def get_document(doc_id: str) -> str:
    """
    Get metadata for a specific document in the knowledge base.

    Returns document name, description, type, page count, and processing status.

    Args:
        doc_id: The document ID to look up

    Returns:
        JSON string with document metadata
    """
    try:
        from core.tree_index import tree_index_store
        return tree_index_store.get_document_info(doc_id)
    except Exception as e:
        logger.error(f"MCP get_document error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_document_structure(doc_id: str) -> str:
    """
    Get the hierarchical structure (table of contents) of a document.

    Returns a tree of sections with titles, page ranges, and summaries.
    Does NOT include full text content — use get_page_content for that.

    Args:
        doc_id: The document ID

    Returns:
        JSON string with the document's hierarchical structure
    """
    try:
        from core.tree_index import tree_index_store
        return tree_index_store.get_document_structure(doc_id)
    except Exception as e:
        logger.error(f"MCP get_document_structure error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_page_content(doc_id: str, pages: str) -> str:
    """
    Get the text content of specific pages from a document.

    Args:
        doc_id: The document ID
        pages: Page specification — supports "5" (single), "3-7" (range), "1,3,5" (list)

    Returns:
        JSON string with page numbers and their text content
    """
    try:
        from core.tree_index import tree_index_store
        return tree_index_store.get_page_content(doc_id, pages)
    except Exception as e:
        logger.error(f"MCP get_page_content error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
async def search_web(
    query: str,
    max_pages: int = 5,
) -> str:
    """
    Perform a deep web search with real content scraping.

    Uses multiple search engines (Bing/Baidu/Sogou), scrapes the top results,
    and synthesizes a comprehensive answer using LLM.

    Args:
        query: The search query
        max_pages: Maximum number of pages to scrape (default 5, max 10)

    Returns:
        Synthesized answer with source citations
    """
    try:
        from core.web_scraper import WebScraper
        from infrastructure.llm_client import get_sync_llm

        max_pages = min(max_pages, 10)
        scraper = WebScraper(max_concurrent=3, timeout=30)

        # Step 1: Search engines to get URLs
        search_results = await _web_search_urls(query)
        if not search_results:
            return "No search results found for the given query."

        # Step 2: Scrape top pages
        urls = [r["url"] for r in search_results[:max_pages]]
        scraped = await scraper.scrape_urls(urls)

        # Step 3: Synthesize answer
        context_parts = []
        for i, page in enumerate(scraped, 1):
            if page.markdown and len(page.markdown.strip()) > 50:
                context_parts.append(f"[{i}] {page.title or 'Untitled'}\n{page.markdown[:3000]}")

        if not context_parts:
            return "Search completed but no usable content was extracted from the results."

        context = "\n\n---\n\n".join(context_parts)

        # Use LLM to synthesize
        llm = get_sync_llm()
        prompt = (
            f"Based on the following web search results, provide a comprehensive answer to the query.\n"
            f"Query: {query}\n\n"
            f"Search Results:\n{context}\n\n"
            f"Answer in the same language as the query. Cite sources with [1], [2], etc."
        )
        answer = llm.invoke(prompt)
        return answer.content if hasattr(answer, "content") else str(answer)

    except Exception as e:
        logger.error(f"MCP search_web error: {e}", exc_info=True)
        return f"Error performing web search: {e}"


@mcp.tool()
async def memory_add(
    content: str,
    memory_type: str = "semantic",
    keywords: str = "",
) -> str:
    """
    Add a new memory entry to DocMind's persistent memory system.

    Memory types:
    - "raw": Unprocessed user input (exact quotes, preferences)
    - "semantic": Extracted facts and knowledge (concepts, relationships)
    - "episodic": Event-based memories (what happened, when, where)
    - "procedural": How-to knowledge (step-by-step procedures, rules)

    Args:
        content: The memory content to store
        memory_type: Type of memory — "raw", "semantic", "episodic", "procedural". Default "semantic"
        keywords: Comma-separated keywords for retrieval (optional)

    Returns:
        Confirmation message with the memory entry ID
    """
    try:
        from core.memory import get_memory_service, MemoryEntry
        from datetime import datetime, timezone
        import hashlib

        service = get_memory_service()
        if service is None:
            return "Error: Memory service not initialized."

        entry_id = hashlib.md5(
            f"{content}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            keywords=kw_list,
            source_session_id="mcp",
            enabled=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        service.add_entry(entry)
        return f"Memory added successfully (id={entry_id}, type={memory_type})"

    except Exception as e:
        logger.error(f"MCP memory_add error: {e}", exc_info=True)
        return f"Error adding memory: {e}"


@mcp.tool()
async def memory_search(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search DocMind's persistent memory for relevant entries.

    Uses embedding-based semantic search to find the most relevant memories
    for the given query.

    Args:
        query: The search query
        top_k: Number of top results to return (default 5, max 20)

    Returns:
        JSON string with matching memory entries and their relevance scores
    """
    try:
        from core.memory import get_memory_service

        service = get_memory_service()
        if service is None:
            return json.dumps({"error": "Memory service not initialized"})

        top_k = min(top_k, 20)
        results = service._index.search(query, top_k=top_k)

        entries = []
        for entry_id, score in results:
            entry = service.get_entry(entry_id)
            if entry and entry.enabled:
                entries.append({
                    "id": entry.id,
                    "type": entry.type,
                    "content": entry.content,
                    "keywords": entry.keywords,
                    "relevance": round(score, 4),
                })

        return json.dumps(entries, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"MCP memory_search error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# ===========================================================================
# MCP Resources — read-only data the AI can fetch
# ===========================================================================

@mcp.resource("document://list")
def list_documents() -> str:
    """
    List all documents in the knowledge base.

    Returns a JSON array of document metadata entries including
    doc_id, doc_name, type, page_count, and status.
    """
    try:
        from core.tree_index import tree_index_store
        docs = tree_index_store.list_documents()
        return json.dumps(docs, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"MCP resource list_documents error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.resource("document://{doc_id}/structure")
def get_document_structure_resource(doc_id: str) -> str:
    """
    Get the hierarchical structure of a document as a resource.

    This is the resource version of the get_document_structure tool,
    allowing MCP clients to read document structure by URI.
    """
    try:
        from core.tree_index import tree_index_store
        return tree_index_store.get_document_structure(doc_id)
    except Exception as e:
        logger.error(f"MCP resource document structure error: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# ===========================================================================
# MCP Prompts — pre-built prompt templates
# ===========================================================================

@mcp.prompt()
def knowledge_qa(question: str) -> str:
    """
    Prompt template for knowledge base question answering.

    Generates a structured prompt that instructs the AI to answer a question
    using the DocMind knowledge base, with proper citation of sources.

    Args:
        question: The user's question to answer
    """
    return (
        "You are DocMind, an intelligent document assistant with access to a "
        "knowledge base. Answer the user's question using the available tools.\n\n"
        "Instructions:\n"
        "1. Use search_knowledge_base to find relevant information\n"
        "2. If needed, use get_document_structure to understand document layout\n"
        "3. Use get_page_content to read specific pages for detailed answers\n"
        "4. Always cite your sources with document names and page numbers\n"
        "5. If the knowledge base doesn't contain the answer, say so honestly\n"
        "6. Answer in the same language as the question\n\n"
        f"Question: {question}"
    )


# ===========================================================================
# Helper functions
# ===========================================================================

async def _web_search_urls(query: str) -> List[Dict[str, Any]]:
    """
    Search multiple engines and return deduplicated URL results.
    """
    import aiohttp

    results = []
    seen_urls = set()

    # Bing search
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.bing.microsoft.com/v7.0/search?q={query}&count=10"
            headers = {}
            bing_key = getattr(settings, "BING_API_KEY", "")
            if bing_key:
                headers["Ocp-Apim-Subscription-Key"] = bing_key
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("webPages", {}).get("value", []):
                            u = item.get("url", "")
                            if u and u not in seen_urls:
                                seen_urls.add(u)
                                results.append({
                                    "url": u,
                                    "title": item.get("name", ""),
                                    "description": item.get("snippet", ""),
                                })
    except Exception as e:
        logger.warning(f"Bing search failed: {e}")

    # Fallback: use DuckDuckGo HTML search if no Bing key or no results
    if not results:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://html.duckduckgo.com/html/?q={query}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        import re
                        links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html)
                        for link_url, title in links[:10]:
                            # DuckDuckGo uses redirect URLs; extract the real URL
                            real_url = link_url
                            if "uddg=" in link_url:
                                from urllib.parse import urlparse, parse_qs
                                parsed = urlparse(link_url)
                                params = parse_qs(parsed.query)
                                real_url = params.get("uddg", [link_url])[0]
                            if real_url not in seen_urls:
                                seen_urls.add(real_url)
                                results.append({
                                    "url": real_url,
                                    "title": re.sub(r"<[^>]+>", "", title),
                                    "description": "",
                                })
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

    return results


# ===========================================================================
# FastAPI integration helpers
# ===========================================================================

@asynccontextmanager
async def mcp_lifespan(app):
    """
    Lifespan context manager for MCP session manager.
    Mount this in the FastAPI app's lifespan to start/stop MCP sessions.
    """
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(mcp.session_manager.run())
        yield


def get_mcp_asgi_app():
    """
    Return the MCP ASGI application for mounting into FastAPI.

    Usage in main.py:
        from core.mcp_server import get_mcp_asgi_app, mcp_lifespan
        app.mount("/mcp", get_mcp_asgi_app())
    """
    return mcp.streamable_http_app()
