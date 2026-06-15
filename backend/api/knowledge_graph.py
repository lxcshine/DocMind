# -*- coding: utf-8 -*-
"""
Knowledge Graph API

Provides endpoints for visualizing the knowledge graph:
  - GET /api/knowledge-graph/full — Full graph (all entities + relations)
  - GET /api/knowledge-graph/document/{doc_id} — Graph for a specific document
  - GET /api/knowledge-graph/search — Search nodes by keyword
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from core.raganything import get_rag_instance
from infrastructure.response import success_response, error_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.get("/full")
async def get_full_graph(
    max_nodes: int = Query(500, ge=10, le=2000, description="Maximum nodes to return"),
    max_edges: int = Query(1000, ge=10, le=5000, description="Maximum edges to return"),
):
    """
    Get the full knowledge graph for visualization.
    
    Returns nodes (entities) and edges (relations) in a format
    suitable for D3.js force-directed graph rendering.
    """
    try:
        rag = get_rag_instance()
        if rag is None:
            return error_response("RAG not initialized", code=500)

        # Access LightRAG's internal graph storage
        graph_data = await _extract_graph_data(rag, max_nodes, max_edges)
        
        return success_response({
            "nodes": graph_data["nodes"],
            "edges": graph_data["edges"],
            "stats": {
                "total_nodes": len(graph_data["nodes"]),
                "total_edges": len(graph_data["edges"]),
                "node_types": _count_by_type(graph_data["nodes"], "type"),
            },
        })
    except Exception as e:
        logger.error(f"Failed to get full graph: {e}", exc_info=True)
        return error_response(str(e), code=500)


@router.get("/document/{doc_id}")
async def get_document_graph(
    doc_id: str,
    max_nodes: int = Query(200, ge=10, le=1000),
    max_edges: int = Query(500, ge=10, le=3000),
):
    """
    Get the knowledge graph for a specific document.
    
    Filters the graph to only include entities and relations
    extracted from the specified document.
    """
    try:
        rag = get_rag_instance()
        if rag is None:
            return error_response("RAG not initialized", code=500)

        graph_data = await _extract_document_graph(rag, doc_id, max_nodes, max_edges)
        
        return success_response({
            "nodes": graph_data["nodes"],
            "edges": graph_data["edges"],
            "document_id": doc_id,
            "stats": {
                "total_nodes": len(graph_data["nodes"]),
                "total_edges": len(graph_data["edges"]),
            },
        })
    except Exception as e:
        logger.error(f"Failed to get document graph for {doc_id}: {e}", exc_info=True)
        return error_response(str(e), code=500)


@router.get("/search")
async def search_graph(
    query: str = Query(..., min_length=1, description="Search query"),
    max_results: int = Query(50, ge=5, le=200),
):
    """
    Search for nodes in the knowledge graph by keyword.
    
    Returns matching entities and their immediate neighbors.
    """
    try:
        rag = get_rag_instance()
        if rag is None:
            return error_response("RAG not initialized", code=500)

        graph_data = await _search_graph(rag, query, max_results)
        
        return success_response({
            "nodes": graph_data["nodes"],
            "edges": graph_data["edges"],
            "query": query,
            "stats": {
                "matched_nodes": len(graph_data["nodes"]),
                "connected_edges": len(graph_data["edges"]),
            },
        })
    except Exception as e:
        logger.error(f"Failed to search graph: {e}", exc_info=True)
        return error_response(str(e), code=500)


# ============================================================================
# Internal helpers
# ============================================================================

async def _extract_graph_data(
    rag,
    max_nodes: int,
    max_edges: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract graph data from LightRAG's internal storage.
    
    LightRAG stores entities and relationships in its working directory.
    We read the graph structure and convert it to a D3-friendly format.
    """
    import json
    from pathlib import Path
    
    working_dir = Path(rag.config.working_dir)
    
    nodes = []
    edges = []
    
    # Try to load entity data from LightRAG's storage
    entities_file = working_dir / "kv_store_llm_response_cache.json"
    if entities_file.exists():
        try:
            with open(entities_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            
            # Extract entities from cache
            seen_entities = set()
            for key, value in cache.items():
                if "entity" in key.lower():
                    try:
                        entity_data = json.loads(value) if isinstance(value, str) else value
                        if isinstance(entity_data, dict):
                            entity_name = entity_data.get("entity_name", key)
                            if entity_name not in seen_entities and len(nodes) < max_nodes:
                                nodes.append({
                                    "id": entity_name,
                                    "label": entity_name,
                                    "type": "entity",
                                    "properties": {
                                        "description": entity_data.get("description", ""),
                                        "entity_type": entity_data.get("entity_type", "unknown"),
                                    },
                                })
                                seen_entities.add(entity_name)
                    except (json.JSONDecodeError, AttributeError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to load entities from {entities_file}: {e}")
    
    # Try to load relationship data
    relationships_file = working_dir / "kv_store_full_docs.json"
    if relationships_file.exists():
        try:
            with open(relationships_file, "r", encoding="utf-8") as f:
                docs = json.load(f)
            
            for key, value in docs.items():
                if len(edges) >= max_edges:
                    break
                try:
                    rel_data = json.loads(value) if isinstance(value, str) else value
                    if isinstance(rel_data, dict) and "relationships" in rel_data:
                        for rel in rel_data["relationships"]:
                            source = rel.get("source", "")
                            target = rel.get("target", "")
                            if source and target:
                                edges.append({
                                    "source": source,
                                    "target": target,
                                    "relation": rel.get("relationship", "related_to"),
                                    "weight": rel.get("weight", 1.0),
                                })
                except (json.JSONDecodeError, AttributeError):
                    continue
        except Exception as e:
            logger.warning(f"Failed to load relationships: {e}")
    
    # If no data found in files, try LightRAG's in-memory graph
    if not nodes and hasattr(rag, "rag") and hasattr(rag.rag, "chunk_entity_relation_graph"):
        try:
            graph = rag.rag.chunk_entity_relation_graph
            # Extract nodes
            for node_id in list(graph.nodes())[:max_nodes]:
                node_data = graph.nodes[node_id]
                nodes.append({
                    "id": node_id,
                    "label": node_data.get("entity_name", node_id),
                    "type": "entity",
                    "properties": {
                        "description": node_data.get("description", ""),
                        "entity_type": node_data.get("entity_type", "unknown"),
                    },
                })
            
            # Extract edges
            for source, target, edge_data in list(graph.edges(data=True))[:max_edges]:
                edges.append({
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relationship", "related_to"),
                    "weight": edge_data.get("weight", 1.0),
                })
        except Exception as e:
            logger.warning(f"Failed to extract from in-memory graph: {e}")
    
    return {"nodes": nodes, "edges": edges}


async def _extract_document_graph(
    rag,
    doc_id: str,
    max_nodes: int,
    max_edges: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract graph data filtered by document ID."""
    # For now, return the full graph (filtering by doc_id requires
    # tracking which entities came from which document, which LightRAG
    # doesn't expose directly yet)
    # TODO: Implement document-level filtering when LightRAG supports it
    return await _extract_graph_data(rag, max_nodes, max_edges)


async def _search_graph(
    rag,
    query: str,
    max_results: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Search for nodes matching the query."""
    full_graph = await _extract_graph_data(rag, max_results * 2, max_results * 4)
    
    # Filter nodes by query
    query_lower = query.lower()
    matched_nodes = [
        node for node in full_graph["nodes"]
        if query_lower in node["label"].lower()
        or query_lower in node.get("properties", {}).get("description", "").lower()
    ][:max_results]
    
    matched_ids = {node["id"] for node in matched_nodes}
    
    # Filter edges to only include those connected to matched nodes
    matched_edges = [
        edge for edge in full_graph["edges"]
        if edge["source"] in matched_ids or edge["target"] in matched_ids
    ]
    
    return {"nodes": matched_nodes, "edges": matched_edges}


def _count_by_type(items: List[Dict], key: str) -> Dict[str, int]:
    """Count items by a specific key."""
    counts = {}
    for item in items:
        value = item.get(key, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts
