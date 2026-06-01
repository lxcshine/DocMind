"""
DocMind Backend - Agentic Retrieval

PageIndex-style LLM-powered tree search agent.
Two modes:
  - retrieve(): tool-calls only, returns context + sources (for streaming)
  - answer(): single-pass retrieve AND answer in one agentic flow

Agent tools:
  - get_document()           -- document metadata
  - get_document_structure() -- tree structure index
  - get_page_content()       -- text content of specific pages
"""

import json
import logging
from typing import Dict, List, Optional, Any, Callable
from openai import OpenAI

from core.tree_index import tree_index_store

logger = logging.getLogger(__name__)

RETRIEVE_AGENT_SYSTEM_PROMPT = """You are a document retrieval agent. Your ONLY job is to find and return relevant information from documents. You MUST call tools for EVERY document the user asks about. NEVER answer the question yourself -- just gather the information.

TOOLS:
1. get_document(doc_id) - Get document metadata: name, description, page count.
2. get_document_structure(doc_id) - Get the full tree structure to find relevant sections.
3. get_page_content(doc_id, pages) - Get text of specific pages. Use tight ranges: '5-7', '3,8', or '12'.

MANDATORY RULES:
- You MUST call get_document() for ALL documents available before doing anything else.
- You MUST call get_document_structure() for relevant documents to find the right sections.
- You MUST call get_page_content() to fetch actual text from the relevant pages.
- NEVER fetch the whole document -- only the sections that are relevant.
- After you have gathered all necessary information, output a single paragraph summarizing what you found (key concepts, formulas, page ranges).
- Do NOT output JSON, do NOT output code blocks. Just a plain text summary."""

ANSWER_AGENT_SYSTEM_PROMPT = """You are a document research assistant. You MUST use the tools to find relevant information from the knowledge base before answering. Never answer from your own knowledge without checking the documents first.

TOOLS:
1. get_document(doc_id) - Get document metadata: name, description, page count.
2. get_document_structure(doc_id) - Get the full tree structure to find relevant sections.
3. get_page_content(doc_id, pages) - Get text of specific pages. Use tight ranges: '5-7', '3,8', or '12'.

MANDATORY RULES:
- You MUST call get_document() for ALL documents available before doing anything else.
- You MUST call get_document_structure() for relevant documents to find what you need.
- You MUST call get_page_content() to fetch actual text from the relevant pages.
- NEVER fetch the whole document -- only the sections that are relevant.
- After gathering enough information, ANSWER the user's question directly.

FORMATTING RULES:
- Mathematical formulas MUST use LaTeX: $inline$ for inline, $$display$$ for display math.
- When comparing multiple items, use a properly formatted Markdown table with aligned columns.
- Table columns MUST be separated by | with header separator row using |---|.
- Use bullet points and numbered lists for clarity.
- Always respond in the same language as the user's question."""


class AgenticRetriever:
    """
    LLM-powered agentic document retriever.

    Single-pass architecture (PageIndex-style):
    LLM navigates document tree structures, fetches precise page content,
    and answers the user's question -- all in one agentic flow.
    """

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model: str = "gemini-2.5-flash",
            max_tool_rounds: int = 5,
    ):
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        logger.info(f"AgenticRetriever initialized: model={model}")

    def answer(
            self,
            query: str,
            doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Single-pass agentic retrieval + answering.

        On LLM failure, falls back to keyword retrieval + direct answer.
        """
        if doc_ids is None:
            docs = tree_index_store.list_documents()
            doc_ids = [d["doc_id"] for d in docs]

        if not doc_ids:
            return {
                "answer": "",
                "sources": [],
                "thinking": "No documents available in the knowledge base.",
            }

        try:
            return self._answer_via_llm(query, doc_ids)
        except Exception as e:
            logger.warning(f"Agentic answer via LLM failed ({e}), falling back to keyword retrieval")
            fallback = self._retrieve_keyword_fallback(query, doc_ids)
            return {
                "answer": "",
                "sources": fallback["sources"],
                "thinking": f"LLM retrieval unavailable ({e}), showing keyword-matched results.",
                "context": fallback["context"],
            }

    def _answer_via_llm(
            self,
            query: str,
            doc_ids: List[str],
    ) -> Dict[str, Any]:
        """LLM-powered single-pass agentic retrieval + answering."""

        messages = [
            {"role": "system", "content": ANSWER_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_initial_prompt(query, doc_ids)},
        ]

        tool_definitions = self._get_tool_definitions()
        tool_functions = self._get_tool_functions()

        doc_name_map: Dict[str, str] = {}
        raw_sources: List[Dict] = []

        for round_num in range(self.max_tool_rounds):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=0,
                max_tokens=4096,
            )

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                messages.append(self._format_assistant_tool_call(message))

                for tc in message.tool_calls:
                    func_name = tc.function.name
                    func = tool_functions.get(func_name)
                    if func:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = func(**args)
                        except Exception as e:
                            result = json.dumps({"error": str(e)})
                        self._track_source(func_name, args, result, doc_name_map, raw_sources)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        logger.info(f"Agent tool call: {func_name}({tc.function.arguments})")
            else:
                answer_text = message.content or ""
                sources = self._build_sources(raw_sources, doc_name_map)
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "thinking": "",
                }

        messages.append({
            "role": "user",
            "content": "Based on the information you have gathered, please answer the user's original question directly.",
        })
        final_response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        answer_text = final_response.choices[0].message.content or ""
        sources = self._build_sources(raw_sources, doc_name_map)
        return {
            "answer": answer_text,
            "sources": sources,
            "thinking": "",
        }

    def retrieve(
            self,
            query: str,
            doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Tool-calls only: gather context from documents without generating an answer.

        Used by the /stream endpoint to separate retrieval from answer generation,
        enabling real token-by-token streaming of the final answer.

        On LLM failure, falls back to keyword-based retrieval.
        """
        if doc_ids is None:
            docs = tree_index_store.list_documents()
            doc_ids = [d["doc_id"] for d in docs]

        if not doc_ids:
            return {"context": "", "sources": []}

        try:
            return self._retrieve_via_llm(query, doc_ids)
        except Exception as e:
            logger.warning(f"Agentic retrieval via LLM failed ({e}), falling back to keyword search")
            return self._retrieve_keyword_fallback(query, doc_ids)

    def _retrieve_via_llm(
            self,
            query: str,
            doc_ids: List[str],
    ) -> Dict[str, Any]:
        """LLM-powered agentic retrieval with tool calls."""

        messages = [
            {"role": "system", "content": RETRIEVE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_initial_prompt(query, doc_ids)},
        ]

        tool_definitions = self._get_tool_definitions()
        tool_functions = self._get_tool_functions()

        doc_name_map: Dict[str, str] = {}
        raw_sources: List[Dict] = []
        retrieval_summaries: List[str] = []
        fetched_content: Dict[str, str] = {}

        for round_num in range(self.max_tool_rounds):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto",
                temperature=0,
                max_tokens=2048,
            )

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                messages.append(self._format_assistant_tool_call(message))

                for tc in message.tool_calls:
                    func_name = tc.function.name
                    func = tool_functions.get(func_name)
                    if func:
                        try:
                            args = json.loads(tc.function.arguments)
                            result = func(**args)
                        except Exception as e:
                            result = json.dumps({"error": str(e)})
                        self._track_source(func_name, args, result, doc_name_map, raw_sources)
                        self._accumulate_content(func_name, args, result, doc_name_map, fetched_content)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        logger.info(f"Agent tool call: {func_name}({tc.function.arguments})")
            else:
                summary = message.content or ""
                if summary.strip():
                    retrieval_summaries.append(summary)
                break

        context = self._format_context(fetched_content, doc_name_map, retrieval_summaries)
        sources = self._build_sources(raw_sources, doc_name_map)
        return {"context": context, "sources": sources}

    def _accumulate_content(
            self,
            func_name: str,
            args: dict,
            result: str,
            doc_name_map: Dict[str, str],
            fetched_content: Dict[str, str],
    ) -> None:
        if func_name != "get_page_content":
            return
        doc_id = args.get("doc_id", "")
        pages = args.get("pages", "")
        doc_name = doc_name_map.get(doc_id, doc_id[:8])
        key = f"{doc_name}:{pages}"
        if key not in fetched_content:
            fetched_content[key] = result
        else:
            fetched_content[key] = result

    def _format_context(
            self,
            fetched_content: Dict[str, str],
            doc_name_map: Dict[str, str],
            summaries: List[str],
    ) -> str:
        parts = []
        for key, content in fetched_content.items():
            truncated = content[:8000] if len(content) > 8000 else content
            parts.append(f"## {key}\n\n{truncated}")
        if summaries:
            parts.append("## Agent Summary\n\n" + "\n".join(summaries))
        return "\n\n---\n\n".join(parts) if parts else ""

    def _retrieve_keyword_fallback(
            self,
            query: str,
            doc_ids: List[str],
    ) -> Dict[str, Any]:
        """Keyword-based fallback retrieval when LLM tool calling fails.

        Fetches all page content from all documents and uses simple
        TF-based keyword matching to rank pages by relevance.
        """
        logger.info(f"Keyword fallback retrieval for '{query[:80]}...' across {len(doc_ids)} docs")

        query_keywords = set(query.lower().split())
        scored_pages = []

        for doc_id in doc_ids:
            meta = tree_index_store.get_document_meta(doc_id)
            if not meta:
                continue
            page_count = meta.get("page_count", 0)
            doc_name = meta.get("doc_name", doc_id[:8])

            structure = tree_index_store.get_document_structure(doc_id)
            section_map = {}
            if structure:
                try:
                    struct_data = json.loads(structure) if isinstance(structure, str) else structure
                    self._flatten_structure(struct_data, section_map)
                except Exception:
                    pass

            max_pages_per_call = min(page_count, 10)
            for start in range(1, page_count + 1, max_pages_per_call):
                end = min(start + max_pages_per_call - 1, page_count)
                pages_str = str(start) if start == end else f"{start}-{end}"
                content = tree_index_store.get_page_content(doc_id, pages_str)
                if not content:
                    continue

                content_lower = content.lower()
                score = sum(1 for kw in query_keywords if kw in content_lower)
                if score == 0:
                    continue

                scored_pages.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "pages": pages_str,
                    "score": score,
                    "content": content,
                })

        if not scored_pages:
            for doc_id in doc_ids:
                meta = tree_index_store.get_document_meta(doc_id)
                if not meta:
                    continue
                page_count = meta.get("page_count", 0)
                doc_name = meta.get("doc_name", doc_id[:8])
                if page_count > 0:
                    first_half = min(page_count, max(1, page_count // 2))
                    pages_str = f"1-{first_half}"
                    content = tree_index_store.get_page_content(doc_id, pages_str)
                    if content:
                        scored_pages.append({
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "pages": pages_str,
                            "score": 1,
                            "content": content[:3000],
                        })

        scored_pages.sort(key=lambda x: x["score"], reverse=True)

        top_pages = scored_pages[:5]
        context_parts = []
        sources = []
        seen = set()

        for sp in top_pages:
            key = f"{sp['doc_id']}:{sp['pages']}"
            if key in seen:
                continue
            seen.add(key)
            truncated = sp["content"][:6000]
            context_parts.append(f"## {sp['doc_name']} (Pages {sp['pages']})\n\n{truncated}")
            sources.append({
                "doc_id": sp["doc_id"],
                "doc_name": sp["doc_name"],
                "pages": sp["pages"],
                "section": "",
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        logger.info(f"Keyword fallback found {len(top_pages)} relevant page ranges")

        return {"context": context, "sources": sources}

    @staticmethod
    def _flatten_structure(nodes, section_map, parent_title=""):
        """Flatten tree structure into a page-to-section mapping."""
        if not isinstance(nodes, list):
            return
        for node in nodes:
            title = node.get("title", "")
            start = node.get("start_index", 0)
            end = node.get("end_index", 0)
            if title and start:
                for p in range(start, end + 1):
                    section_map[p] = title
            if node.get("nodes"):
                AgenticRetriever._flatten_structure(node["nodes"], section_map, title)

    @staticmethod
    def _format_tool_messages_as_context(messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if content and len(content) > 50:
                    truncated = content[:6000] if len(content) > 6000 else content
                    parts.append(truncated)
        return "\n\n---\n\n".join(parts) if parts else ""

    def _build_initial_prompt(self, query: str, doc_ids: List[str]) -> str:
        docs_summary = []
        for did in doc_ids:
            meta = tree_index_store.get_document_meta(did)
            if meta:
                docs_summary.append(
                    f"- {did}: {meta.get('doc_name', 'Unknown')} "
                    f"({meta.get('page_count', 0)} pages) - {meta.get('doc_description', '')}"
                )

        return (
            f"User question: {query}\n\n"
            f"Available documents:\n"
            f"{chr(10).join(docs_summary) if docs_summary else 'None'}\n\n"
            f"Use the tools to find relevant information, then answer the question directly."
        )

    def _track_source(
            self,
            func_name: str,
            args: dict,
            result: str,
            doc_name_map: Dict[str, str],
            raw_sources: List[Dict],
    ) -> None:
        doc_id = args.get("doc_id", "")
        if func_name == "get_document":
            try:
                info = json.loads(result)
                if info.get("doc_name"):
                    doc_name_map[doc_id] = info["doc_name"]
            except Exception:
                pass
        elif func_name == "get_page_content":
            pages = args.get("pages", "")
            raw_sources.append({"doc_id": doc_id, "pages": pages})

    def _build_sources(
            self,
            raw_sources: List[Dict],
            doc_name_map: Dict[str, str],
    ) -> List[Dict]:
        seen = set()
        sources = []
        for rs in raw_sources:
            did = rs["doc_id"]
            pages = rs["pages"]
            key = f"{did}:{pages}"
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "doc_id": did,
                "doc_name": doc_name_map.get(did, did[:8]),
                "pages": pages,
                "section": "",
            })
        return sources

    @staticmethod
    def _format_assistant_tool_call(message) -> dict:
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }

    def _get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": "Get document metadata: name, description, page count, type, and status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "The document ID to get info for.",
                            }
                        },
                        "required": ["doc_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_structure",
                    "description": "Get the full tree structure (without text) to find relevant sections. Returns titles, hierarchy, and page ranges.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "The document ID to get the structure for.",
                            }
                        },
                        "required": ["doc_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_content",
                    "description": "Get the text content of specific pages. Use tight ranges: '5-7' for pages 5-7, '3,8' for pages 3 and 8, '12' for page 12.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "The document ID to get page content from.",
                            },
                            "pages": {
                                "type": "string",
                                "description": "Page range to fetch, e.g. '5-7', '3,8', or '12'.",
                            },
                        },
                        "required": ["doc_id", "pages"],
                    },
                },
            },
        ]

    def _get_tool_functions(self) -> Dict[str, Callable]:
        return {
            "get_document": lambda doc_id: tree_index_store.get_document_info(doc_id),
            "get_document_structure": lambda doc_id: tree_index_store.get_document_structure(doc_id),
            "get_page_content": lambda doc_id, pages: tree_index_store.get_page_content(doc_id, pages),
        }


def create_agentic_retriever(
        api_key: str = None,
        base_url: str = None,
        model: str = "gemini-2.5-flash",
) -> AgenticRetriever:
    return AgenticRetriever(api_key=api_key, base_url=base_url, model=model)
