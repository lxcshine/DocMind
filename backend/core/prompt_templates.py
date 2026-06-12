"""
Prompt templates for DocMind

Centralized prompt management with version tracking.
All prompts used across the application are registered here.

Version history:
  v1 — Initial centralized registry (chat + agent + multimodal prompts)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROMPT_REGISTRY_VERSION = "1.0.0"


class PromptRegistry:
    """
    Stable prompt container with atomic snapshot swapping and version tracking.

    Features:
      - Atomic swap: replace all prompts at once (no torn reads)
      - Version tracking: each prompt has a version tag
      - Hot-reload: swap prompts at runtime without restart
      - Audit log: every swap is logged with version info
    """

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}
        self._versions: Dict[str, str] = {}

    def swap(self, prompts: Dict[str, str], version: str = "") -> None:
        self._data = dict(prompts)
        if version:
            for key in prompts:
                self._versions[key] = version
        logger.info(f"[PromptRegistry] Swapped {len(prompts)} prompts, version={version or 'unversioned'}")

    def snapshot(self) -> Dict[str, str]:
        return dict(self._data)

    def register(self, key: str, value: str, version: str = "") -> None:
        self._data[key] = value
        if version:
            self._versions[key] = version

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(key, default)

    def version_of(self, key: str) -> str:
        return self._versions.get(key, "unknown")

    def list_all(self) -> Dict[str, Dict[str, str]]:
        """Return all prompts with their versions."""
        return {k: {"content": v[:100] + "...", "version": self._versions.get(k, "unknown")} for k, v in self._data.items()}


PROMPTS = PromptRegistry()

_V = PROMPT_REGISTRY_VERSION  # shorthand for version tag

# ===== System Prompts =====
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed, accurate descriptions."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "You are an expert data analyst. Provide detailed table analysis with specific insights."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "You are an expert mathematician. Provide detailed mathematical analysis."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "You are an expert content analyst specializing in {content_type} content."
)

# ===== Image Analysis Prompts =====
PROMPTS["vision_prompt"] = """Please analyze this image in detail and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Always use specific names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance (max 100 words)"
    }}
}}

Additional context:
- Section Path: {section_path}
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval.
Use a semantic entity_name; do not return file names or figure numbers unless they are the actual title."""

PROMPTS["vision_prompt_with_context"] = """Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
    - Describe the overall composition and layout
    - Identify all objects, people, text, and visual elements
    - Explain relationships between elements and how they relate to the surrounding context
    - Note colors, lighting, and visual style
    - Describe any actions or activities shown
    - Include technical details if relevant (charts, diagrams, etc.)
    - Reference connections to the surrounding content when relevant
    - Always use specific names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Document structure:
- Section Path: {section_path}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for knowledge retrieval.
Use a semantic entity_name; do not return file names or figure numbers unless they are the actual title."""

# ===== Table Analysis Prompts =====
PROMPTS["table_prompt"] = """Please analyze this table content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }}
}}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data."""

PROMPTS["table_prompt_with_context"] = """Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content."""

# ===== Equation Analysis Prompts =====
PROMPTS["equation_prompt"] = """Please analyze this mathematical equation and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }}
}}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance."""

PROMPTS["equation_prompt_with_context"] = """Please analyze this mathematical equation considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions in the context of surrounding content
    - Mathematical operations and functions used
    - Application domain and context based on surrounding material
    - Physical or theoretical significance
    - Relationship to other mathematical concepts mentioned in the context
    - Practical applications or use cases
    - How the equation relates to the broader discussion or framework
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance within the broader context."""

# ===== Generic Content Analysis Prompts =====
PROMPTS["generic_prompt"] = """Please analyze this {content_type} content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }}
}}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval."""

# ===== Modal Chunk Templates =====
PROMPTS["image_chunk"] = """
Image Content Analysis:
- Section Path: {section_path}
- Neighbor Text: {neighbor_text}
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""

# ===== Query-related Prompts =====
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Please briefly describe the main content, key elements, and important information in this image."
)
PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "You are a professional image analyst who can accurately describe image content."
)
PROMPTS["QUERY_TABLE_ANALYSIS"] = """Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""
PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "You are a professional data analyst who can accurately analyze table data."
)
PROMPTS["QUERY_EQUATION_ANALYSIS"] = """Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."""
PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "You are a mathematics expert who can clearly explain mathematical formulas."
)
PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."
)

# ============================================================================
# Chat & Agent Prompts (centralized from chat.py / agentic_retrieve.py)
# ============================================================================

PROMPTS.register(
    "DIRECT_CHAT_SYSTEM",
    (
        "You are a helpful research assistant. Always respond in the same language as the user's question. "
        "If the user asks in Chinese, you MUST answer in Chinese. "
        "Provide clear, well-structured answers with proper formatting. "
        "Mathematical formulas MUST use LaTeX: $inline$ for inline math, $$display$$ for display math. "
        "When comparing multiple items, use a properly formatted Markdown table."
    ),
    version=_V,
)

PROMPTS.register(
    "KB_CHAT_SYSTEM_BASE",
    (
        "You are a research assistant helping users understand documents. "
        "Always respond in the same language as the user's question. "
        "Base your answer on the retrieved context. "
        "Cite specific sections or pages when relevant. "
        "Use LaTeX for mathematical formulas: $inline$ for inline, $$display$$ for display. "
        "When comparing multiple items, use a properly formatted Markdown table. "
        "IMPORTANT: Markdown tables MUST follow this exact format:\n"
        "| Header1 | Header2 | Header3 |\n"
        "|---------|---------|---------|\n"
        "| Cell1   | Cell2   | Cell3   |\n"
        "Each row must start and end with '|'. The separator line must use at least 3 dashes '---' per column. "
        "NEVER put the separator on the same line as the header row."
    ),
    version=_V,
)

PROMPTS.register(
    "KB_CHAT_SYSTEM_DETAILED_SUFFIX",
    (
        " The user has requested a DETAILED response. "
        "Provide comprehensive, in-depth analysis with maximum detail. "
        "Include all relevant information, examples, and explanations. "
        "Your response should be thorough and extensive, up to 5000 words if necessary."
    ),
    version=_V,
)

PROMPTS.register(
    "RETRIEVE_AGENT_SYSTEM",
    (
        "You are a document retrieval agent. Your ONLY job is to find and return relevant information from documents. "
        "You MUST call tools for EVERY document the user asks about. NEVER answer the question yourself -- just gather the information.\n\n"
        "TOOLS:\n"
        "1. get_document(doc_id) - Get document metadata: name, description, page count.\n"
        "2. get_document_structure(doc_id) - Get the full tree structure to find relevant sections.\n"
        "3. get_page_content(doc_id, pages) - Get text of specific pages. Use tight ranges: '5-7', '3,8', or '12'.\n\n"
        "MANDATORY RULES:\n"
        "- You MUST call get_document() for ALL documents available before doing anything else.\n"
        "- You MUST call get_document_structure() for relevant documents to find the right sections.\n"
        "- You MUST call get_page_content() to fetch actual text from the relevant pages.\n"
        "- NEVER fetch the whole document -- only the sections that are relevant.\n"
        "- After you have gathered all necessary information, output a single paragraph summarizing what you found "
        "(key concepts, formulas, page ranges).\n"
        "- Do NOT output JSON, do NOT output code blocks. Just a plain text summary."
    ),
    version=_V,
)

PROMPTS.register(
    "ANSWER_AGENT_SYSTEM",
    (
        "You are a document research assistant. You MUST use the tools to find relevant information from the knowledge base "
        "before answering. Never answer from your own knowledge without checking the documents first.\n\n"
        "TOOLS:\n"
        "1. get_document(doc_id) - Get document metadata: name, description, page count.\n"
        "2. get_document_structure(doc_id) - Get the full tree structure to find relevant sections.\n"
        "3. get_page_content(doc_id, pages) - Get text of specific pages. Use tight ranges: '5-7', '3,8', or '12'.\n\n"
        "MANDATORY RULES:\n"
        "- You MUST call get_document() for ALL documents available before doing anything else.\n"
        "- You MUST call get_document_structure() for relevant documents to find what you need.\n"
        "- You MUST call get_page_content() to fetch actual text from the relevant pages.\n"
        "- NEVER fetch the whole document -- only the sections that are relevant.\n"
        "- After gathering enough information, ANSWER the user's question directly.\n\n"
        "FORMATTING RULES:\n"
        "- Mathematical formulas MUST use LaTeX: $inline$ for inline, $$display$$ for display math.\n"
        "- When comparing multiple items, use a properly formatted Markdown table with aligned columns.\n"
        "- Table columns MUST be separated by | with header separator row using |---|.\n"
        "- Use bullet points and numbered lists for clarity.\n"
        "- Always respond in the same language as the user's question."
    ),
    version=_V,
)

PROMPTS.register(
    "RAG_STREAM_PASS_THROUGH",
    (
        "You are a knowledge base assistant. You will receive a pre-generated answer "
        "from a RAG system. Your job is to present this answer to the user as-is, "
        "preserving all formatting, formulas, and tables. Do NOT add or remove content. "
        "Simply output the answer you receive. Always respond in the same language as the user's question."
    ),
    version=_V,
)