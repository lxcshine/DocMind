"""
Context extraction for multimodal processing

Based on RAG-Anything's ContextExtractor, adapted for DocMind.
Provides surrounding context for items in MinerU content lists
to enable context-aware analysis of images, tables, and equations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context extraction"""
    context_window: int = 1
    """Number of pages/chunks to include before and after current item."""
    context_mode: str = "page"
    """Context extraction mode: 'page' for page-based, 'chunk' for chunk-based."""
    max_context_tokens: int = 2000
    """Maximum number of tokens in extracted context."""
    include_headers: bool = True
    """Whether to include document headers and titles in context."""
    include_captions: bool = True
    """Whether to include image/table captions in context."""
    filter_content_types: List[str] = field(default_factory=lambda: ["text"])
    """Content types to include in context extraction."""


class ContextExtractor:
    """Extracts surrounding context from content lists for multimodal processing."""

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or ContextConfig()
        self.tokenizer = tokenizer
        self.content_source: Optional[Any] = None
        self.content_format: str = "auto"

    def set_content_source(self, content_source: Any, content_format: str = "auto"):
        """Set content source for context extraction."""
        self.content_source = content_source
        self.content_format = content_format
        logger.info(f"Content source set with format: {content_format}")

    def extract_context(
        self,
        content_source: Any,
        current_item_info: Dict[str, Any],
        content_format: str = "auto",
    ) -> str:
        """Extract context for current processing item."""
        try:
            if isinstance(content_source, list):
                return self._extract_from_content_list(content_source, current_item_info)
            elif isinstance(content_source, dict):
                return self._extract_from_dict_source(content_source, current_item_info)
            elif isinstance(content_source, str):
                return self._extract_from_text_source(content_source, current_item_info)
            else:
                logger.warning(f"Unsupported content source type: {type(content_source)}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return ""

    def _extract_from_content_list(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context from MinerU-style content list."""
        if self.config.context_mode == "page":
            return self._extract_page_context(content_list, current_item_info)
        elif self.config.context_mode == "chunk":
            return self._extract_chunk_context(content_list, current_item_info)
        else:
            return self._extract_page_context(content_list, current_item_info)

    def _extract_page_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on page boundaries."""
        current_page = current_item_info.get("page_idx", 0)
        window_size = self.config.context_window

        start_page = max(0, current_page - window_size)
        end_page = current_page + window_size + 1

        context_texts = []

        for item in content_list:
            item_page = item.get("page_idx", 0)
            item_type = item.get("type", "")

            if (
                start_page <= item_page < end_page
                and item_type in self.config.filter_content_types
            ):
                text = self._extract_text_from_item(item)
                if text and text.strip():
                    if item_page != current_page:
                        context_texts.append(f"[Page {item_page}] {text}")
                    else:
                        context_texts.append(text)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_chunk_context(
        self, content_list: List[Dict], current_item_info: Dict
    ) -> str:
        """Extract context based on content chunks."""
        current_index = current_item_info.get("index", 0)
        window_size = self.config.context_window

        start_idx = max(0, current_index - window_size)
        end_idx = min(len(content_list), current_index + window_size + 1)

        context_texts = []

        for i in range(start_idx, end_idx):
            if i != current_index:
                item = content_list[i]
                item_type = item.get("type", "")

                if item_type in self.config.filter_content_types:
                    text = self._extract_text_from_item(item)
                    if text and text.strip():
                        context_texts.append(text)

        context = "\n".join(context_texts)
        return self._truncate_context(context)

    def _extract_text_from_item(self, item: Dict) -> str:
        """Extract text content from a content item."""
        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("text", "")
            text_level = item.get("text_level", 0)

            if self.config.include_headers and text_level > 0:
                return f"{'#' * text_level} {text}"
            return text

        elif item_type == "image" and self.config.include_captions:
            captions = item.get("image_caption", item.get("img_caption", []))
            if captions:
                if isinstance(captions, list):
                    return f"[Image: {', '.join(captions)}]"
                return f"[Image: {captions}]"

        elif item_type == "table" and self.config.include_captions:
            captions = item.get("table_caption", [])
            if captions:
                if isinstance(captions, list):
                    return f"[Table: {', '.join(captions)}]"
                return f"[Table: {captions}]"

        return ""

    def _extract_from_dict_source(
        self, dict_source: Dict, current_item_info: Dict
    ) -> str:
        """Extract context from dictionary-based content source."""
        if "content" in dict_source:
            return self._truncate_context(str(dict_source["content"]))
        elif "text" in dict_source:
            return self._truncate_context(str(dict_source["text"]))
        else:
            text_parts = []
            for value in dict_source.values():
                if isinstance(value, str):
                    text_parts.append(value)
            return self._truncate_context("\n".join(text_parts))

    def _extract_from_text_source(
        self, text_source: str, current_item_info: Dict
    ) -> str:
        """Extract context from plain text source."""
        return self._truncate_context(text_source)

    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum token limit."""
        if not context:
            return ""

        if self.tokenizer:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= self.config.max_context_tokens:
                return context

            truncated_tokens = tokens[: self.config.max_context_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)

            last_period = truncated_text.rfind(".")
            last_newline = truncated_text.rfind("\n")

            if last_period > len(truncated_text) * 0.8:
                return truncated_text[: last_period + 1]
            elif last_newline > len(truncated_text) * 0.8:
                return truncated_text[:last_newline]
            else:
                return truncated_text + "..."
        else:
            if len(context) <= self.config.max_context_tokens:
                return context

            truncated = context[: self.config.max_context_tokens]
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")

            if last_period > len(truncated) * 0.8:
                return truncated[: last_period + 1]
            elif last_newline > len(truncated) * 0.8:
                return truncated[:last_newline]
            else:
                return truncated + "..."