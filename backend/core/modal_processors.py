"""
Multimodal content processors for image, table, and equation analysis

Based on RAG-Anything's modalprocessors.py, adapted for DocMind.
Provides specialized processors for different content types with
context-aware analysis and entity extraction.
"""

from __future__ import annotations

import json
import base64
import hashlib
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from pathlib import Path
from dataclasses import asdict

from openai import OpenAI

from core.prompt_templates import PROMPTS
from core.context_extractor import ContextExtractor, ContextConfig

logger = logging.getLogger(__name__)


def compute_mdhash_id(content: str, prefix: str = "md-") -> str:
    """Generate a deterministic ID from content."""
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def normalize_caption_list(value: Any) -> List[str]:
    """Return captions and footnotes as a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def get_table_body(item: Dict[str, Any]) -> str:
    """Read table content across common content-list alias fields."""
    if item.get("table_body") not in (None, ""):
        body = item.get("table_body")
    elif item.get("table_data") not in (None, ""):
        body = item.get("table_data")
    else:
        body = item.get("text", "")

    if isinstance(body, list):
        if not body:
            return ""
        if all(isinstance(row, (list, tuple)) for row in body):
            rendered_rows = [
                "| " + " | ".join(str(cell) for cell in row) + " |"
                for row in body
            ]
            if len(rendered_rows) >= 1:
                column_count = max(len(row) for row in body)
                separator = "| " + " | ".join(["---"] * column_count) + " |"
                rendered_rows.insert(1, separator)
            return "\n".join(rendered_rows)
        return "\n".join(str(row) for row in body)
    return str(body)


def get_equation_text_and_format(item: Dict[str, Any]) -> Tuple[str, str]:
    """Read equation content while preserving LaTeX aliases."""
    text = str(item.get("text", "") or "").strip()
    latex = str(item.get("latex", "") or "").strip()
    equation = str(item.get("equation", "") or "").strip()
    equation_format = str(item.get("text_format", "") or "").strip()

    if text:
        equation_text = text
    elif latex:
        equation_text = latex
        if not equation_format:
            equation_format = "latex"
    elif equation:
        equation_text = equation
    else:
        equation_text = ""

    return equation_text, equation_format


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """Validate if a file is a valid image file."""
    try:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"]
        if not any(str(path).lower().endswith(ext) for ext in image_extensions):
            return False

        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


class BaseModalProcessor:
    """Base class for modal processors."""

    def __init__(
        self,
        lightrag: Any,
        modal_caption_func: Optional[Callable] = None,
        context_extractor: Optional[ContextExtractor] = None,
    ):
        self.lightrag = lightrag
        self.modal_caption_func = modal_caption_func
        self.context_extractor = context_extractor or ContextExtractor()
        self.content_source: Optional[Any] = None
        self.content_format: str = "auto"

    def set_content_source(self, content_source: Any, content_format: str = "auto"):
        """Set content source for context extraction."""
        self.content_source = content_source
        self.content_format = content_format
        self.context_extractor.set_content_source(content_source, content_format)

    def _get_context_for_item(self, item_info: Dict[str, Any]) -> str:
        """Get context for current processing item."""
        if not self.content_source:
            return ""
        try:
            return self.context_extractor.extract_context(
                self.content_source, item_info, self.content_format
            )
        except Exception as e:
            logger.error(f"Error getting context for item {item_info}: {e}")
            return ""

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Call LLM for text-based analysis."""
        if self.modal_caption_func is None:
            raise ValueError("No LLM function available for text analysis")

        try:
            result = await self.modal_caption_func(
                system_prompt, user_prompt, response_format=response_format
            )
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def generate_description_only(
        self,
        modal_content: Dict[str, Any],
        content_type: str,
        item_info: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text description and entity info only."""
        raise NotImplementedError("Subclasses must implement this method")


class ImageModalProcessor(BaseModalProcessor):
    """Processor for image content analysis using vision model."""

    async def generate_description_only(
        self,
        modal_content: Dict[str, Any],
        content_type: str,
        item_info: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        image_path = modal_content.get("img_path", "") or modal_content.get("image_path", "")
        captions = normalize_caption_list(
            modal_content.get("image_caption") or modal_content.get("img_caption", [])
        )
        footnotes = normalize_caption_list(modal_content.get("image_footnote", []))
        section_path = modal_content.get("_section_path", "")

        if not entity_name:
            entity_name = f"image_{Path(image_path).stem}" if image_path else "image_entity"

        context = ""
        if item_info:
            context = self._get_context_for_item(item_info)

        if context:
            prompt_template = PROMPTS["vision_prompt_with_context"]
        else:
            prompt_template = PROMPTS["vision_prompt"]

        prompt = prompt_template.format(
            entity_name=entity_name,
            section_path=section_path,
            image_path=image_path,
            captions=", ".join(captions) if captions else "None",
            footnotes=", ".join(footnotes) if footnotes else "None",
            context=context,
        )

        try:
            if self.modal_caption_func is None:
                raise ValueError("No vision model function available")

            result = await self.modal_caption_func(
                PROMPTS["IMAGE_ANALYSIS_SYSTEM"], prompt
            )
            parsed = self._parse_json_response(result, entity_name, "image")
            return parsed.get("detailed_description", ""), parsed.get("entity_info", {})
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return f"Image at {image_path}: {'; '.join(captions)}", {
                "entity_name": entity_name,
                "entity_type": "image",
                "summary": "; ".join(captions) if captions else "No description available",
            }

    def _parse_json_response(
        self, response: str, entity_name: str, entity_type: str
    ) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {
                "detailed_description": response,
                "entity_info": {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "summary": response[:200],
                },
            }


class TableModalProcessor(BaseModalProcessor):
    """Processor for table content analysis."""

    async def generate_description_only(
        self,
        modal_content: Dict[str, Any],
        content_type: str,
        item_info: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        table_img_path = modal_content.get("img_path", "") or modal_content.get("image_path", "")
        table_caption = ", ".join(normalize_caption_list(modal_content.get("table_caption", [])))
        table_body = get_table_body(modal_content)
        table_footnote = ", ".join(normalize_caption_list(modal_content.get("table_footnote", [])))

        if not entity_name:
            entity_name = f"table_{Path(table_img_path).stem}" if table_img_path else "table_entity"

        context = ""
        if item_info:
            context = self._get_context_for_item(item_info)

        if context:
            prompt = PROMPTS["table_prompt_with_context"].format(
                entity_name=entity_name,
                table_img_path=table_img_path,
                table_caption=table_caption or "None",
                table_body=table_body,
                table_footnote=table_footnote or "None",
                context=context,
            )
        else:
            prompt = PROMPTS["table_prompt"].format(
                entity_name=entity_name,
                table_img_path=table_img_path,
                table_caption=table_caption or "None",
                table_body=table_body,
                table_footnote=table_footnote or "None",
            )

        try:
            result = await self._call_llm(PROMPTS["TABLE_ANALYSIS_SYSTEM"], prompt)
            parsed = self._parse_json_response(result, entity_name, "table")
            return parsed.get("detailed_description", ""), parsed.get("entity_info", {})
        except Exception as e:
            logger.error(f"Table analysis failed: {e}")
            return f"Table: {table_caption}\n{table_body[:500]}", {
                "entity_name": entity_name,
                "entity_type": "table",
                "summary": table_caption or "Table analysis unavailable",
            }

    def _parse_json_response(
        self, response: str, entity_name: str, entity_type: str
    ) -> Dict[str, Any]:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {
                "detailed_description": response,
                "entity_info": {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "summary": response[:200],
                },
            }


class EquationModalProcessor(BaseModalProcessor):
    """Processor for equation content analysis."""

    async def generate_description_only(
        self,
        modal_content: Dict[str, Any],
        content_type: str,
        item_info: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        equation_text, equation_format = get_equation_text_and_format(modal_content)

        if not entity_name:
            entity_name = f"equation_{compute_mdhash_id(equation_text, '')[:8]}"

        context = ""
        if item_info:
            context = self._get_context_for_item(item_info)

        if context:
            prompt = PROMPTS["equation_prompt_with_context"].format(
                entity_name=entity_name,
                equation_text=equation_text,
                equation_format=equation_format or "latex",
                context=context,
            )
        else:
            prompt = PROMPTS["equation_prompt"].format(
                entity_name=entity_name,
                equation_text=equation_text,
                equation_format=equation_format or "latex",
            )

        try:
            result = await self._call_llm(PROMPTS["EQUATION_ANALYSIS_SYSTEM"], prompt)
            parsed = self._parse_json_response(result, entity_name, "equation")
            return parsed.get("detailed_description", ""), parsed.get("entity_info", {})
        except Exception as e:
            logger.error(f"Equation analysis failed: {e}")
            return f"Equation: {equation_text}", {
                "entity_name": entity_name,
                "entity_type": "equation",
                "summary": f"Equation: {equation_text[:100]}",
            }

    def _parse_json_response(
        self, response: str, entity_name: str, entity_type: str
    ) -> Dict[str, Any]:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {
                "detailed_description": response,
                "entity_info": {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "summary": response[:200],
                },
            }


class GenericModalProcessor(BaseModalProcessor):
    """Generic processor for fallback content types."""

    async def generate_description_only(
        self,
        modal_content: Dict[str, Any],
        content_type: str,
        item_info: Optional[Dict[str, Any]] = None,
        entity_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        content = str(modal_content)

        if not entity_name:
            entity_name = f"{content_type}_{compute_mdhash_id(content, '')[:8]}"

        prompt = PROMPTS["generic_prompt"].format(
            content_type=content_type,
            entity_name=entity_name,
            content=content[:2000],
        )

        try:
            result = await self._call_llm(
                PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(content_type=content_type),
                prompt,
            )
            parsed = self._parse_json_response(result, entity_name, content_type)
            return parsed.get("detailed_description", ""), parsed.get("entity_info", {})
        except Exception as e:
            logger.error(f"Generic analysis failed for {content_type}: {e}")
            return f"{content_type}: {content[:500]}", {
                "entity_name": entity_name,
                "entity_type": content_type,
                "summary": content[:200],
            }

    def _parse_json_response(
        self, response: str, entity_name: str, entity_type: str
    ) -> Dict[str, Any]:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {
                "detailed_description": response,
                "entity_info": {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "summary": response[:200],
                },
            }


def get_processor_for_type(
    modal_processors: Dict[str, BaseModalProcessor], content_type: str
) -> Optional[BaseModalProcessor]:
    """Get appropriate processor based on content type."""
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        return modal_processors.get("generic")