"""
ResearchFlow Backend - Document Parser

Implements PageIndex document parsing strategy:
1. PDF text extraction using PyMuPDF/PyPDF2
2. Page-level extraction with physical index markers
3. LLM-driven TOC detection and extraction
4. Hierarchical structure building
5. Page index verification and repair
"""

import os
import re
import json
import logging
import tiktoken
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

from openai import OpenAI

logger = logging.getLogger(__name__)


class PageIndexParser:
    """
    PDF parser based on PageIndex architecture.
    
    Pipeline:
      PDF -> Text Extraction -> Page Indexing -> TOC Detection 
           -> TOC Extraction -> Structure Building -> Page Index Verification
    """
    
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        self.model = model or "glm-4-flash"
        self.api_key = api_key
        self.base_url = base_url
        self.enable_toc = True
        self.enable_verification = True
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"PageIndexParser initialized: model={model}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
    
    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text from PDF pages."""
        pages = []
        
        if HAS_PYMUPDF:
            pages = self._extract_with_pymupdf(pdf_path)
        elif HAS_PYPDF2:
            pages = self._extract_with_pypdf2(pdf_path)
        else:
            raise ImportError("Neither PyMuPDF nor PyPDF2 is available")
        
        logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
        return pages
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract pages using PyMuPDF."""
        doc = pymupdf.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append((page_num + 1, text))
        
        doc.close()
        return pages
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract pages using PyPDF2."""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                pages.append((page_num + 1, text))
        
        return pages
    
    def build_page_list(self, pages: List[Tuple[int, str]]) -> List[Tuple[str, int]]:
        """Build page list with physical index markers."""
        page_list = []
        
        for page_num, text in pages:
            marked_text = f"<physical_index_{page_num}>\n{text}\n<physical_index_{page_num}>\n\n"
            token_count = self._count_tokens(marked_text)
            page_list.append((marked_text, token_count))
        
        return page_list
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """Complete PDF parsing pipeline."""
        logger.info(f"Starting PageIndex parsing: {pdf_path}")

        pages = self.extract_pages(pdf_path)

        page_list = self.build_page_list(pages)

        structure = self._extract_structure_from_bookmarks(pdf_path, len(pages))
        if len(structure) <= 1:
            logger.info("PDF bookmarks insufficient, trying LLM-based structure extraction")
            self._try_llm_structure(page_list, pages, structure)

        toc_result = None
        if self.enable_toc and len(structure) <= 1:
            toc_result = self._extract_toc(page_list, pages)
            if toc_result:
                try:
                    llm_structure = self._build_structure(page_list, pages, toc_result)
                    if len(llm_structure) > 1:
                        structure = llm_structure
                        logger.info(f"LLM TOC extraction produced {len(structure)} sections")
                except Exception as e:
                    logger.warning(f"LLM TOC extraction failed: {e}")
            elif len(structure) <= 1:
                try:
                    llm_structure = self._process_no_toc(page_list)
                    if len(llm_structure) > 1:
                        structure = llm_structure
                        logger.info(f"LLM no-TOC extraction produced {len(llm_structure)} sections")
                except Exception as e:
                    logger.warning(f"LLM no-TOC extraction failed: {e}")

        result = {
            "pages": pages,
            "page_list": page_list,
            "toc": toc_result,
            "structure": structure,
            "page_count": len(pages),
        }
        
        logger.info(f"PageIndex parsing completed: {len(pages)} pages, "
                   f"TOC={'yes' if toc_result else 'no'}, "
                   f"structure items: {len(structure)}")
        
        return result
    
    def _extract_structure_from_bookmarks(self, pdf_path: str, total_pages: int) -> List[Dict]:
        """Extract document structure from PDF bookmarks/outline.

        Zero LLM dependency. Uses PyMuPDF's built-in outline support.
        Returns list of {structure, title, page_from, page_to, physical_index, nodes}.
        """
        try:
            import fitz

            doc = fitz.open(pdf_path)
            toc = doc.get_toc(simple=False)

            if not toc:
                logger.info(f"No PDF bookmarks found in {pdf_path}")
                doc.close()
                return [{"structure": "1", "title": "Document", "physical_index": 1}]

            logger.info(f"Found {len(toc)} PDF bookmark entries")

            section_counter = [0, 0, 0, 0, 0, 0]
            prev_depth = 0
            result = []

            level_stack = [[]] * 7
            page_ranges = {}

            for entry in toc:
                level = min(entry[0], 6)
                title = entry[1].strip()
                page_num = max(1, entry[2])

                if not title:
                    title = f"Section {entry[2]}"

                if level > prev_depth + 1:
                    level = prev_depth + 1

                if level <= prev_depth:
                    for i in range(level, len(section_counter)):
                        section_counter[i] = 0

                section_counter[level - 1] += 1

                parts = []
                for i in range(level):
                    parts.append(str(section_counter[i]))
                structure = ".".join(parts)

                section_entry = {
                    "structure": structure,
                    "title": title,
                    "page_from": page_num,
                    "page_to": page_num,
                    "physical_index": page_num,
                    "nodes": [],
                }
                result.append(section_entry)

                if level == 1:
                    level_stack[0].append(section_entry)
                    level_stack[1] = section_entry["nodes"]
                    level_stack[2] = []
                    level_stack[3] = []
                elif level == 2:
                    if level_stack[1] is not None:
                        level_stack[1].append(section_entry)
                    level_stack[2] = section_entry["nodes"]
                    level_stack[3] = []
                else:
                    parent_nodes = level_stack[level - 1]
                    if parent_nodes is not None and isinstance(parent_nodes, list):
                        parent_nodes.append(section_entry)
                    level_stack[level] = section_entry["nodes"]

                prev_depth = level

            def _assign_page_ranges(nodes, parent_end):
                if not isinstance(nodes, list):
                    return
                for i, node in enumerate(nodes):
                    if i + 1 < len(nodes):
                        node["page_to"] = max(node.get("page_from", node.get("physical_index", 1)),
                                             nodes[i + 1].get("page_from", nodes[i + 1].get("physical_index", 1)) - 1)
                    else:
                        node["page_to"] = parent_end
                    if node.get("nodes"):
                        _assign_page_ranges(node["nodes"], node["page_to"])

            top_level = [e for e in result if len(e["structure"].split(".")) == 1]
            if not top_level and result:
                top_level = [result[0]]
                result[0]["structure"] = "1"

            for i in range(len(top_level) - 1):
                top_level[i]["page_to"] = max(top_level[i].get("page_from", top_level[i].get("physical_index", 1)),
                                             top_level[i + 1].get("page_from", top_level[i + 1].get("physical_index", 1)) - 1)
            if top_level:
                top_level[-1]["page_to"] = total_pages
                for entry in top_level:
                    if entry.get("nodes"):
                        _assign_page_ranges(entry["nodes"], entry["page_to"])

            if len(top_level) == 1 and not top_level[0].get("nodes"):
                doc.close()
                logger.info("PDF bookmarks exist but only single top-level entry, using as fallback structure")
                return top_level

            doc.close()
            logger.info(f"Successfully extracted {len(top_level)} top-level sections from PDF bookmarks")
            return top_level

        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for bookmark extraction")
            return [{"structure": "1", "title": "Document", "physical_index": 1}]
        except Exception as e:
            logger.error(f"Bookmark extraction failed: {e}")
            return [{"structure": "1", "title": "Document", "physical_index": 1}]

    def _try_llm_structure(self, page_list, pages, fallback_structure):
        """Attempt LLM-based structure extraction, keeping fallback on failure."""
        pass

    def _extract_toc(self, page_list: List[Tuple[str, int]], 
                     pages: List[Tuple[int, str]]) -> Optional[Dict]:
        """Extract table of contents."""
        try:
            toc_page_list = self._find_toc_pages(page_list)
            
            if not toc_page_list:
                return None
            
            toc_content = ""
            for page_index in toc_page_list:
                toc_content += page_list[page_index][0]
            
            has_page_index = self._detect_page_index(toc_content)
            
            return {
                "toc_content": toc_content,
                "toc_page_list": toc_page_list,
                "page_index_given_in_toc": has_page_index
            }
        except Exception as e:
            logger.error(f"TOC extraction failed: {e}")
            return None
    
    def _find_toc_pages(self, page_list: List[Tuple[str, int]], max_pages: int = 20) -> List[int]:
        """Find pages containing table of contents."""
        toc_page_list = []
        last_page_is_yes = False
        
        for i in range(min(len(page_list), max_pages)):
            detected = self._toc_detector_single_page(page_list[i][0])
            
            if detected == 'yes':
                toc_page_list.append(i)
                last_page_is_yes = True
            elif detected == 'no' and last_page_is_yes:
                break
        
        return toc_page_list
    
    def _toc_detector_single_page(self, content: str) -> str:
        """Detect if a page contains table of contents."""
        prompt = f"""
Your job is to detect if there is a table of content provided in the given text.

Given text: {content[:3000]}

return the following JSON format:
{{
    "thinking": "<reasoning>",
    "toc_detected": "<yes or no>",
}}

Directly return the final JSON structure. Do not output anything else.
Please note: abstract, summary, notation list, figure list, table list, etc. are not table of contents."""

        try:
            response = self._llm_completion(prompt)
            json_result = self._extract_json(response)
            return json_result.get('toc_detected', 'no')
        except Exception as e:
            logger.warning(f"TOC detection failed: {e}")
            return 'no'
    
    def _detect_page_index(self, toc_content: str) -> str:
        """Detect if TOC contains page numbers."""
        prompt = f"""
You will be given a table of contents.

Your job is to detect if there are page numbers/indices given within the table of contents.

Given text: {toc_content[:3000]}

Reply format:
{{
    "thinking": "<reasoning>",
    "page_index_given_in_toc": "<yes or no>"
}}

Directly return the final JSON structure. Do not output anything else."""

        try:
            response = self._llm_completion(prompt)
            json_result = self._extract_json(response)
            return json_result.get('page_index_given_in_toc', 'no')
        except Exception as e:
            logger.warning(f"Page index detection failed: {e}")
            return 'no'
    
    def _build_structure(self, page_list: List[Tuple[str, int]], 
                        pages: List[Tuple[int, str]],
                        toc_result: Optional[Dict]) -> List[Dict]:
        """Build hierarchical document structure."""
        try:
            if toc_result and toc_result.get('toc_content'):
                structure = self._process_toc(toc_result, page_list)
            else:
                structure = self._process_no_toc(page_list)
            
            return structure
        except Exception as e:
            logger.error(f"Structure building failed: {e}")
            return [{"structure": "1", "title": "Document", "physical_index": 1}]
    
    def _process_toc(self, toc_result: Dict, page_list: List[Tuple[str, int]]) -> List[Dict]:
        """Process document with TOC."""
        toc_content = toc_result['toc_content']
        
        prompt = f"""
You are given a table of contents. Your job is to transform it into a JSON format.

structure is the numeric system which represents the index of the hierarchy section.
For example: 1, 1.1, 1.2, 2, 2.1, etc.

The response should be in the following JSON format: 
[
    {{
        "structure": "<structure index, e.g., '1' or '1.1'>",
        "title": "<title of the section>",
        "page": <page number or null>
    }},
    ...
]

Table of contents:
{toc_content[:5000]}

Directly return the final JSON structure, do not output anything else."""

        try:
            response = self._llm_completion(prompt)
            json_result = self._extract_json(response)
            
            if isinstance(json_result, dict) and 'table_of_contents' in json_result:
                json_result = json_result['table_of_contents']
            
            return json_result if isinstance(json_result, list) else []
        except Exception as e:
            logger.error(f"TOC processing failed: {e}")
            return []
    
    def _process_no_toc(self, page_list: List[Tuple[str, int]]) -> List[Dict]:
        """Process document without TOC."""
        page_contents = [page[0] for page in page_list[:10]]
        combined_text = "\n".join(page_contents)
        
        prompt = f"""
You are given the first pages of a document. Your job is to extract the main sections and their structure.

The response should be in the following JSON format: 
[
    {{
        "structure": "<structure index, e.g., '1' or '1.1'>",
        "title": "<title of the section>",
        "physical_index": <page number>
    }},
    ...
]

Document text:
{combined_text[:8000]}

Directly return the final JSON structure, do not output anything else."""

        try:
            response = self._llm_completion(prompt)
            json_result = self._extract_json(response)
            return json_result if isinstance(json_result, list) else []
        except Exception as e:
            logger.error(f"No-TOC processing failed: {e}")
            return []
    
    def _llm_completion(self, prompt: str) -> str:
        """Call LLM API for completion."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    
    def _extract_json(self, content: str) -> Dict:
        """Extract JSON from LLM response with robust error recovery."""
        try:
            start_idx = content.find("```json")
            if start_idx != -1:
                start_idx += 7
                end_idx = content.rfind("```")
                json_content = content[start_idx:end_idx].strip()
            else:
                json_content = content.strip()

            json_content = json_content.replace('None', 'null')
            json_content = json_content.replace('\n', ' ').replace('\r', ' ')
            json_content = ' '.join(json_content.split())

            return json.loads(json_content)
        except Exception as e:
            logger.warning(f"JSON parse failed, attempting repair: {e}")
            logger.warning(f"Broken JSON (first 1000 chars): {json_content[:1000]}")
            try:
                repaired = self._repair_json(json_content)
                return json.loads(repaired)
            except Exception as e2:
                logger.error(f"JSON repair also failed: {e2}")
                return {}

    def _repair_json(self, text: str) -> str:
        """Attempt to repair common JSON formatting issues from LLM output."""
        result = []
        i = 0
        in_string = False
        escape_next = False
        last_char = ""

        while i < len(text):
            ch = text[i]

            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue

            if ch == '\\' and in_string:
                result.append(ch)
                escape_next = True
                i += 1
                continue

            if ch == '"':
                in_string = not in_string
                result.append(ch)
                last_char = ch
                i += 1
                continue

            if not in_string:
                if ch == "'":
                    result.append('"')
                    last_char = '"'
                    i += 1
                    continue

                if ch == '{' or ch == ',':
                    j = i + 1
                    while j < len(text) and text[j] in ' \t':
                        j += 1
                    if j < len(text) and text[j] not in '"{}[],' and text[j] not in '0123456789tfn-':
                        result.append(ch)
                        result.append('"')
                        last_char = ch
                        i += 1
                        continue

                if ch in '0123456789tfn-' and last_char in '{,':
                    j = i
                    while j < len(text) and text[j] not in ':,}':
                        j += 1
                    key = text[i:j].strip()
                    if key and not key.startswith('"'):
                        result.append('"')
                        result.append(key)
                        result.append('"')
                        i = j
                        continue

                if ch == ',' and i + 1 < len(text):
                    j = i + 1
                    while j < len(text) and text[j] in ' \t':
                        j += 1
                    if j < len(text) and text[j] in '}]':
                        i += 1
                        continue

            result.append(ch)
            last_char = ch
            i += 1

        repaired = ''.join(result)
        logger.warning(f"Repaired JSON (first 1000 chars): {repaired[:1000]}")
        return repaired


def parse_pdf(pdf_path: str, model: str = None, api_key: str = None, base_url: str = None) -> Dict:
    """Convenience function to parse PDF."""
    parser = PageIndexParser(model=model, api_key=api_key, base_url=base_url)
    return parser.parse(pdf_path)
