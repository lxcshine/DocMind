"""
DocMind Backend - Document Upload Handler

PageIndex-style document processing pipeline:
  1. Upload -> Save (store file permanently)
  2. Add to Knowledge Base -> Parse -> Build Tree Index -> Generate Summaries -> Store JSON
"""

import os
import uuid
import json
import logging
import threading
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import base64

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from core.parser import PageIndexParser
from core.tree_index import tree_index_store
from core.progress import progress_tracker

logger = logging.getLogger(__name__)


class DocumentUploadHandler:
    """
    Handles document upload and processing.

    Step 1: Upload file -> Save to disk permanently
    Step 2: User clicks "Add to KB" -> PageIndex tree index pipeline
    """

    def __init__(
        self,
        upload_dir: str = "./uploads",
        model: str = "gemini-2.5-flash",
        api_key: str = None,
        base_url: str = None,
        meta_store=None,
    ):
        self.upload_dir = Path(upload_dir)
        os.makedirs(self.upload_dir, exist_ok=True)
        self.meta_store = meta_store
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.parser = PageIndexParser(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

        logger.info("DocumentUploadHandler initialized (vectorless mode)")

    def save_file(self, file_content: bytes, filename: str, doc_id: str = None) -> Dict:
        """Step 1: Save uploaded file to disk permanently."""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        file_ext = Path(filename).suffix.lower()
        save_filename = f"{doc_id}_{filename}"
        save_path = self.upload_dir / save_filename

        with open(save_path, "wb") as f:
            f.write(file_content)

        file_size = len(file_content)

        progress_tracker.create(doc_id, filename)
        progress_tracker.update(doc_id, status="uploaded", progress=100, current_step="File saved")

        return {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": str(save_path),
            "file_size": file_size,
            "file_type": file_ext.lstrip('.').upper(),
            "status": "uploaded",
        }

    def process_document(self, doc_id: str, filename: str = None) -> Dict:
        """Step 2: Process a saved document through PageIndex tree index pipeline."""
        progress = progress_tracker.get(doc_id)
        if not progress:
            return {"error": "Document not found", "doc_id": doc_id}

        if progress.status == "processing":
            return {"error": "Document is already being processed", "doc_id": doc_id}

        progress_tracker.update(doc_id, status="processing", progress=0, current_step="Starting...")

        thread = threading.Thread(
            target=self._process_document_thread,
            args=(doc_id, filename),
            daemon=True,
        )
        thread.start()

        return {"doc_id": doc_id, "status": "processing"}

    def _process_document_thread(self, doc_id: str, filename: str = None):
        """Background thread for document processing."""
        try:
            progress = progress_tracker.get(doc_id)
            if not progress:
                return

            if not filename:
                filename = progress.filename

            file_path = self.upload_dir / f"{doc_id}_{filename}"
            if not file_path.exists():
                progress_tracker.update(doc_id, error="File not found")
                return

            file_ext = Path(filename).suffix.lower()

            if file_ext == '.pdf':
                self._process_pdf_tree(doc_id, filename, str(file_path))
            elif file_ext in ['.txt', '.md']:
                self._process_text_tree(doc_id, filename, str(file_path))
            else:
                progress_tracker.update(doc_id, error=f"Unsupported file type: {file_ext}")

        except Exception as e:
            logger.error(f"Processing failed for {doc_id}: {e}")
            progress_tracker.update(doc_id, error=str(e))

    def _process_pdf_tree(self, doc_id: str, filename: str, file_path: str):
        logger.info(f"[PDF Tree] Starting processing for {doc_id}")
        total_steps = 7
        progress_tracker.set_step(doc_id, 1, total_steps, "Parsing PDF and extracting pages...")

        parse_result = self.parser.parse(file_path)
        pages = parse_result["pages"]
        flat_structure = parse_result["structure"]
        logger.info(f"[PDF Tree] Parsed {len(pages)} pages for {doc_id}")

        progress_tracker.set_step(doc_id, 2, total_steps, f"Parsed {len(pages)} pages, building tree structure...")

        tree = self._build_tree_from_flat(flat_structure, len(pages))
        logger.info(f"[PDF Tree] Tree built for {doc_id}")

        progress_tracker.set_step(doc_id, 3, total_steps, "Generating section summaries...")

        self._add_node_ids(tree)
        self._add_node_text(tree, pages)
        self._generate_summaries(tree)
        logger.info(f"[PDF Tree] Summaries generated for {doc_id}")

        doc_description = self._generate_doc_description(tree, pages)

        progress_tracker.set_step(doc_id, 4, total_steps, "Building page content index...")

        page_content_list = []
        for page_num, page_text in pages:
            page_content_list.append({"page": page_num, "content": page_text})

        progress_tracker.set_step(doc_id, 5, total_steps, "Rendering PDF pages to images...")

        image_paths = []
        if HAS_PYMUPDF:
            try:
                images_dir = tree_index_store.index_dir / f"{doc_id}_images"
                images_dir.mkdir(parents=True, exist_ok=True)
                pdf_doc = pymupdf.open(file_path)
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    mat = pymupdf.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_path = images_dir / f"{page_num + 1}.jpg"
                    pix.save(str(img_path))
                    image_paths.append(str(img_path))
                pdf_doc.close()
                logger.info(f"[PDF Tree] Rendered {len(image_paths)} page images for {doc_id}")
            except Exception as e:
                logger.warning(f"[PDF Tree] Failed to render page images: {e}")

        doc_index = {
            "doc_id": doc_id,
            "doc_name": filename,
            "doc_description": doc_description,
            "type": "pdf",
            "path": file_path,
            "page_count": len(pages),
            "structure": tree,
            "pages": page_content_list,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        node_count = self._count_nodes(tree)

        progress_tracker.set_step(doc_id, 6, total_steps, "Saving tree index to disk...")
        logger.info(f"[PDF Tree] About to call save_document for {doc_id}")

        tree_index_store.save_document(doc_index)
        logger.info(f"[PDF Tree] save_document completed for {doc_id}")

        progress_tracker.update(
            doc_id,
            status="completed",
            progress=100,
            current_step="Tree index built and stored",
            sections_count=node_count,
        )
        logger.info(f"[PDF Tree] Progress updated to 100% for {doc_id}")

        if self.meta_store:
            self.meta_store.update_document(doc_id, status="completed", sections_count=node_count)
            logger.info(f"[PDF Tree] Meta store updated for {doc_id}")

        logger.info(f"PDF {doc_id} processed: {len(pages)} pages, {node_count} tree nodes")

    def _process_text_tree(self, doc_id: str, filename: str, file_path: str):
        total_steps = 4
        progress_tracker.set_step(doc_id, 1, total_steps, "Reading text file...")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        progress_tracker.set_step(doc_id, 2, total_steps, "Building structure...")

        lines = content.split('\n')
        tree = self._build_text_tree(lines)
        self._add_node_ids(tree)

        doc_description = self._generate_text_description(content)

        line_content_list = [
            {"page": i + 1, "content": lines[i]}
            for i in range(min(len(lines), 1000))
        ]

        doc_index = {
            "doc_id": doc_id,
            "doc_name": filename,
            "doc_description": doc_description,
            "type": "md",
            "path": file_path,
            "page_count": len(line_content_list),
            "structure": tree,
            "pages": line_content_list,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        node_count = self._count_nodes(tree)

        progress_tracker.set_step(doc_id, 3, total_steps, "Saving tree index to disk...")

        tree_index_store.save_document(doc_index)

        progress_tracker.update(
            doc_id,
            status="completed",
            progress=100,
            current_step="Tree index built and stored",
            sections_count=node_count,
        )

        if self.meta_store:
            self.meta_store.update_document(doc_id, status="completed", sections_count=node_count)

        logger.info(f"Text {doc_id} processed: {node_count} tree nodes")

    def _build_tree_from_flat(self, flat_structure: List[Dict], total_pages: int) -> List[Dict]:
        """Build a hierarchical tree from a flat structure list.

        Preserves nested hierarchy from PDF bookmarks when available.
        Each entry may have a `nodes` field with pre-built children.
        """
        if not flat_structure:
            return [{
                "title": "Document",
                "structure": "1",
                "level": 1,
                "start_index": 1,
                "end_index": total_pages,
                "nodes": [],
            }]

        tree = []

        for entry in flat_structure:
            structure = entry.get("structure", "1")
            title = entry.get("title", "")

            page_from = entry.get("page_from")
            if page_from is None:
                page_from = entry.get("physical_index", 1)
            if page_from is None:
                page_from = 1

            page_to = entry.get("page_to")
            if page_to is None:
                page_to = total_pages

            level = len(structure.split('.')) if structure else 1

            children = entry.get("nodes")
            child_nodes = []
            if children and isinstance(children, list) and len(children) > 0:
                child_nodes = self._build_tree_from_flat(children, total_pages)

            node = {
                "title": title,
                "structure": structure,
                "level": level,
                "start_index": page_from,
                "end_index": page_to,
                "nodes": child_nodes,
            }

            tree.append(node)

        self._calculate_end_indices(tree, total_pages)

        return tree

    def _calculate_end_indices(self, nodes: List[Dict], total_pages: int):
        """Calculate end_index for each node based on children/siblings."""
        for i, node in enumerate(nodes):
            if node.get("nodes"):
                self._calculate_end_indices(node["nodes"], total_pages)
                last_child = node["nodes"][-1]
                node["end_index"] = last_child.get("end_index", total_pages)

            if i + 1 < len(nodes):
                node["end_index"] = min(
                    node.get("end_index", total_pages),
                    nodes[i + 1].get("start_index", total_pages) - 1
                )

    def _add_node_ids(self, tree: List[Dict], counter: int = 0) -> int:
        """Add sequential node IDs to tree nodes."""
        for node in tree:
            node["node_id"] = str(counter).zfill(4)
            counter += 1
            if node.get("nodes"):
                counter = self._add_node_ids(node["nodes"], counter)
        return counter

    def _add_node_text(self, tree: List[Dict], pages: List[tuple]):
        """Add text content to each node from page data."""
        for node in tree:
            start = node.get("start_index", 1)
            end = node.get("end_index", 1)

            text_parts = []
            for page_num, page_text in pages:
                if start <= page_num <= end:
                    text_parts.append(page_text)

            node["text"] = "\n\n".join(text_parts)

            if node.get("nodes"):
                self._add_node_text(node["nodes"], pages)

    def _generate_summaries(self, tree: List[Dict]):
        """Generate summaries for tree nodes using LLM."""
        for node in tree:
            if node.get("nodes"):
                self._generate_summaries(node["nodes"])

            text = node.get("text", "")
            if not text.strip():
                node["summary"] = f"# {node.get('title', 'Section')}"
                continue

            summary = self._generate_node_summary(node["title"], text[:3000])
            node["summary"] = summary

            if "text" in node:
                del node["text"]

    def _generate_node_summary(self, title: str, text: str) -> str:
        """Generate a summary for a single tree node."""
        prompt = f"""Summarize the following section of a document in 2-3 sentences.

Section title: {title}

Content:
{text}

Return only the summary text, nothing else."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed for '{title}': {e}")
            return f"# {title}"

    def _generate_doc_description(self, tree: List[Dict], pages: List[tuple]) -> str:
        """Generate a document-level description using LLM."""
        titles = []
        for node in tree[:5]:
            titles.append(node.get("title", ""))
            if node.get("nodes"):
                for child in node["nodes"][:3]:
                    titles.append(f"  - {child.get('title', '')}")

        first_page_text = ""
        if pages:
            first_page_text = pages[0][1][:1000] if len(pages[0]) > 1 else ""

        prompt = f"""Describe this document in one sentence, mentioning what it covers.

Document sections:
{chr(10).join(titles)}

First page excerpt:
{first_page_text[:500]}

Return only the description, nothing else."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Doc description generation failed: {e}")
            return f"A document with {len(pages)} pages covering {len(tree)} main sections."

    def _generate_text_description(self, content: str) -> str:
        """Generate a description for a text/markdown document."""
        prompt = f"""Describe this document in one sentence.

First lines:
{content[:500]}

Return only the description, nothing else."""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Text description generation failed: {e}")
            return "A text document."

    def _build_text_tree(self, lines: List[str]) -> List[Dict]:
        """Build a simple tree from text lines using heading patterns."""
        import re
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        tree = []
        stack = []

        for i, line in enumerate(lines[:200]):
            m = heading_pattern.match(line.strip())
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()

                node = {
                    "title": title,
                    "structure": str(len(tree) + 1) if level == 1 else "",
                    "level": level,
                    "start_index": i + 1,
                    "end_index": len(lines),
                    "nodes": [],
                }

                while stack and stack[-1]["level"] >= level:
                    stack.pop()

                if stack:
                    node["structure"] = f"{stack[-1].get('structure', '1')}.{len(stack[-1].get('nodes', [])) + 1}"
                    stack[-1].setdefault("nodes", []).append(node)
                else:
                    node["structure"] = str(len(tree) + 1)
                    tree.append(node)

                stack.append(node)

        if not tree:
            tree = [{
                "title": "Document",
                "structure": "1",
                "level": 1,
                "start_index": 1,
                "end_index": len(lines),
                "nodes": [],
            }]

        self._calculate_end_indices(tree, len(lines))
        return tree

    @staticmethod
    def _count_nodes(tree: List[Dict]) -> int:
        count = 0
        for node in tree:
            count += 1
            if node.get("nodes"):
                count += DocumentUploadHandler._count_nodes(node["nodes"])
        return count

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document, its file, and its tree index."""
        try:
            tree_index_store.delete_document(doc_id)

            progress = progress_tracker.get(doc_id)
            if progress:
                filename = progress.filename
                file_path = self.upload_dir / f"{doc_id}_{filename}"
                if file_path.exists():
                    os.remove(file_path)

            progress_tracker.delete(doc_id)

            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_document_stats(self) -> Dict:
        """Get document statistics."""
        return tree_index_store.get_stats()
