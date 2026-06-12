# -*- coding: utf-8 -*-
"""
Context Manager — industrial-grade conversation context handling.

Solves the "50-turn degradation" problem:
  - Agent forgets early instructions
  - Repeats already-given answers
  - Expanding context window makes things worse (more noise, higher latency)

Root Causes & Solutions:
  ┌──────────────────────────────┬──────────────────────────────────────────────┐
  │ Root Cause                   │ Solution                                     │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Single-pass flat summary     │ Hierarchical Summary Compression:            │
  │ (quality degrades at 50+     │   chunk → chunk_summary → session_summary    │
  │ turns)                       │   Each level is progressively more compact   │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ System instruction mixed     │ Instruction Persistence Layer:               │
  │ with context, gets diluted   │   system instruction is NEVER compressed     │
  │ as context grows             │   context layers are injected separately     │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Fixed token budget           │ Adaptive Budget Allocator:                   │
  │ (4096 reserve regardless     │   budget scales with conversation depth      │
  │ of conversation depth)       │   deep conversations → more compression      │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ No dedup — LLM repeats       │ Dedup Guard:                                 │
  │ previously given answers     │   detect semantic overlap with history        │
  │                              │   inject anti-repetition instruction          │
  └──────────────────────────────┴──────────────────────────────────────────────┘

Architecture (v2 — Hierarchical + Instruction Persistence):
  ┌──────────────────────────────────────────────────────┐
  │  [1] System Instruction (NEVER compressed)           │  ← IMMUTABLE
  │      - Role definition, output format, constraints   │
  ├──────────────────────────────────────────────────────┤
  │  [2] Instruction Reinforcement                       │  ← HIGH ATTENTION
  │      - Key rules repeated from system instruction    │
  ├──────────────────────────────────────────────────────┤
  │  [3] Session Summary (hierarchical compression)      │  ← HIGH ATTENTION
  │      - Distilled from all previous turns             │
  ├──────────────────────────────────────────────────────┤
  │  [4] Memory Context (long-term user memories)        │  ← HIGH ATTENTION
  ├──────────────────────────────────────────────────────┤
  │  [5] KEY Recent History (most relevant turns)        │  ← HIGH ATTENTION
  ├──────────────────────────────────────────────────────┤
  │  [6] Older History (less critical)                   │  ← LOWER ATTENTION
  ├──────────────────────────────────────────────────────┤
  │  [7] Anti-Repetition Guard (dedup context)           │  ← HIGH ATTENTION
  ├──────────────────────────────────────────────────────┤
  │  [8] Recap of Key Points                             │  ← HIGH ATTENTION
  ├──────────────────────────────────────────────────────┤
  │  [9] Current User Message                            │  ← HIGHEST ATTENTION
  └──────────────────────────────────────────────────────┘

Hierarchical Summary Compression:
  Instead of one flat summary of dropped messages, we maintain a 3-level
  hierarchy that preserves information density regardless of conversation length:

  Level 0: Raw messages (recent N turns, kept verbatim)
  Level 1: Chunk summaries (every ~6 turns compressed into ~100 tokens)
  Level 2: Session summary (all chunk summaries compressed into ~200 tokens)

  As the conversation grows:
  - New turns → Level 0 (verbatim)
  - Old Level 0 turns → compressed into Level 1 chunk
  - Old Level 1 chunks → compressed into Level 2 session summary
  - Level 2 is capped at ~200 tokens (never grows)

  This ensures:
  - Information density stays constant (not proportional to conversation length)
  - Early instructions are preserved in session summary
  - Token budget is predictable regardless of conversation depth
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.token_counter import (
    count_message_tokens,
    count_tokens,
    trim_messages,
    get_model_context_size,
)
from infrastructure.tracing import get_tracer
from infrastructure.reranker import get_reranker, sandwich_reorder_texts

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_COMPLETION_RESERVE = 4096

# Hierarchical summary compression settings
CHUNK_SIZE_TURNS = 6          # How many turns per chunk summary
CHUNK_SUMMARY_MAX_TOKENS = 150  # Max tokens per chunk summary
SESSION_SUMMARY_MAX_TOKENS = 300  # Max tokens for the session-level summary

# Adaptive budget ratios (scale with conversation depth)
SHALLOW_HISTORY_RATIO = 0.60   # < 10 turns: 60% of context for history
DEEP_HISTORY_RATIO = 0.40      # 50+ turns: 40% for history (more compression)
RECAP_RESERVE_RATIO = 0.12     # 12% of history budget for recap
INSTRUCTION_REINFORCE_RATIO = 0.03  # 3% for instruction reinforcement

# Anti-repetition settings
DEDUP_SIMILARITY_THRESHOLD = 0.6  # Jaccard similarity threshold for dedup
DEDUP_MAX_HISTORY_ANSWERS = 10    # How many past answers to check against

# Attention guidance prompt
ATTENTION_GUIDANCE = (
    "\n\nIMPORTANT: Pay equal attention to ALL parts of the context provided, "
    "including information that appears in the middle of the conversation history. "
    "Do not overlook or forget any details mentioned earlier in the conversation."
)

# Anti-repetition instruction (injected when dedup is triggered)
ANTI_REPETITION_INSTRUCTION = (
    "\n\nCRITICAL: The user is asking a follow-up question in an ongoing conversation. "
    "Do NOT repeat information you have already provided in earlier turns. "
    "Only provide NEW information that directly addresses the user's current question. "
    "If you have already answered a similar question, reference that answer briefly "
    "and add only the new details the user is seeking."
)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ContextBudget:
    """Token budget configuration with adaptive allocation."""
    model: str = ""
    context_size: int = 0
    completion_reserve: int = DEFAULT_COMPLETION_RESERVE
    history_budget: int = 0
    # Adaptive sub-budgets (computed in __post_init__)
    instruction_budget: int = 0
    summary_budget: int = 0
    recap_budget: int = 0
    dedup_budget: int = 0

    def __post_init__(self):
        if not self.context_size:
            self.context_size = get_model_context_size(self.model)
        self.history_budget = self.context_size - self.completion_reserve


@dataclass
class ChunkSummary:
    """A compressed summary of a chunk of conversation turns."""
    chunk_id: str
    turn_range: Tuple[int, int]  # (start_turn, end_turn)
    summary_text: str
    token_count: int
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionSummary:
    """The top-level session summary, distilled from all chunk summaries."""
    summary_text: str
    token_count: int
    chunk_count: int  # How many chunks were compressed into this
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class HierarchicalSummaryStore:
    """
    Manages hierarchical summary compression for a single conversation session.

    Three levels:
      Level 0: Raw messages (recent, kept verbatim in context)
      Level 1: Chunk summaries (each covers ~6 turns)
      Level 2: Session summary (distilled from all chunk summaries)

    Thread-safe via internal lock.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._lock = threading.Lock()
        self._chunk_summaries: OrderedDict[str, ChunkSummary] = OrderedDict()
        self._session_summary: Optional[SessionSummary] = None
        self._total_turns_processed: int = 0

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunk_summaries)

    @property
    def session_summary_text(self) -> str:
        with self._lock:
            if self._session_summary:
                return self._session_summary.summary_text
            return ""

    @property
    def session_summary_tokens(self) -> int:
        with self._lock:
            if self._session_summary:
                return self._session_summary.token_count
            return 0

    def update(
        self,
        history: List[Dict[str, Any]],
    ) -> None:
        """
        Update the hierarchical summaries based on the current history.

        Called every time build_messages is invoked. Determines which new
        chunks need to be summarized and whether the session summary
        needs to be regenerated.
        """
        with self._lock:
            total_turns = len([m for m in history if m.get("role") in ("user", "assistant")])

            if total_turns <= self._total_turns_processed:
                return  # No new turns since last update

            logger.info(
                f"[HierarchicalSummary] Session {self.session_id}: "
                f"updating from {self._total_turns_processed} to {total_turns} turns"
            )

            # Determine which chunks need summarization
            # A chunk is a group of CHUNK_SIZE_TURNS turns
            total_chunks = total_turns // CHUNK_SIZE_TURNS
            existing_chunks = len(self._chunk_summaries)

            if total_chunks <= existing_chunks:
                self._total_turns_processed = total_turns
                return  # No new complete chunks

            # Generate summaries for new chunks
            for chunk_idx in range(existing_chunks, total_chunks):
                start_turn = chunk_idx * CHUNK_SIZE_TURNS
                end_turn = start_turn + CHUNK_SIZE_TURNS
                chunk_messages = history[start_turn * 2: end_turn * 2]  # *2 for user+assistant pairs

                if not chunk_messages:
                    continue

                chunk_id = f"chunk_{chunk_idx}"
                summary_text = self._generate_chunk_summary(chunk_messages, chunk_idx)
                token_count = count_tokens(summary_text)

                # Cap chunk summary size
                if token_count > CHUNK_SUMMARY_MAX_TOKENS:
                    summary_text = summary_text[:CHUNK_SUMMARY_MAX_TOKENS * 4]
                    token_count = count_tokens(summary_text)

                self._chunk_summaries[chunk_id] = ChunkSummary(
                    chunk_id=chunk_id,
                    turn_range=(start_turn, end_turn),
                    summary_text=summary_text,
                    token_count=token_count,
                )

                logger.info(
                    f"[HierarchicalSummary] Created {chunk_id}: "
                    f"turns {start_turn}-{end_turn}, "
                    f"{token_count} tokens"
                )

            # Regenerate session summary if we have 2+ chunk summaries
            if len(self._chunk_summaries) >= 2:
                self._regenerate_session_summary()

            self._total_turns_processed = total_turns

    def get_compressed_context(self, recent_turns: int = 0) -> str:
        """
        Build the compressed context string from hierarchical summaries.

        Args:
            recent_turns: Number of recent turns that will be kept verbatim.
                          Chunk summaries covering these turns are excluded.

        Returns:
            A string combining session summary + relevant chunk summaries.
        """
        with self._lock:
            parts = []

            # Level 2: Session summary (always included if available)
            if self._session_summary:
                parts.append(
                    f"[Session Overview — key points from the entire conversation]:\n"
                    f"{self._session_summary.summary_text}"
                )

            # Level 1: Chunk summaries (exclude chunks that overlap with recent turns)
            recent_chunk_start = recent_turns // CHUNK_SIZE_TURNS
            for chunk_id, chunk in self._chunk_summaries.items():
                if chunk.turn_range[0] >= recent_chunk_start * CHUNK_SIZE_TURNS:
                    continue  # This chunk is covered by verbatim recent history
                parts.append(
                    f"[Turns {chunk.turn_range[0]}-{chunk.turn_range[1]}]: "
                    f"{chunk.summary_text}"
                )

            if not parts:
                return ""

            return "\n\n".join(parts)

    def _generate_chunk_summary(self, messages: List[Dict[str, Any]], chunk_idx: int) -> str:
        """Generate a summary for a chunk of messages."""
        # Try LLM summary first
        llm_summary = self._llm_summarize_chunk(messages, chunk_idx)
        if llm_summary:
            return llm_summary

        # Fallback: heuristic extraction
        return self._heuristic_chunk_summary(messages)

    def _llm_summarize_chunk(self, messages: List[Dict[str, Any]], chunk_idx: int) -> str:
        """Use LLM to generate a structured chunk summary."""
        try:
            from infrastructure.llm_client import get_sync_llm
            client = get_sync_llm()
            if client is None:
                return ""

            # Build compact representation
            conv_text = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                conv_text.append(f"{role}: {content}")

            conv_str = "\n".join(conv_text)

            prompt = (
                "Summarize this conversation segment in 2-3 bullet points. "
                "Focus on: key decisions, conclusions, user preferences, and "
                "any instructions the user gave that must be remembered. "
                "Be extremely concise.\n\n"
                f"Conversation segment (turns {chunk_idx * CHUNK_SIZE_TURNS}-"
                f"{(chunk_idx + 1) * CHUNK_SIZE_TURNS}):\n{conv_str}"
            )

            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=CHUNK_SUMMARY_MAX_TOKENS,
            )

            summary = response.choices[0].message.content or ""
            return summary.strip()

        except Exception as e:
            logger.debug(f"[HierarchicalSummary] LLM chunk summary failed: {e}")
            return ""

    def _heuristic_chunk_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Fast heuristic summary when LLM is unavailable."""
        points = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                sentence = re.split(r'[.。！？\n]', content)[0].strip()
                if len(sentence) > 80:
                    sentence = sentence[:80] + "..."
                if sentence:
                    points.append(f"User asked: {sentence}")
            elif role == "assistant" and content:
                sentence = re.split(r'[.。！？\n]', content)[0].strip()
                if len(sentence) > 100:
                    sentence = sentence[:100] + "..."
                if sentence and len(sentence) > 10:
                    points.append(f"Answered: {sentence}")

        if not points:
            return ""
        if len(points) > 6:
            points = points[:6]
        return " | ".join(points)

    def _regenerate_session_summary(self) -> None:
        """Regenerate the Level 2 session summary from all chunk summaries."""
        if not self._chunk_summaries:
            return

        all_summaries = "\n\n".join(
            f"[Turns {c.turn_range[0]}-{c.turn_range[1]}]: {c.summary_text}"
            for c in self._chunk_summaries.values()
        )

        # Try LLM compression
        try:
            from infrastructure.llm_client import get_sync_llm
            client = get_sync_llm()
            if client is not None:
                prompt = (
                    "You are compressing multiple conversation summaries into a single "
                    "concise session overview. Focus on:\n"
                    "1. The user's core goal/intent\n"
                    "2. Key decisions and conclusions reached\n"
                    "3. Important user preferences or constraints\n"
                    "4. Any standing instructions that must be followed\n\n"
                    "Be extremely concise (max 200 tokens).\n\n"
                    f"Summaries to compress:\n{all_summaries}"
                )
                response = client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=SESSION_SUMMARY_MAX_TOKENS,
                )
                summary_text = response.choices[0].message.content or ""
                token_count = count_tokens(summary_text)

                if token_count > SESSION_SUMMARY_MAX_TOKENS:
                    summary_text = summary_text[:SESSION_SUMMARY_MAX_TOKENS * 4]
                    token_count = count_tokens(summary_text)

                self._session_summary = SessionSummary(
                    summary_text=summary_text.strip(),
                    token_count=token_count,
                    chunk_count=len(self._chunk_summaries),
                )

                logger.info(
                    f"[HierarchicalSummary] Session summary regenerated: "
                    f"{len(self._chunk_summaries)} chunks → {token_count} tokens"
                )
                return
        except Exception as e:
            logger.debug(f"[HierarchicalSummary] LLM session summary failed: {e}")

        # Fallback: concatenate chunk summaries with truncation
        combined = all_summaries
        if count_tokens(combined) > SESSION_SUMMARY_MAX_TOKENS:
            combined = combined[:SESSION_SUMMARY_MAX_TOKENS * 4]

        self._session_summary = SessionSummary(
            summary_text=combined,
            token_count=count_tokens(combined),
            chunk_count=len(self._chunk_summaries),
        )


# ============================================================================
# Dedup Guard — Detect and suppress repetitive answers
# ============================================================================

class DedupGuard:
    """
    Detects when the LLM is about to repeat information from previous answers.

    Uses Jaccard similarity on keyword sets (fast, no model needed).
    When repetition is detected, injects an anti-repetition instruction.
    """

    @staticmethod
    def _tokenize(text: str) -> set:
        """Simple whitespace + punctuation tokenizer for Jaccard similarity."""
        # Remove common stop words and punctuation
        words = re.findall(r'\w+', text.lower())
        # Simple stop word removal for better signal
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because',
            'but', 'and', 'or', 'if', 'while', 'about', 'up', 'it',
            'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这',
        }
        return {w for w in words if w not in stop_words and len(w) > 1}

    @staticmethod
    def jaccard_similarity(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    @classmethod
    def check_repetition(
        cls,
        current_query: str,
        history: List[Dict[str, Any]],
        max_history: int = DEDUP_MAX_HISTORY_ANSWERS,
    ) -> Tuple[bool, str]:
        """
        Check if the current query is semantically similar to previous queries.

        Returns:
            (is_repetitive, context_hint)
            - is_repetitive: True if the query overlaps significantly with history
            - context_hint: A string listing what was already answered (for injection)
        """
        if not history or len(history) < 2:
            return False, ""

        query_tokens = cls._tokenize(current_query)
        if not query_tokens:
            return False, ""

        # Extract recent assistant answers and their corresponding user questions
        recent_answers = []
        user_msgs = [m for m in history if m.get("role") == "user"]
        assistant_msgs = [m for m in history if m.get("role") == "assistant"]

        # Pair them up (take the last N pairs)
        pairs = []
        for i in range(min(len(user_msgs), len(assistant_msgs), max_history)):
            idx = -(i + 1)
            if abs(idx) <= len(user_msgs) and abs(idx) <= len(assistant_msgs):
                pairs.append((user_msgs[idx], assistant_msgs[idx]))

        similar_topics = []
        for user_msg, assistant_msg in pairs:
            past_query_tokens = cls._tokenize(user_msg.get("content", ""))
            similarity = cls.jaccard_similarity(query_tokens, past_query_tokens)

            if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                # Extract a brief summary of what was already answered
                answer_content = assistant_msg.get("content", "")
                brief = answer_content[:150].replace("\n", " ").strip()
                if len(brief) >= 150:
                    brief += "..."
                similar_topics.append({
                    "query": user_msg.get("content", "")[:80],
                    "answer_brief": brief,
                    "similarity": similarity,
                })

        if not similar_topics:
            return False, ""

        # Build context hint
        hint_parts = []
        for i, topic in enumerate(similar_topics[:3], 1):
            hint_parts.append(
                f"{i}. You already answered about '{topic['query']}' "
                f"(similarity: {topic['similarity']:.0%}): "
                f"{topic['answer_brief']}"
            )

        context_hint = (
            "[Previously answered similar questions — DO NOT repeat]:\n"
            + "\n".join(hint_parts)
        )

        logger.info(
            f"[DedupGuard] Repetition detected: {len(similar_topics)} similar "
            f"past queries found for current query"
        )

        return True, context_hint


# ============================================================================
# Adaptive Budget Allocator
# ============================================================================

class AdaptiveBudgetAllocator:
    """
    Dynamically allocates token budgets based on conversation depth.

    Shallow conversations (< 10 turns): More space for verbatim history
    Deep conversations (50+ turns): More compression, less verbatim

    This prevents the "expand context window → worse results" problem
    by keeping information density constant regardless of depth.
    """

    @staticmethod
    def compute_budgets(
        context_size: int,
        completion_reserve: int,
        history_turn_count: int,
        system_prompt_tokens: int,
        current_message_tokens: int,
    ) -> Dict[str, int]:
        """
        Compute adaptive token budgets for each context section.

        Returns dict with keys:
          - total_available: Total tokens available for all sections
          - instruction: Tokens for system instruction (never compressed)
          - summary: Tokens for hierarchical summaries
          - verbatim_history: Tokens for recent verbatim messages
          - recap: Tokens for recap section
          - dedup: Tokens for dedup context
          - rag: Tokens for RAG context
        """
        total_available = context_size - completion_reserve - system_prompt_tokens - current_message_tokens

        if total_available <= 0:
            return {
                "total_available": max(0, total_available),
                "instruction": 0,
                "summary": 0,
                "verbatim_history": 0,
                "recap": 0,
                "dedup": 0,
                "rag": 0,
            }

        # Compute depth factor: 0.0 (shallow) → 1.0 (deep)
        depth_factor = min(1.0, max(0.0, (history_turn_count - 5) / 45))

        # Interpolate history ratio based on depth
        history_ratio = SHALLOW_HISTORY_RATIO + depth_factor * (DEEP_HISTORY_RATIO - SHALLOW_HISTORY_RATIO)

        # Instruction reinforcement: small fixed budget
        instruction_budget = min(int(total_available * INSTRUCTION_REINFORCE_RATIO), 200)

        # Summary budget: grows with depth (more chunks = more summaries)
        summary_budget = int(total_available * (0.05 + depth_factor * 0.15))  # 5% → 20%

        # Recap budget
        recap_budget = int(total_available * RECAP_RESERVE_RATIO)

        # Dedup budget
        dedup_budget = int(total_available * 0.05) if history_turn_count > 10 else 0

        # RAG budget (if applicable)
        rag_budget = int(total_available * 0.15)

        # Verbatim history gets the rest
        fixed_budgets = instruction_budget + summary_budget + recap_budget + dedup_budget + rag_budget
        verbatim_budget = max(0, int(total_available * history_ratio) - fixed_budgets)

        logger.info(
            f"[AdaptiveBudget] depth_factor={depth_factor:.2f}, "
            f"turns={history_turn_count}, "
            f"total_available={total_available}, "
            f"instruction={instruction_budget}, "
            f"summary={summary_budget}, "
            f"verbatim={verbatim_budget}, "
            f"recap={recap_budget}, "
            f"dedup={dedup_budget}, "
            f"rag={rag_budget}"
        )

        return {
            "total_available": total_available,
            "instruction": instruction_budget,
            "summary": summary_budget,
            "verbatim_history": verbatim_budget,
            "recap": recap_budget,
            "dedup": dedup_budget,
            "rag": rag_budget,
        }


# ============================================================================
# Context Manager (v2)
# ============================================================================

class ContextManager:
    """
    Manages conversation context with industrial-grade long-conversation handling.

    Key improvements over v1:
      1. Hierarchical Summary Compression (chunk → chunk_summary → session_summary)
      2. Instruction Persistence (system instruction never compressed)
      3. Adaptive Budget Allocation (scales with conversation depth)
      4. Dedup Guard (detects and suppresses repetitive answers)
    """

    def __init__(
        self,
        model: str = "",
        completion_reserve: int = DEFAULT_COMPLETION_RESERVE,
    ):
        self._budget = ContextBudget(
            model=model,
            completion_reserve=completion_reserve,
        )
        self._summary_stores: Dict[str, HierarchicalSummaryStore] = {}
        self._lock = threading.Lock()
        self._budget_allocator = AdaptiveBudgetAllocator()
        self._dedup_guard = DedupGuard()

    @property
    def budget(self) -> ContextBudget:
        return self._budget

    def get_summary_store(self, session_id: str) -> HierarchicalSummaryStore:
        """Get or create a HierarchicalSummaryStore for a session."""
        with self._lock:
            if session_id not in self._summary_stores:
                self._summary_stores[session_id] = HierarchicalSummaryStore(session_id)
            return self._summary_stores[session_id]

    def build_messages(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        current_message: str,
        *,
        memory_context: str = "",
        model: Optional[str] = None,
        rag_context: str = "",
        session_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build the final messages array with hierarchical compression and
        Lost-in-the-Middle mitigation.

        v2 Architecture:
          [1] System Instruction (NEVER compressed)
          [2] Instruction Reinforcement (key rules repeated)
          [3] Session Summary (hierarchical compression)
          [4] Memory Context
          [5] KEY Recent History (verbatim)
          [6] Older History (less critical)
          [7] Anti-Repetition Guard (dedup context)
          [8] Recap of Key Points
          [9] Current User Message

        Args:
            system_prompt: The base system prompt for this mode.
            history: Full conversation history (list of {role, content} dicts).
            current_message: The current user message.
            memory_context: Retrieved long-term memory context.
            model: Override model for token counting.
            rag_context: RAG-retrieved context to inject.
            session_id: Session ID for hierarchical summary tracking.

        Returns:
            (messages, meta) where meta contains budget info for tracing.
        """
        tracer = get_tracer()
        use_model = model or self._budget.model
        budget = ContextBudget(
            model=use_model,
            completion_reserve=self._budget.completion_reserve,
        )

        history_turn_count = len([m for m in history if m.get("role") in ("user", "assistant")])

        logger.info(
            f"[ContextManager:v2] build_messages called: "
            f"history={len(history)} msgs ({history_turn_count} turns), "
            f"model={use_model}, session={session_id}, "
            f"has_memory={bool(memory_context)}, has_rag={bool(rag_context)}, "
            f"current_msg={current_message[:80]!r}..."
        )

        # ====================================================================
        # Phase 1: Compute adaptive budgets
        # ====================================================================
        system_tokens_estimate = count_tokens(system_prompt) + 10
        current_tokens = count_tokens(current_message) + 4

        budgets = self._budget_allocator.compute_budgets(
            context_size=budget.context_size,
            completion_reserve=budget.completion_reserve,
            history_turn_count=history_turn_count,
            system_prompt_tokens=system_tokens_estimate,
            current_message_tokens=current_tokens,
        )

        # ====================================================================
        # Phase 2: Update hierarchical summaries
        # ====================================================================
        summary_store = None
        compressed_context = ""
        if session_id:
            summary_store = self.get_summary_store(session_id)
            summary_store.update(history)
            # Get compressed context, excluding recent turns that will be verbatim
            recent_verbatim_turns = max(4, budgets["verbatim_history"] // 200)  # rough estimate
            compressed_context = summary_store.get_compressed_context(recent_verbatim_turns)

            logger.info(
                f"[ContextManager:v2] Hierarchical summary: "
                f"chunks={summary_store.chunk_count}, "
                f"session_summary_tokens={summary_store.session_summary_tokens}, "
                f"compressed_context_len={len(compressed_context)}"
            )

        # ====================================================================
        # Phase 3: Build system instruction (NEVER compressed)
        # ====================================================================
        messages: List[Dict[str, Any]] = []

        # [1] System Instruction — immutable, never compressed
        messages.append({"role": "system", "content": system_prompt})

        # [2] Instruction Reinforcement — repeat key rules at the end of system section
        reinforcement = self._build_instruction_reinforcement(system_prompt)
        if reinforcement:
            messages.append({"role": "system", "content": reinforcement})

        # [3] Session Summary (hierarchical compression)
        if compressed_context:
            # Trim to budget
            if count_tokens(compressed_context) > budgets["summary"]:
                compressed_context = compressed_context[:budgets["summary"] * 4]
            messages.append({
                "role": "system",
                "content": f"[Conversation History Summary]:\n{compressed_context}",
            })

        # [4] Memory Context
        if memory_context:
            memory_section = (
                f"[Relevant User Memories]:\n{memory_context}\n"
                "Use the above memories to personalize your response when relevant."
            )
            messages.append({"role": "system", "content": memory_section})

        # Add attention guidance
        messages.append({"role": "system", "content": ATTENTION_GUIDANCE.strip()})

        # ====================================================================
        # Phase 4: Inject RAG context
        # ====================================================================
        if rag_context:
            rag_messages = self._build_rag_context_messages(rag_context)
            rag_tokens = count_message_tokens(rag_messages)

            if rag_tokens < budgets["rag"]:
                messages.extend(rag_messages)
            else:
                rag_messages = self._sandwich_rag_context(
                    rag_context, budgets["rag"], query=current_message,
                )
                messages.extend(rag_messages)

        # ====================================================================
        # Phase 5: Fit verbatim history with sandwich structure
        # ====================================================================
        system_tokens = count_message_tokens(messages)
        remaining = budgets["total_available"] - system_tokens

        if remaining <= 0:
            logger.warning("[ContextManager:v2] No budget left for history after system messages")
            messages.append({"role": "user", "content": current_message})
            return messages, self._build_meta(budget, system_tokens, 0, current_tokens, False, False, 0)

        # Fit history into remaining budget
        history_messages, was_compressed, recap_text = self._fit_history_adaptive(
            history=history,
            token_budget=remaining,
            recap_budget=budgets["recap"],
            summary_store=summary_store,
        )

        # Split into key (beginning) and older (middle) sections
        key_history, older_history = self._split_history_for_sandwich(
            history_messages, current_query=current_message,
        )

        messages.extend(key_history)
        messages.extend(older_history)

        # ====================================================================
        # Phase 6: Anti-repetition guard
        # ====================================================================
        is_repetitive, dedup_hint = self._dedup_guard.check_repetition(
            current_message, history,
        )
        if is_repetitive and dedup_hint:
            messages.append({"role": "system", "content": dedup_hint})
            messages.append({"role": "system", "content": ANTI_REPETITION_INSTRUCTION.strip()})
            logger.info("[ContextManager:v2] Anti-repetition guard activated")

        # ====================================================================
        # Phase 7: Recap + Current message
        # ====================================================================
        if recap_text:
            messages.append({"role": "system", "content": recap_text})

        messages.append({"role": "user", "content": current_message})

        # ====================================================================
        # Phase 8: Final safety trim
        # ====================================================================
        messages = trim_messages(
            messages,
            model=use_model,
            completion_reserve=budget.completion_reserve,
            keep_system=True,
        )

        history_tokens = count_message_tokens(key_history) + count_message_tokens(older_history)
        meta = self._build_meta(
            budget, system_tokens, history_tokens, current_tokens,
            was_compressed, bool(recap_text), history_turn_count,
        )

        # Add v2-specific meta
        meta["hierarchical_chunks"] = summary_store.chunk_count if summary_store else 0
        meta["session_summary_tokens"] = summary_store.session_summary_tokens if summary_store else 0
        meta["dedup_triggered"] = is_repetitive
        meta["depth_factor"] = min(1.0, max(0.0, (history_turn_count - 5) / 45))

        with tracer.span("context.build", **meta) as span:
            pass

        return messages, meta

    def build_rag_messages(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        current_message: str,
        *,
        memory_context: str = "",
        model: Optional[str] = None,
        rag_context: str = "",
        session_id: str = "",
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Build context for RAG mode."""
        llm_messages, meta = self.build_messages(
            system_prompt=system_prompt,
            history=history,
            current_message=current_message,
            memory_context=memory_context,
            model=model,
            rag_context=rag_context,
            session_id=session_id,
        )
        return current_message, llm_messages, meta

    # ========================================================================
    # Instruction Persistence
    # ========================================================================

    def _build_instruction_reinforcement(self, system_prompt: str) -> str:
        """
        Extract and reinforce key instructions from the system prompt.

        This ensures that even in very long conversations (50+ turns),
        the LLM still follows the core instructions. The reinforcement
        is placed right after the system prompt (high attention position).
        """
        # Extract key constraints/instructions from the system prompt
        # Look for sentences containing imperative keywords
        imperative_patterns = [
            r'(?:MUST|ALWAYS|NEVER|DO NOT|IMPORTANT|CRITICAL|REQUIRED)[^.]*\.',
            r'(?:必须|始终|绝不|不要|重要|关键|务必)[^。]*。',
        ]

        key_rules = []
        for pattern in imperative_patterns:
            matches = re.findall(pattern, system_prompt, re.IGNORECASE)
            key_rules.extend(matches)

        if not key_rules:
            return ""

        # Deduplicate and limit
        seen = set()
        unique_rules = []
        for rule in key_rules:
            normalized = rule.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_rules.append(rule.strip())

        if len(unique_rules) > 5:
            unique_rules = unique_rules[:5]

        reinforcement = (
            "[CRITICAL INSTRUCTIONS — must follow these rules in every response]:\n"
            + "\n".join(f"- {rule}" for rule in unique_rules)
        )

        logger.debug(
            f"[ContextManager] Instruction reinforcement: "
            f"{len(unique_rules)} rules extracted"
        )

        return reinforcement

    # ========================================================================
    # Adaptive History Fitting
    # ========================================================================

    def _fit_history_adaptive(
        self,
        history: List[Dict[str, Any]],
        token_budget: int,
        recap_budget: int,
        summary_store: Optional[HierarchicalSummaryStore] = None,
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Fit history into budget using adaptive strategy.

        If we have a summary_store with hierarchical summaries, we can
        skip the oldest messages (they're already in the summaries).
        This is the key insight: instead of trimming from the front and
        generating a flat summary, we use the pre-built hierarchical
        summaries and only keep recent verbatim messages.

        Returns:
            (kept_messages, was_compressed, recap_text)
        """
        if not history:
            return [], False, ""

        total = count_message_tokens(history)

        logger.info(
            f"[ContextManager:FitAdaptive] Fitting {len(history)} msgs "
            f"({total} tokens) into budget={token_budget}, "
            f"recap_budget={recap_budget}, "
            f"has_summary_store={summary_store is not None}"
        )

        # If we have hierarchical summaries, we can skip messages that
        # are already covered by chunk summaries
        if summary_store and summary_store.chunk_count > 0:
            # Calculate how many recent turns to keep verbatim
            # Older turns are already in chunk summaries
            covered_turns = summary_store.chunk_count * CHUNK_SIZE_TURNS
            covered_messages = covered_turns * 2  # user + assistant pairs

            if covered_messages < len(history):
                # Only keep messages NOT covered by chunk summaries
                recent_history = history[covered_messages:]
                logger.info(
                    f"[ContextManager:FitAdaptive] Using hierarchical summaries: "
                    f"skipping {covered_messages} msgs (covered by {summary_store.chunk_count} chunks), "
                    f"keeping {len(recent_history)} recent msgs"
                )
            else:
                recent_history = history
        else:
            recent_history = history

        # Check if recent history fits
        recent_total = count_message_tokens(recent_history)
        main_budget = token_budget - recap_budget

        if recent_total <= main_budget:
            # Everything fits
            recap_text = ""
            if len(recent_history) >= 6:
                recap_text = self._extract_recap(recent_history)
                recap_tokens = count_tokens(recap_text)
                if recap_tokens > recap_budget:
                    recap_text = recap_text[:recap_budget * 4]
            return list(recent_history), len(recent_history) < len(history), recap_text

        # Need to trim — keep most recent messages that fit
        kept = []
        used = 0
        for msg in reversed(recent_history):
            msg_tokens = count_message_tokens([msg])
            if used + msg_tokens > main_budget:
                break
            kept.append(msg)
            used += msg_tokens

        kept.reverse()
        was_compressed = len(kept) < len(history)

        logger.info(
            f"[ContextManager:FitAdaptive] Trimmed: "
            f"{len(recent_history)} → {len(kept)} msgs kept, "
            f"was_compressed={was_compressed}"
        )

        # Generate recap from kept messages
        recap_text = ""
        if len(kept) >= 4:
            recap_text = self._extract_recap(kept)
            recap_tokens = count_tokens(recap_text)
            if recap_tokens > recap_budget:
                recap_text = recap_text[:recap_budget * 4]

        return kept, was_compressed, recap_text

    # ========================================================================
    # Lost-in-the-Middle: Sandwich Reordering
    # ========================================================================

    def _split_history_for_sandwich(
        self,
        history_messages: List[Dict[str, Any]],
        current_query: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split history into key (beginning) and older (middle) sections."""
        if len(history_messages) <= 4:
            return history_messages, []

        if current_query:
            try:
                reranker = get_reranker()
                if reranker is not None:
                    return self._relevance_split(history_messages, current_query, reranker)
            except Exception as e:
                logger.debug(f"[ContextManager:Split] Relevance split failed: {e}")

        # Positional heuristic
        split_point = max(2, len(history_messages) * 2 // 3)
        return history_messages[:split_point], history_messages[split_point:]

    def _relevance_split(
        self,
        history_messages: List[Dict[str, Any]],
        query: str,
        reranker,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split by relevance score."""
        chunks = []
        for msg in history_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            chunks.append(f"{role}: {content[:200]}")

        scored = reranker.rerank(query, chunks)
        mid = len(scored) // 2
        key_indices = {s.index for s in scored[:mid]}
        older_indices = {s.index for s in scored[mid:]}

        key_history = [history_messages[i] for i in sorted(key_indices) if i < len(history_messages)]
        older_history = [history_messages[i] for i in sorted(older_indices) if i < len(history_messages)]

        logger.info(
            f"[ContextManager:RelevanceSplit] {len(history_messages)} msgs: "
            f"key={len(key_history)}, older={len(older_history)}"
        )

        return key_history, older_history

    # ========================================================================
    # Recap Extraction
    # ========================================================================

    def _extract_recap(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key points for the recap section (before current message)."""
        # Try LLM summary first
        llm_summary = self._llm_summarize_messages(messages)
        if llm_summary:
            return (
                f"[Key points from earlier in this conversation — pay attention to these]:\n"
                f"{llm_summary}"
            )

        # Heuristic fallback
        key_points = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                sentence = re.split(r'[.。！？\n]', content)[0].strip()
                if len(sentence) > 120:
                    sentence = sentence[:120] + "..."
                if sentence:
                    key_points.append(f"User asked: {sentence}")
            elif role == "assistant" and content:
                sentence = re.split(r'[.。！？\n]', content)[0].strip()
                if len(sentence) > 150:
                    sentence = sentence[:150] + "..."
                if sentence and len(sentence) > 10:
                    key_points.append(f"Assistant: {sentence}")

        if not key_points:
            return ""

        if len(key_points) > 8:
            key_points = key_points[:8]

        recap = "\n".join(f"- {p}" for p in key_points)
        return (
            f"[Key points from earlier in this conversation — pay attention to these]:\n"
            f"{recap}"
        )

    def _llm_summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Use LLM to generate a structured summary."""
        if len(messages) < 4:
            return ""

        try:
            from infrastructure.llm_client import get_sync_llm
            client = get_sync_llm()
            if client is None:
                return ""

            conversation_text = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                conversation_text.append(f"{role}: {content}")

            conv_str = "\n".join(conversation_text)

            summary_prompt = (
                "Summarize the key points of this conversation in 3-5 bullet points. "
                "Focus on: what the user asked, what was decided/concluded, and any "
                "important context that should be remembered. Be concise.\n\n"
                f"Conversation:\n{conv_str}"
            )

            response = client.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0,
                max_tokens=256,
            )

            summary = response.choices[0].message.content or ""
            if len(summary) > 500:
                summary = summary[:500] + "..."
            return summary.strip()

        except Exception as e:
            logger.debug(f"[ContextManager:LLMSummary] Failed: {e}")
            return ""

    # ========================================================================
    # RAG Context Handling
    # ========================================================================

    def _build_rag_context_messages(self, rag_context: str) -> List[Dict[str, Any]]:
        """Build messages for RAG context injection."""
        if not rag_context:
            return []
        return [
            {"role": "user", "content": f"[Retrieved knowledge base context]:\n\n{rag_context}"},
            {"role": "assistant", "content": "I've reviewed the retrieved context and will use it to answer."},
        ]

    def _sandwich_rag_context(
        self,
        rag_context: str,
        token_budget: int,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Apply reranking + sandwich reordering to RAG context."""
        chunks = re.split(r'\n---\n|\n##\s', rag_context)
        chunks = [c.strip() for c in chunks if c.strip()]

        if len(chunks) <= 2:
            return self._build_rag_context_messages(rag_context[:token_budget * 4])

        reranker = get_reranker()
        if query and reranker is not None:
            reordered_texts = sandwich_reorder_texts(query, chunks, reranker)
        else:
            reordered_texts = [chunks[0]]
            if len(chunks) > 1:
                reordered_texts.append(chunks[-1])
            reordered_texts.extend(chunks[1:-1])

        sandwiched = "\n\n---\n\n".join(reordered_texts)
        if count_tokens(sandwiched) > token_budget:
            sandwiched = sandwiched[:token_budget * 4]

        return self._build_rag_context_messages(sandwiched)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _trim_to_budget(
        self,
        messages: List[Dict[str, Any]],
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Aggressively trim messages to fit budget."""
        if not messages:
            return messages
        kept = []
        used = 0
        for msg in reversed(messages):
            msg_tokens = count_message_tokens([msg])
            if used + msg_tokens > token_budget:
                break
            kept.append(msg)
            used += msg_tokens
        kept.reverse()
        return kept

    def _build_meta(
        self,
        budget: ContextBudget,
        system_tokens: int,
        history_tokens: int,
        current_tokens: int,
        was_compressed: bool,
        has_recap: bool,
        turn_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "model": budget.model,
            "context_size": budget.context_size,
            "history_budget": budget.history_budget,
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "current_tokens": current_tokens,
            "total_tokens": system_tokens + history_tokens + current_tokens,
            "was_compressed": was_compressed,
            "has_recap": has_recap,
            "turn_count": turn_count,
        }

    def invalidate_summary(self, session_id: str):
        """Clear cached summary store for a session."""
        with self._lock:
            self._summary_stores.pop(session_id, None)


# ============================================================================
# Singleton
# ============================================================================

_context_manager: Optional[ContextManager] = None


def get_context_manager(model: str = "") -> ContextManager:
    """Get or create the global ContextManager."""
    global _context_manager
    if _context_manager is None or (model and _context_manager._budget.model != model):
        _context_manager = ContextManager(model=model)
    return _context_manager
