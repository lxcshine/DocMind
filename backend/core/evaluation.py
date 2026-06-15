# -*- coding: utf-8 -*-
"""
RAG Evaluation Harness — 质量评估框架

工业级 RAG 系统必须有量化的评估指标，用于：
  - 检测回归（prompt 变更后自动对比 baseline）
  - 模型切换 A/B 测试
  - 检索质量监控
  - CI/CD 质量门禁

评估指标（基于 RAGAS 框架）：
  1. Faithfulness — 回答是否忠于检索内容
  2. Answer Relevancy — 回答是否切题
  3. Context Precision — 检索结果中相关内容的排名
  4. Context Recall — 是否检索到了所有需要的信息
  5. Answer Correctness — 答案是否正确

Usage:
    from core.evaluation import EvaluationHarness
    
    harness = EvaluationHarness()
    report = await harness.evaluate_dataset("tests/evaluation/golden_qa.json")
    print(report.summary())
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from infrastructure.llm_client import get_sync_llm
from infrastructure.token_counter import count_tokens

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EvaluationCase:
    """单个评估用例"""
    id: str
    category: str  # factual / analytical / comparative
    question: str
    expected_answer: str
    context_documents: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """单个用例的评估结果"""
    case_id: str
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_context: str
    
    # RAGAS 指标
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    
    # 性能指标
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    # 元数据
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """评估报告"""
    dataset_name: str
    total_cases: int
    completed_cases: int
    failed_cases: int
    
    # 平均指标
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_correctness: float = 0.0
    
    # 性能指标
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0
    
    # 详细结果
    results: List[EvaluationResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """生成摘要报告"""
        return f"""
=== RAG Evaluation Report ===
Dataset: {self.dataset_name}
Cases: {self.completed_cases}/{self.total_cases} completed, {self.failed_cases} failed

Quality Metrics (0-1 scale):
  Faithfulness:        {self.avg_faithfulness:.3f}
  Answer Relevancy:    {self.avg_answer_relevancy:.3f}
  Context Precision:   {self.avg_context_precision:.3f}
  Context Recall:      {self.avg_context_recall:.3f}
  Answer Correctness:  {self.avg_answer_correctness:.3f}

Performance:
  Avg Latency:         {self.avg_latency_ms:.1f} ms
  P95 Latency:         {self.p95_latency_ms:.1f} ms
  Total Tokens:        {self.total_tokens:,}

Quality Gate: {'PASS' if self.passes_quality_gate() else 'FAIL'}
"""
    
    def passes_quality_gate(self) -> bool:
        """检查是否通过质量门禁"""
        return (
            self.avg_faithfulness >= 0.85
            and self.avg_answer_relevancy >= 0.80
            and self.avg_context_precision >= 0.75
            and self.avg_context_recall >= 0.80
            and self.avg_answer_correctness >= 0.75
            and self.p95_latency_ms <= 5000
        )


# ============================================================================
# 评估引擎
# ============================================================================

class EvaluationHarness:
    """
    RAG 质量评估引擎
    
    支持：
      - 批量评估数据集
      - 自动计算 RAGAS 指标
      - 生成 HTML/JSON 报告
      - 质量门禁检查
    """
    
    def __init__(self):
        self.llm = get_sync_llm()
    
    async def evaluate_dataset(
        self,
        dataset_path: str,
        max_concurrent: int = 3,
    ) -> EvaluationReport:
        """
        评估整个数据集
        
        Args:
            dataset_path: JSON 文件路径（golden_qa.json 格式）
            max_concurrent: 最大并发评估数
            
        Returns:
            EvaluationReport 对象
        """
        # 加载数据集
        dataset = self._load_dataset(dataset_path)
        dataset_name = Path(dataset_path).stem
        
        logger.info(f"[Evaluation] Starting evaluation: {len(dataset)} cases")
        
        # 并发评估
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(case: EvaluationCase) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_case(case)
        
        tasks = [evaluate_with_semaphore(case) for case in dataset]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[Evaluation] Case {dataset[i].id} failed: {result}")
                valid_results.append(EvaluationResult(
                    case_id=dataset[i].id,
                    question=dataset[i].question,
                    expected_answer=dataset[i].expected_answer,
                    actual_answer="",
                    retrieved_context="",
                    error=str(result),
                ))
            else:
                valid_results.append(result)
        
        # 生成报告
        report = self._generate_report(dataset_name, valid_results)
        
        logger.info(f"[Evaluation] Complete:\n{report.summary()}")
        
        return report
    
    async def _evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        """评估单个用例"""
        start_time = time.time()
        
        try:
            # 1. 执行 RAG 查询
            from core.raganything import get_rag_instance, query as rag_query
            
            rag = get_rag_instance()
            if rag is None:
                raise RuntimeError("RAG not initialized")
            
            # 执行查询（mix 模式，包含检索 + 生成）
            actual_answer = await rag_query(rag, case.question, mode="mix")
            
            # 2. 获取检索到的上下文（用于评估检索质量）
            retrieved_context = await rag_query(rag, case.question, mode="naive")
            
            # 3. 计算 RAGAS 指标
            faithfulness = await self._compute_faithfulness(actual_answer, retrieved_context)
            answer_relevancy = await self._compute_answer_relevancy(case.question, actual_answer)
            context_precision = await self._compute_context_precision(case.question, retrieved_context, case.expected_answer)
            context_recall = await self._compute_context_recall(retrieved_context, case.expected_answer)
            answer_correctness = await self._compute_answer_correctness(actual_answer, case.expected_answer)
            
            latency_ms = (time.time() - start_time) * 1000
            tokens_used = count_tokens(actual_answer) + count_tokens(retrieved_context)
            
            return EvaluationResult(
                case_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer=actual_answer,
                retrieved_context=retrieved_context,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                answer_correctness=answer_correctness,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
            
        except Exception as e:
            logger.error(f"[Evaluation] Case {case.id} failed: {e}", exc_info=True)
            return EvaluationResult(
                case_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer="",
                retrieved_context="",
                error=str(e),
            )
    
    # ========================================================================
    # RAGAS 指标计算
    # ========================================================================
    
    async def _compute_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: 回答是否忠于检索内容
        
        方法：让 LLM 判断回答中的每个声明是否能在 context 中找到支撑
        """
        prompt = f"""
You are an expert evaluator. Assess whether the answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Instructions:
1. Identify all factual claims in the answer
2. For each claim, check if it can be inferred from the context
3. Return a score from 0.0 to 1.0:
   - 1.0: All claims are supported by context
   - 0.5: Some claims are supported, some are not
   - 0.0: Most claims are NOT supported by context

Return ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Faithfulness computation failed: {e}")
            return 0.0
    
    async def _compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy: 回答是否切题
        
        方法：从回答反推问题，计算与原问题的相似度
        """
        prompt = f"""
You are an expert evaluator. Assess how relevant the answer is to the question.

Question: {question}

Answer: {answer}

Instructions:
1. Read the question carefully
2. Check if the answer directly addresses the question
3. Return a score from 0.0 to 1.0:
   - 1.0: Answer directly and completely addresses the question
   - 0.5: Answer partially addresses the question
   - 0.0: Answer is irrelevant to the question

Return ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Answer relevancy computation failed: {e}")
            return 0.0
    
    async def _compute_context_precision(
        self,
        question: str,
        context: str,
        expected_answer: str,
    ) -> float:
        """
        Context Precision: 检索结果中相关内容的排名
        
        方法：判断 context 中是否包含回答问题所需的信息
        """
        prompt = f"""
You are an expert evaluator. Assess the precision of the retrieved context.

Question: {question}

Expected Answer: {expected_answer}

Retrieved Context:
{context}

Instructions:
1. Check if the context contains information needed to answer the question
2. Return a score from 0.0 to 1.0:
   - 1.0: Context contains all necessary information
   - 0.5: Context contains some necessary information
   - 0.0: Context lacks necessary information

Return ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Context precision computation failed: {e}")
            return 0.0
    
    async def _compute_context_recall(self, context: str, expected_answer: str) -> float:
        """
        Context Recall: 是否检索到了所有需要的信息
        
        方法：判断 expected_answer 中的信息是否都在 context 中
        """
        prompt = f"""
You are an expert evaluator. Assess the recall of the retrieved context.

Expected Answer: {expected_answer}

Retrieved Context:
{context}

Instructions:
1. Identify key information in the expected answer
2. Check if that information is present in the context
3. Return a score from 0.0 to 1.0:
   - 1.0: All key information is in the context
   - 0.5: Some key information is in the context
   - 0.0: Most key information is missing from the context

Return ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Context recall computation failed: {e}")
            return 0.0
    
    async def _compute_answer_correctness(self, actual_answer: str, expected_answer: str) -> float:
        """
        Answer Correctness: 答案是否正确
        
        方法：语义相似度 + 事实重叠度
        """
        prompt = f"""
You are an expert evaluator. Assess the correctness of the answer.

Expected Answer: {expected_answer}

Actual Answer: {actual_answer}

Instructions:
1. Compare the actual answer to the expected answer
2. Check for semantic equivalence (not exact wording)
3. Return a score from 0.0 to 1.0:
   - 1.0: Answers are semantically equivalent
   - 0.5: Answers are partially correct
   - 0.0: Answers are incorrect or contradictory

Return ONLY a number between 0.0 and 1.0, nothing else.
"""
        
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Answer correctness computation failed: {e}")
            return 0.0
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _load_dataset(self, dataset_path: str) -> List[EvaluationCase]:
        """加载评估数据集"""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        cases = []
        for item in data.get("cases", []):
            cases.append(EvaluationCase(
                id=item["id"],
                category=item.get("category", "factual"),
                question=item["question"],
                expected_answer=item["expected_answer"],
                context_documents=item.get("context_documents", []),
                metadata=item.get("metadata", {}),
            ))
        
        return cases
    
    def _generate_report(
        self,
        dataset_name: str,
        results: List[EvaluationResult],
    ) -> EvaluationReport:
        """生成评估报告"""
        completed = [r for r in results if not r.error]
        failed = [r for r in results if r.error]
        
        if not completed:
            return EvaluationReport(
                dataset_name=dataset_name,
                total_cases=len(results),
                completed_cases=0,
                failed_cases=len(failed),
            )
        
        # 计算平均值
        avg_faithfulness = np.mean([r.faithfulness for r in completed])
        avg_answer_relevancy = np.mean([r.answer_relevancy for r in completed])
        avg_context_precision = np.mean([r.context_precision for r in completed])
        avg_context_recall = np.mean([r.context_recall for r in completed])
        avg_answer_correctness = np.mean([r.answer_correctness for r in completed])
        
        avg_latency = np.mean([r.latency_ms for r in completed])
        p95_latency = np.percentile([r.latency_ms for r in completed], 95)
        total_tokens = sum(r.tokens_used for r in completed)
        
        return EvaluationReport(
            dataset_name=dataset_name,
            total_cases=len(results),
            completed_cases=len(completed),
            failed_cases=len(failed),
            avg_faithfulness=float(avg_faithfulness),
            avg_answer_relevancy=float(avg_answer_relevancy),
            avg_context_precision=float(avg_context_precision),
            avg_context_recall=float(avg_context_recall),
            avg_answer_correctness=float(avg_answer_correctness),
            avg_latency_ms=float(avg_latency),
            p95_latency_ms=float(p95_latency),
            total_tokens=int(total_tokens),
            results=results,
        )
    
    def export_report(self, report: EvaluationReport, output_path: str) -> None:
        """导出报告为 JSON"""
        data = {
            "dataset_name": report.dataset_name,
            "total_cases": report.total_cases,
            "completed_cases": report.completed_cases,
            "failed_cases": report.failed_cases,
            "metrics": {
                "avg_faithfulness": report.avg_faithfulness,
                "avg_answer_relevancy": report.avg_answer_relevancy,
                "avg_context_precision": report.avg_context_precision,
                "avg_context_recall": report.avg_context_recall,
                "avg_answer_correctness": report.avg_answer_correctness,
            },
            "performance": {
                "avg_latency_ms": report.avg_latency_ms,
                "p95_latency_ms": report.p95_latency_ms,
                "total_tokens": report.total_tokens,
            },
            "quality_gate": report.passes_quality_gate(),
            "results": [
                {
                    "case_id": r.case_id,
                    "question": r.question,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "answer_correctness": r.answer_correctness,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[Evaluation] Report exported to {output_path}")


# ============================================================================
# 单例
# ============================================================================

_harness: Optional[EvaluationHarness] = None


def get_evaluation_harness() -> EvaluationHarness:
    """获取全局评估引擎实例"""
    global _harness
    if _harness is None:
        _harness = EvaluationHarness()
    return _harness
