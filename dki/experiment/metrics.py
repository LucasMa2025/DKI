"""
Metrics Calculator for DKI Experiments
Computes evaluation metrics for RAG vs DKI comparison
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class MetricsCalculator:
    """
    Calculate evaluation metrics for DKI experiments.
    
    Metrics:
    - Memory Recall: How often relevant memories are used
    - Hallucination Rate: False information in responses
    - BLEU/ROUGE: Text quality metrics
    - Latency: Response time measurements
    """
    
    def __init__(self):
        self._nltk_initialized = False
        self._rouge_scorer = None
    
    def _init_nltk(self):
        """Initialize NLTK for BLEU computation."""
        if self._nltk_initialized:
            return
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            self._nltk_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
    
    def compute_bleu(
        self,
        reference: str,
        hypothesis: str,
        n_gram: int = 4,
    ) -> float:
        """
        Compute BLEU score.
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            n_gram: Maximum n-gram order
            
        Returns:
            BLEU score (0-1)
        """
        self._init_nltk()
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize
            
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            
            smoothie = SmoothingFunction().method1
            
            weights = tuple([1.0 / n_gram] * n_gram)
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothie,
            )
            
            return float(score)
            
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return 0.0
    
    def compute_rouge(
        self,
        reference: str,
        hypothesis: str,
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            
        Returns:
            Dict with rouge-1, rouge-2, rouge-l scores
        """
        try:
            from rouge_score import rouge_scorer
            
            if self._rouge_scorer is None:
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'],
                    use_stemmer=True,
                )
            
            scores = self._rouge_scorer.score(reference, hypothesis)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
            }
            
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_memory_recall(
        self,
        expected_memories: List[str],
        response: str,
        threshold: float = 0.5,
    ) -> Tuple[float, List[str]]:
        """
        Compute memory recall rate (keyword matching).
        
        Args:
            expected_memories: List of memory contents expected in response
            response: Generated response
            threshold: Minimum keyword match ratio
            
        Returns:
            (recall_rate, matched_memories)
        """
        if not expected_memories:
            return 1.0, []
        
        response_lower = response.lower()
        matched = []
        
        for memory in expected_memories:
            # Extract keywords from memory
            keywords = self._extract_keywords(memory)
            
            if not keywords:
                continue
            
            # Check keyword overlap
            matches = sum(1 for kw in keywords if kw.lower() in response_lower)
            match_ratio = matches / len(keywords)
            
            if match_ratio >= threshold:
                matched.append(memory)
        
        recall = len(matched) / len(expected_memories)
        return recall, matched

    def compute_content_recall(
        self,
        expected_memories: List[str],
        response: str,
        injection_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        多维度内容召回率评估 (v7.0)。
        
        解决原 compute_memory_recall 仅基于关键词匹配、
        天然偏向 RAG 直接 prompt 拼接的问题。
        
        三层召回:
        1. keyword_recall: 关键词匹配 (沿用原方式，保持向后兼容)
        2. injection_recall: 评估实际注入到 prompt 的记忆覆盖率
           - DKI: 偏好注入 + recall_v4 后缀注入的历史
           - RAG: 检索上下文
        3. semantic_recall: 语义相似度 (基于字符级 n-gram 重叠, 无需外部嵌入模型)
        
        Args:
            expected_memories: 期望的记忆内容列表
            response: 模型响应文本
            injection_info: 注入信息 (InjectionInfo.to_dict())
                - mode: 'dki' | 'rag' | 'baseline'
                - preference_text: 偏好文本
                - history_suffix: 历史后缀
                - rag_context: RAG 检索上下文
                - final_input: 最终输入
                
        Returns:
            Dict 包含:
              - keyword_recall: 关键词召回率
              - injection_recall: 注入覆盖率
              - semantic_recall: 语义召回率 (字符级 n-gram)
              - combined_recall: 加权综合召回率
        """
        if not expected_memories:
            return {
                'keyword_recall': 1.0,
                'injection_recall': 1.0,
                'semantic_recall': 1.0,
                'combined_recall': 1.0,
            }
        
        # 1. keyword_recall (原方式)
        kw_recall, _ = self.compute_memory_recall(
            expected_memories=expected_memories,
            response=response,
            threshold=0.3,
        )
        
        # 2. injection_recall (注入覆盖率)
        injection_recall = 0.0
        if injection_info:
            # 构建注入文本池
            injected_text_parts = []
            if injection_info.get('preference_text'):
                injected_text_parts.append(injection_info['preference_text'])
            if injection_info.get('history_suffix'):
                injected_text_parts.append(injection_info['history_suffix'])
            if injection_info.get('rag_context'):
                injected_text_parts.append(injection_info['rag_context'])
            if injection_info.get('final_input'):
                injected_text_parts.append(injection_info['final_input'])
            
            if injected_text_parts:
                injected_text = ' '.join(injected_text_parts).lower()
                matched_count = 0
                for memory in expected_memories:
                    mem_keywords = self._extract_keywords(memory)
                    if not mem_keywords:
                        continue
                    hits = sum(1 for kw in mem_keywords if kw.lower() in injected_text)
                    if hits / len(mem_keywords) >= 0.3:
                        matched_count += 1
                injection_recall = matched_count / len(expected_memories)
        
        # 3. semantic_recall (字符级 n-gram 相似度, 无需外部模型)
        semantic_recall = self._compute_char_ngram_recall(expected_memories, response)
        
        # 加权综合 (DKI 应更多依赖 injection_recall 和 semantic_recall)
        mode = injection_info.get('mode', 'baseline') if injection_info else 'baseline'
        if mode == 'dki':
            # DKI 通过 KV 注入隐式影响，关键词匹配权重降低
            combined = 0.2 * kw_recall + 0.4 * injection_recall + 0.4 * semantic_recall
        elif mode == 'rag':
            # RAG 直接拼接 prompt，关键词匹配权重较高
            combined = 0.4 * kw_recall + 0.3 * injection_recall + 0.3 * semantic_recall
        else:
            combined = kw_recall  # baseline 仅关键词
        
        return {
            'keyword_recall': round(kw_recall, 4),
            'injection_recall': round(injection_recall, 4),
            'semantic_recall': round(semantic_recall, 4),
            'combined_recall': round(combined, 4),
        }
    
    def _compute_char_ngram_recall(
        self,
        expected_memories: List[str],
        response: str,
        n: int = 3,
    ) -> float:
        """
        基于字符级 n-gram 的语义召回率。
        
        不依赖外部嵌入模型，通过计算期望记忆与响应之间的
        字符级 n-gram 重叠度来评估语义相似度。
        
        适用于中英文混合场景。
        """
        if not expected_memories or not response:
            return 0.0
        
        response_lower = response.lower()
        response_ngrams = self._char_ngrams(response_lower, n)
        
        if not response_ngrams:
            return 0.0
        
        scores = []
        for memory in expected_memories:
            mem_lower = memory.lower()
            mem_ngrams = self._char_ngrams(mem_lower, n)
            
            if not mem_ngrams:
                continue
            
            # 计算 n-gram 交集占比
            overlap = mem_ngrams & response_ngrams
            precision = len(overlap) / len(mem_ngrams) if mem_ngrams else 0.0
            scores.append(precision)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _char_ngrams(self, text: str, n: int = 3) -> set:
        """提取字符级 n-grams (去除空白)。"""
        text = re.sub(r'\s+', '', text)
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    def _extract_keywords(self, text: str, min_len: int = 2) -> List[str]:
        """Extract keywords from text (supports Chinese bigram segmentation)."""
        # 英文: 按单词边界分词
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        en_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'shall',
            'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
            'this', 'that', 'these', 'those',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'and', 'or', 'but', 'not', 'if', 'so', 'than', 'too', 'very',
        }
        en_keywords = [w for w in en_words if w not in en_stopwords and len(w) >= min_len]
        
        # 中文: 提取连续中文段落, 然后用 bigram 滑窗分词
        cn_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        cn_stopchars = set('的了是在我你他她们有这那个也就都不吗呢吧啊啦呀请和与')
        cn_keywords = []
        for seg in cn_segments:
            # 先去除停用字
            filtered = ''.join(c for c in seg if c not in cn_stopchars)
            # bigram 滑窗
            for i in range(len(filtered) - 1):
                bigram = filtered[i:i+2]
                if len(bigram) == 2:
                    cn_keywords.append(bigram)
        
        all_keywords = en_keywords + cn_keywords
        # 去重保序, 限制数量
        seen = set()
        unique = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:20]
    
    def compute_hallucination_rate(
        self,
        response: str,
        grounding_texts: List[str],
        known_facts: Optional[List[str]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Estimate hallucination rate (aggregate).
        
        This is a simplified heuristic-based approach.
        For production, use model-based hallucination detection.
        
        Args:
            response: Generated response
            grounding_texts: Source texts (memories, context)
            known_facts: Additional known facts
            
        Returns:
            (hallucination_rate, detected_hallucinations)
        """
        decomposed = self.compute_hallucination_decomposed(
            response=response,
            grounding_texts=grounding_texts,
            known_facts=known_facts,
        )
        total_rate = decomposed['total_rate']
        all_hallucinations = decomposed['fabricated_claims'] + decomposed['irrelevant_claims']
        return total_rate, all_hallucinations

    def compute_hallucination_decomposed(
        self,
        response: str,
        grounding_texts: List[str],
        known_facts: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decomposed hallucination analysis (论文 Table 1b).
        
        将幻觉分解为两类:
        1. fabricated-detail: 编造细节型 — 响应包含具体数字、日期、地址等
           细节，但在 grounding 文本中找不到支撑
        2. irrelevant/off-topic: 无关/跑题型 — 响应内容与查询或 grounding
           文本的主题不相关
        
        Args:
            response: Generated response
            grounding_texts: Source texts (memories, context)
            known_facts: Additional known facts
            query: Original query (用于 off-topic 检测)
            
        Returns:
            Dict with:
              - fabricated_rate: float
              - irrelevant_rate: float  
              - total_rate: float
              - fabricated_claims: List[str]
              - irrelevant_claims: List[str]
              - total_claims: int
        """
        # Combine grounding sources
        all_grounding = ' '.join(grounding_texts)
        if known_facts:
            all_grounding += ' ' + ' '.join(known_facts)
        
        grounding_lower = all_grounding.lower()
        query_lower = (query or '').lower()
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return {
                'fabricated_rate': 0.0,
                'irrelevant_rate': 0.0,
                'total_rate': 0.0,
                'fabricated_claims': [],
                'irrelevant_claims': [],
                'total_claims': 0,
            }
        
        fabricated_claims = []
        irrelevant_claims = []
        
        for claim in claims:
            claim_lower = claim.lower()
            claim_keywords = self._extract_keywords(claim)
            if not claim_keywords:
                continue
            
            # Check if claim is grounded
            grounded_count = sum(1 for kw in claim_keywords if kw.lower() in grounding_lower)
            grounded_ratio = grounded_count / len(claim_keywords) if claim_keywords else 0.0
            
            if grounded_ratio >= 0.3:
                # Claim is at least partially grounded — not hallucination
                continue
            
            # Determine hallucination type
            has_specifics = self._has_specific_details(claim)
            
            if has_specifics:
                # Contains specific numbers/dates/addresses but not grounded
                fabricated_claims.append(claim)
            else:
                # No specific details, check if off-topic
                # Off-topic: claim keywords don't overlap with query or grounding topic
                topic_overlap = False
                if query_lower:
                    query_keywords = self._extract_keywords(query_lower)
                    topic_overlap = any(
                        kw.lower() in claim_lower for kw in query_keywords
                    )
                if not topic_overlap:
                    # Also check grounding topic overlap
                    grounding_keywords = self._extract_keywords(all_grounding)[:20]
                    topic_overlap = any(
                        kw.lower() in claim_lower for kw in grounding_keywords
                    )
                
                if not topic_overlap:
                    irrelevant_claims.append(claim)
                else:
                    # On-topic but ungrounded — classify as fabricated
                    fabricated_claims.append(claim)
        
        total_hallucinations = len(fabricated_claims) + len(irrelevant_claims)
        
        return {
            'fabricated_rate': len(fabricated_claims) / len(claims) if claims else 0.0,
            'irrelevant_rate': len(irrelevant_claims) / len(claims) if claims else 0.0,
            'total_rate': total_hallucinations / len(claims) if claims else 0.0,
            'fabricated_claims': fabricated_claims,
            'irrelevant_claims': irrelevant_claims,
            'total_claims': len(claims),
        }
    
    def _has_specific_details(self, text: str) -> bool:
        """
        检测文本是否包含具体细节 (数字、日期、地址、价格等)。
        
        用于区分 fabricated-detail vs irrelevant hallucination。
        """
        # 数字 (电话、价格、数量等)
        if re.search(r'\d{2,}', text):
            return True
        # 日期模式
        if re.search(r'\d{4}[-/年]\d{1,2}[-/月]', text):
            return True
        # 地址关键词 + 具体信息
        address_indicators = ['路', '街', '号', '区', '市', '省', 'street', 'road', 'avenue', 'blvd']
        if any(ind in text.lower() for ind in address_indicators):
            return True
        # 价格模式
        if re.search(r'[¥$€£]\s*\d+|[\d]+\s*[元块美元]', text):
            return True
        # 百分比
        if re.search(r'\d+\.?\d*\s*%', text):
            return True
        return False
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Split into sentences
        sentences = re.split(r'[.!?。！？]', text)
        
        # Filter for factual claims (simple heuristic)
        claims = []
        fact_indicators = ['is', 'are', 'was', 'were', 'has', 'have', '是', '有',
                          '在', '为', '于', '约', '大约', '位于', '包含', '提供']
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            if any(ind in sent.lower() for ind in fact_indicators):
                claims.append(sent)
        
        return claims
    
    def compute_latency_stats(
        self,
        latencies: List[float],
    ) -> Dict[str, float]:
        """
        Compute latency statistics.
        
        Args:
            latencies: List of latency values in ms
            
        Returns:
            Dict with p50, p95, p99, mean, std
        """
        if not latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'std': 0}
        
        arr = np.array(latencies)
        
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }
    
    def compute_all_metrics(
        self,
        responses: List[Dict[str, Any]],
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a batch of responses.
        
        Args:
            responses: List of response dicts with 'text', 'latency_ms', 'memories_used', etc.
            references: Optional reference texts for BLEU/ROUGE
            
        Returns:
            Aggregated metrics dict
        """
        metrics = {
            'count': len(responses),
            'latency': {},
            'memory_recall': {},
            'text_quality': {},
        }
        
        latencies = [r.get('latency_ms', 0) for r in responses]
        metrics['latency'] = self.compute_latency_stats(latencies)
        
        if references and len(references) == len(responses):
            bleu_scores = []
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for resp, ref in zip(responses, references):
                bleu = self.compute_bleu(ref, resp.get('text', ''))
                bleu_scores.append(bleu)
                
                rouge = self.compute_rouge(ref, resp.get('text', ''))
                for k, v in rouge.items():
                    rouge_scores[k].append(v)
            
            metrics['text_quality'] = {
                'bleu_mean': float(np.mean(bleu_scores)),
                'rouge1_mean': float(np.mean(rouge_scores['rouge1'])),
                'rouge2_mean': float(np.mean(rouge_scores['rouge2'])),
                'rougeL_mean': float(np.mean(rouge_scores['rougeL'])),
            }
        
        return metrics
