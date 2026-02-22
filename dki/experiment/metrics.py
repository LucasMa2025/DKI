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
        Compute memory recall rate.
        
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
    
    def _extract_keywords(self, text: str, min_len: int = 3) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]+\b', text)
        keywords = [w for w in words if len(w) >= min_len]
        return keywords[:10]  # Limit to 10 keywords
    
    def compute_hallucination_rate(
        self,
        response: str,
        grounding_texts: List[str],
        known_facts: Optional[List[str]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Estimate hallucination rate.
        
        This is a simplified heuristic-based approach.
        For production, use model-based hallucination detection.
        
        Args:
            response: Generated response
            grounding_texts: Source texts (memories, context)
            known_facts: Additional known facts
            
        Returns:
            (hallucination_rate, detected_hallucinations)
        """
        # Combine grounding sources
        all_grounding = ' '.join(grounding_texts)
        if known_facts:
            all_grounding += ' ' + ' '.join(known_facts)
        
        grounding_lower = all_grounding.lower()
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return 0.0, []
        
        hallucinations = []
        for claim in claims:
            claim_keywords = self._extract_keywords(claim)
            if not claim_keywords:
                continue
            
            # Check if claim is grounded
            grounded = any(kw.lower() in grounding_lower for kw in claim_keywords)
            if not grounded:
                hallucinations.append(claim)
        
        rate = len(hallucinations) / len(claims) if claims else 0.0
        return rate, hallucinations
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Split into sentences
        sentences = re.split(r'[.!?。！？]', text)
        
        # Filter for factual claims (simple heuristic)
        claims = []
        fact_indicators = ['is', 'are', 'was', 'were', 'has', 'have', '是', '有']
        
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
