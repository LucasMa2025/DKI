"""
Experiment Runner for DKI System
Runs comparison experiments between RAG and DKI
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger
from tqdm import tqdm

from dki.core.dki_system import DKISystem, DKIResponse
from dki.core.rag_system import RAGSystem, RAGResponse
from dki.experiment.metrics import MetricsCalculator
from dki.database.connection import DatabaseManager
from dki.database.repository import ExperimentRepository
from dki.config.config_loader import ConfigLoader


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str = ""
    modes: List[str] = field(default_factory=lambda: ["rag", "dki", "baseline"])
    datasets: List[str] = field(default_factory=lambda: ["persona_chat", "memory_qa"])
    max_samples: int = 100
    max_new_tokens: int = 256
    temperature: float = 0.7
    alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'modes': self.modes,
            'datasets': self.datasets,
            'max_samples': self.max_samples,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'alpha_values': self.alpha_values,
        }


@dataclass
class InjectionInfo:
    """
    注入信息记录 - 用于显示 DKI/RAG 的实际注入内容
    
    DKI: 显示偏好文本 + 历史后缀提示词 (不显示实际 K/V)
    RAG: 显示完整的构造提示词
    """
    mode: str  # 'dki' or 'rag'
    
    # 原始用户查询
    original_query: str = ""
    
    # DKI 偏好注入 (明文)
    preference_text: Optional[str] = None
    preference_tokens: int = 0
    
    # DKI 历史后缀 (明文)
    history_suffix: Optional[str] = None
    history_tokens: int = 0
    history_messages: List[Dict[str, str]] = field(default_factory=list)
    
    # RAG 完整提示词
    rag_prompt: Optional[str] = None
    rag_context: Optional[str] = None  # 检索到的上下文
    
    # 最终发送给模型的输入
    final_input: str = ""
    
    # 注入参数
    alpha: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'original_query': self.original_query,
            'preference_text': self.preference_text,
            'preference_tokens': self.preference_tokens,
            'history_suffix': self.history_suffix,
            'history_tokens': self.history_tokens,
            'history_messages': self.history_messages,
            'rag_prompt': self.rag_prompt,
            'rag_context': self.rag_context,
            'final_input': self.final_input,
            'alpha': self.alpha,
        }
    
    def get_display_text(self) -> str:
        """获取用于显示的格式化文本"""
        lines = []
        lines.append(f"═══════════════════════════════════════════════════════")
        lines.append(f"  模式: {self.mode.upper()}")
        lines.append(f"═══════════════════════════════════════════════════════")
        lines.append(f"")
        lines.append(f"【原始查询】")
        lines.append(f"{self.original_query}")
        lines.append(f"")
        
        if self.mode == 'dki':
            if self.preference_text:
                lines.append(f"【偏好注入】(K/V 注入, α={self.alpha:.2f}, {self.preference_tokens} tokens)")
                lines.append(f"───────────────────────────────────────────────────────")
                lines.append(self.preference_text)
                lines.append(f"")
            
            if self.history_suffix:
                lines.append(f"【历史后缀】(Suffix Prompt, {self.history_tokens} tokens)")
                lines.append(f"───────────────────────────────────────────────────────")
                lines.append(self.history_suffix)
                lines.append(f"")
            
            if self.history_messages:
                lines.append(f"【历史消息】({len(self.history_messages)} 条)")
                lines.append(f"───────────────────────────────────────────────────────")
                for msg in self.history_messages:
                    role = "用户" if msg['role'] == 'user' else "助手"
                    lines.append(f"  [{role}] {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                lines.append(f"")
        
        elif self.mode == 'rag':
            if self.rag_context:
                lines.append(f"【检索上下文】")
                lines.append(f"───────────────────────────────────────────────────────")
                lines.append(self.rag_context)
                lines.append(f"")
            
            if self.rag_prompt:
                lines.append(f"【完整提示词】")
                lines.append(f"───────────────────────────────────────────────────────")
                lines.append(self.rag_prompt)
                lines.append(f"")
        
        lines.append(f"【最终输入】")
        lines.append(f"───────────────────────────────────────────────────────")
        # 截断过长的输入
        final = self.final_input
        if len(final) > 2000:
            final = final[:1000] + "\n... (中间省略) ...\n" + final[-500:]
        lines.append(final)
        lines.append(f"")
        lines.append(f"═══════════════════════════════════════════════════════")
        
        return "\n".join(lines)


@dataclass
class ExperimentResult:
    """Single experiment result."""
    mode: str
    dataset: str
    sample_id: str
    query: str
    response: str
    latency_ms: float
    memories_used: List[str]
    alpha: Optional[float] = None
    cache_hit: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    # 新增: 注入信息
    injection_info: Optional[InjectionInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'dataset': self.dataset,
            'sample_id': self.sample_id,
            'query': self.query,
            'response': self.response,
            'latency_ms': self.latency_ms,
            'memories_used': self.memories_used,
            'alpha': self.alpha,
            'cache_hit': self.cache_hit,
            'metrics': self.metrics,
            'injection_info': self.injection_info.to_dict() if self.injection_info else None,
        }


class ExperimentRunner:
    """
    Run comparison experiments between RAG and DKI.
    
    Supported experiments:
    1. Hallucination Comparison: Same recall, compare hallucination rates
    2. Latency Comparison: First turn vs subsequent turns
    3. Memory Recall Test: Without explicit prompt hints
    4. Alpha Sensitivity Analysis: Vary α from 0 to 1
    """
    
    def __init__(
        self,
        dki_system: Optional[DKISystem] = None,
        rag_system: Optional[RAGSystem] = None,
        output_dir: str = "./experiment_results",
    ):
        self.config = ConfigLoader().config
        
        self.dki_system = dki_system
        self.rag_system = rag_system
        self.metrics = MetricsCalculator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database
        self.db_manager = DatabaseManager(
            db_path=self.config.database.path,
        )
    
    def _ensure_systems(self):
        """Ensure DKI and RAG systems are initialized."""
        if self.dki_system is None:
            self.dki_system = DKISystem()
        if self.rag_system is None:
            self.rag_system = RAGSystem()
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a full experiment.
        
        Args:
            config: Experiment configuration
            data_path: Path to experiment data (JSON file)
            
        Returns:
            Experiment results dict
        """
        self._ensure_systems()
        
        logger.info(f"Starting experiment: {config.name}")
        
        # Create experiment record
        with self.db_manager.session_scope() as db:
            exp_repo = ExperimentRepository(db)
            experiment = exp_repo.create(
                name=config.name,
                config=config.to_dict(),
                description=config.description,
            )
            experiment_id = experiment.id
            exp_repo.update_status(experiment_id, 'running')
        
        results = {
            'experiment_id': experiment_id,
            'config': config.to_dict(),
            'started_at': datetime.now().isoformat(),
            'results_by_mode': {},
            'aggregated_metrics': {},
        }
        
        try:
            # Load data
            if data_path:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = self._load_default_data(config.datasets)
            
            # Run for each mode
            for mode in config.modes:
                logger.info(f"Running mode: {mode}")
                mode_results = self._run_mode(mode, data, config)
                results['results_by_mode'][mode] = mode_results
            
            # Compute aggregated metrics
            results['aggregated_metrics'] = self._aggregate_metrics(results['results_by_mode'])
            results['completed_at'] = datetime.now().isoformat()
            
            # Update experiment status
            with self.db_manager.session_scope() as db:
                exp_repo = ExperimentRepository(db)
                exp_repo.update_status(experiment_id, 'completed')
                
                # Store results
                for mode, mode_results in results['results_by_mode'].items():
                    exp_repo.add_result(
                        experiment_id=experiment_id,
                        mode=mode,
                        dataset='combined',
                        metrics=mode_results.get('metrics', {}),
                        sample_count=len(mode_results.get('samples', [])),
                    )
            
            # Save results to file
            self._save_results(results)
            
            logger.info(f"Experiment completed: {experiment_id}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            with self.db_manager.session_scope() as db:
                exp_repo = ExperimentRepository(db)
                exp_repo.update_status(experiment_id, 'failed')
            raise
        
        return results
    
    def _load_default_data(self, datasets: List[str]) -> List[Dict[str, Any]]:
        """Load default experiment data."""
        data = []
        data_dir = Path("./data")
        
        for dataset in datasets:
            data_file = data_dir / f"{dataset}.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                    for item in dataset_data:
                        item['_dataset'] = dataset
                    data.extend(dataset_data)
        
        return data
    
    def _run_mode(
        self,
        mode: str,
        data: List[Dict[str, Any]],
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        """Run experiment for a specific mode."""
        samples = data[:config.max_samples]
        results = []
        
        session_id = f"exp_{mode}_{int(time.time())}"
        
        # Add memories from data
        for item in samples:
            memories = item.get('personas', []) + item.get('supporting_facts', [])
            if 'memory' in item:
                memories.append(item['memory'])
            
            for mem in memories:
                if mode == 'dki':
                    self.dki_system.add_memory(session_id, mem)
                elif mode == 'rag':
                    self.rag_system.add_memory(session_id, mem)
        
        # Run queries
        for item in tqdm(samples, desc=f"Running {mode}"):
            queries = self._extract_queries(item)
            
            for query in queries:
                result = self._run_single_query(
                    mode=mode,
                    query=query,
                    session_id=session_id,
                    item=item,
                    config=config,
                )
                results.append(result)
        
        # Compute mode metrics
        mode_metrics = self._compute_mode_metrics(results)
        
        return {
            'mode': mode,
            'samples': [r.to_dict() for r in results],
            'metrics': mode_metrics,
        }
    
    def _extract_queries(self, item: Dict[str, Any]) -> List[str]:
        """Extract queries from data item."""
        queries = []
        
        if 'query' in item:
            queries.append(item['query'])
        elif 'question' in item:
            queries.append(item['question'])
        elif 'turns' in item:
            for turn in item['turns']:
                if 'query' in turn:
                    queries.append(turn['query'])
        
        return queries
    
    def _run_single_query(
        self,
        mode: str,
        query: str,
        session_id: str,
        item: Dict[str, Any],
        config: ExperimentConfig,
        user_id: str = "experiment_user",
    ) -> ExperimentResult:
        """Run a single query and capture injection info."""
        try:
            if mode == 'dki':
                response = self.dki_system.chat(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                
                # 构造 DKI 注入信息
                hybrid_info = response.metadata.get('hybrid_injection', {})
                injection_info = InjectionInfo(
                    mode='dki',
                    original_query=query,
                    preference_text=hybrid_info.get('preference_text'),
                    preference_tokens=hybrid_info.get('preference_tokens', 0),
                    history_suffix=hybrid_info.get('history_suffix_text'),
                    history_tokens=hybrid_info.get('history_tokens', 0),
                    history_messages=hybrid_info.get('history_messages', []),
                    final_input=hybrid_info.get('final_input', query),
                    alpha=response.gating_decision.alpha,
                )
                
                return ExperimentResult(
                    mode=mode,
                    dataset=item.get('_dataset', 'unknown'),
                    sample_id=item.get('id', item.get('session_id', '')),
                    query=query,
                    response=response.text,
                    latency_ms=response.latency_ms,
                    memories_used=[m.memory_id for m in response.memories_used],
                    alpha=response.gating_decision.alpha,
                    cache_hit=response.cache_hit,
                    injection_info=injection_info,
                )
                
            elif mode == 'rag':
                response = self.rag_system.chat(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                
                # 构造 RAG 注入信息
                prompt_info = response.prompt_info
                injection_info = InjectionInfo(
                    mode='rag',
                    original_query=query,
                    rag_context=prompt_info.retrieved_context if prompt_info else None,
                    rag_prompt=prompt_info.final_prompt if prompt_info else None,
                    history_messages=prompt_info.history_messages if prompt_info else [],
                    final_input=prompt_info.final_prompt if prompt_info else query,
                )
                
                return ExperimentResult(
                    mode=mode,
                    dataset=item.get('_dataset', 'unknown'),
                    sample_id=item.get('id', item.get('session_id', '')),
                    query=query,
                    response=response.text,
                    latency_ms=response.latency_ms,
                    memories_used=[m.memory_id for m in response.memories_used],
                    injection_info=injection_info,
                )
                
            else:  # baseline
                # Baseline: no memory injection
                output = self.dki_system.model.generate(
                    prompt=query,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                
                # Baseline 无注入
                injection_info = InjectionInfo(
                    mode='baseline',
                    original_query=query,
                    final_input=query,
                )
                
                return ExperimentResult(
                    mode=mode,
                    dataset=item.get('_dataset', 'unknown'),
                    sample_id=item.get('id', item.get('session_id', '')),
                    query=query,
                    response=output.text,
                    latency_ms=output.latency_ms,
                    memories_used=[],
                    injection_info=injection_info,
                )
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ExperimentResult(
                mode=mode,
                dataset=item.get('_dataset', 'unknown'),
                sample_id=item.get('id', ''),
                query=query,
                response=f"ERROR: {e}",
                latency_ms=0,
                memories_used=[],
            )
    
    def _compute_mode_metrics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compute metrics for a mode."""
        latencies = [r.latency_ms for r in results]
        
        metrics = {
            'count': len(results),
            'latency': self.metrics.compute_latency_stats(latencies),
            'memory_usage': {
                'total_memories_used': sum(len(r.memories_used) for r in results),
                'avg_memories_per_query': sum(len(r.memories_used) for r in results) / max(len(results), 1),
            },
        }
        
        # Alpha stats for DKI
        alphas = [r.alpha for r in results if r.alpha is not None]
        if alphas:
            import numpy as np
            metrics['alpha'] = {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'min': float(np.min(alphas)),
                'max': float(np.max(alphas)),
            }
        
        # Cache hit rate for DKI
        cache_hits = [r.cache_hit for r in results]
        if any(cache_hits):
            metrics['cache_hit_rate'] = sum(cache_hits) / len(cache_hits)
        
        return metrics
    
    def _aggregate_metrics(self, results_by_mode: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate metrics across modes for comparison."""
        aggregated = {}
        
        for mode, mode_data in results_by_mode.items():
            metrics = mode_data.get('metrics', {})
            aggregated[mode] = {
                'latency_p50': metrics.get('latency', {}).get('p50', 0),
                'latency_p95': metrics.get('latency', {}).get('p95', 0),
                'avg_memories': metrics.get('memory_usage', {}).get('avg_memories_per_query', 0),
            }
            
            if 'alpha' in metrics:
                aggregated[mode]['alpha_mean'] = metrics['alpha']['mean']
            if 'cache_hit_rate' in metrics:
                aggregated[mode]['cache_hit_rate'] = metrics['cache_hit_rate']
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{results['experiment_id']}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def run_alpha_sensitivity(
        self,
        data_path: Optional[str] = None,
        alpha_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run α sensitivity analysis.
        
        Tests DKI performance across different α values.
        """
        self._ensure_systems()
        
        alpha_values = alpha_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        logger.info(f"Running α sensitivity analysis with values: {alpha_values}")
        
        # Load data
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data_file = Path("./data/alpha_sensitivity.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []
        
        results = {
            'alpha_values': alpha_values,
            'results_by_alpha': {},
        }
        
        session_id = f"alpha_exp_{int(time.time())}"
        
        # Add memories
        for item in data[:50]:
            if 'memory' in item:
                self.dki_system.add_memory(session_id, item['memory'])
        
        # Test each alpha
        for alpha in alpha_values:
            alpha_results = []
            
            for item in tqdm(data[:50], desc=f"α={alpha}"):
                query = item.get('query', '')
                if not query:
                    continue
                
                response = self.dki_system.chat(
                    query=query,
                    session_id=session_id,
                    force_alpha=alpha,
                )
                
                alpha_results.append({
                    'query': query,
                    'response': response.text,
                    'latency_ms': response.latency_ms,
                    'actual_alpha': response.gating_decision.alpha,
                })
            
            latencies = [r['latency_ms'] for r in alpha_results]
            results['results_by_alpha'][str(alpha)] = {
                'samples': alpha_results,
                'latency_stats': self.metrics.compute_latency_stats(latencies),
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"alpha_sensitivity_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"α sensitivity results saved to {filepath}")
        return results
    
    def run_latency_comparison(
        self,
        n_turns: int = 10,
    ) -> Dict[str, Any]:
        """
        Run latency comparison between first turn and subsequent turns.
        
        Tests session cache effectiveness.
        """
        self._ensure_systems()
        
        logger.info(f"Running latency comparison with {n_turns} turns")
        
        session_id = f"latency_exp_{int(time.time())}"
        
        # Add some memories
        memories = [
            "User prefers vegetarian food.",
            "User lives in Beijing.",
            "User enjoys hiking.",
        ]
        for mem in memories:
            self.dki_system.add_memory(session_id, mem)
        
        queries = [
            "What should I eat for dinner?",
            "Recommend a weekend activity.",
            "What's the weather like?",
            "Suggest a restaurant.",
            "What hobbies should I try?",
        ] * 2
        
        results = {
            'dki_latencies': [],
            'rag_latencies': [],
        }
        
        # DKI turns
        for i, query in enumerate(queries[:n_turns]):
            response = self.dki_system.chat(query, session_id)
            results['dki_latencies'].append({
                'turn': i + 1,
                'latency_ms': response.latency_ms,
                'cache_hit': response.cache_hit,
            })
        
        # RAG turns
        rag_session_id = f"rag_latency_exp_{int(time.time())}"
        for mem in memories:
            self.rag_system.add_memory(rag_session_id, mem)
        
        for i, query in enumerate(queries[:n_turns]):
            response = self.rag_system.chat(query, rag_session_id)
            results['rag_latencies'].append({
                'turn': i + 1,
                'latency_ms': response.latency_ms,
            })
        
        # Compute stats
        dki_first = results['dki_latencies'][0]['latency_ms'] if results['dki_latencies'] else 0
        dki_subsequent = [r['latency_ms'] for r in results['dki_latencies'][1:]]
        rag_all = [r['latency_ms'] for r in results['rag_latencies']]
        
        import numpy as np
        results['summary'] = {
            'dki_first_turn': dki_first,
            'dki_subsequent_mean': float(np.mean(dki_subsequent)) if dki_subsequent else 0,
            'rag_mean': float(np.mean(rag_all)) if rag_all else 0,
            'speedup_subsequent': (
                float(np.mean(rag_all)) / float(np.mean(dki_subsequent))
                if dki_subsequent and np.mean(dki_subsequent) > 0 else 0
            ),
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"latency_comparison_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Latency comparison results saved to {filepath}")
        return results


class InjectionInfoViewer:
    """
    注入信息查看器 - 用于显示和比较 DKI/RAG 的注入内容
    
    功能:
    - 显示 DKI 偏好注入 (明文, 不显示 K/V)
    - 显示 DKI 历史后缀提示词
    - 显示 RAG 完整提示词
    - 支持最大化和复制
    """
    
    def __init__(self, output_dir: str = "./injection_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._injection_history: List[InjectionInfo] = []
    
    def record(self, injection_info: InjectionInfo) -> None:
        """记录注入信息"""
        self._injection_history.append(injection_info)
    
    def get_latest(self, n: int = 10) -> List[InjectionInfo]:
        """获取最近的注入信息"""
        return self._injection_history[-n:]
    
    def display(self, injection_info: InjectionInfo) -> str:
        """
        显示注入信息 (返回格式化文本)
        
        可用于:
        - 控制台输出
        - 保存到文件
        - UI 显示
        """
        return injection_info.get_display_text()
    
    def compare(self, dki_info: InjectionInfo, rag_info: InjectionInfo) -> str:
        """
        并排比较 DKI 和 RAG 的注入信息
        """
        lines = []
        lines.append("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║                              DKI vs RAG 注入信息对比                                               ║")
        lines.append("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        # 原始查询
        lines.append(f"【原始查询】")
        lines.append(f"  {dki_info.original_query}")
        lines.append("")
        
        # DKI 部分
        lines.append("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                                          DKI 注入                                                   │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────────────────────────────┤")
        
        if dki_info.preference_text:
            lines.append(f"│ 【偏好注入】(K/V 注入, α={dki_info.alpha:.2f}, {dki_info.preference_tokens} tokens)")
            lines.append(f"│   {dki_info.preference_text[:80]}{'...' if len(dki_info.preference_text) > 80 else ''}")
        else:
            lines.append("│ 【偏好注入】无")
        
        if dki_info.history_messages:
            lines.append(f"│ 【历史消息】({len(dki_info.history_messages)} 条, {dki_info.history_tokens} tokens)")
            for msg in dki_info.history_messages[:3]:
                role = "用户" if msg['role'] == 'user' else "助手"
                lines.append(f"│   [{role}] {msg['content'][:60]}{'...' if len(msg['content']) > 60 else ''}")
            if len(dki_info.history_messages) > 3:
                lines.append(f"│   ... 还有 {len(dki_info.history_messages) - 3} 条消息")
        else:
            lines.append("│ 【历史消息】无")
        
        lines.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # RAG 部分
        lines.append("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                                          RAG 注入                                                   │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────────────────────────────┤")
        
        if rag_info.rag_context:
            lines.append(f"│ 【检索上下文】")
            context_preview = rag_info.rag_context[:200].replace('\n', ' ')
            lines.append(f"│   {context_preview}{'...' if len(rag_info.rag_context) > 200 else ''}")
        else:
            lines.append("│ 【检索上下文】无")
        
        if rag_info.history_messages:
            lines.append(f"│ 【历史消息】({len(rag_info.history_messages)} 条)")
            for msg in rag_info.history_messages[:3]:
                role = "用户" if msg['role'] == 'user' else "助手"
                lines.append(f"│   [{role}] {msg['content'][:60]}{'...' if len(msg['content']) > 60 else ''}")
        else:
            lines.append("│ 【历史消息】无")
        
        lines.append("│")
        lines.append(f"│ 【完整提示词长度】{len(rag_info.rag_prompt or '')} 字符")
        lines.append("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")
        
        return "\n".join(lines)
    
    def save_to_file(self, injection_info: InjectionInfo, filename: Optional[str] = None) -> str:
        """保存注入信息到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"injection_{injection_info.mode}_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        content = self.display(injection_info)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Injection info saved to {filepath}")
        return str(filepath)
    
    def save_comparison(
        self,
        dki_info: InjectionInfo,
        rag_info: InjectionInfo,
        filename: Optional[str] = None,
    ) -> str:
        """保存对比信息到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"injection_comparison_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        content = self.compare(dki_info, rag_info)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Injection comparison saved to {filepath}")
        return str(filepath)
    
    def export_json(self, injection_info: InjectionInfo) -> Dict[str, Any]:
        """导出为 JSON 格式 (用于 API)"""
        return {
            **injection_info.to_dict(),
            'display_text': injection_info.get_display_text(),
        }
    
    def get_copyable_text(self, injection_info: InjectionInfo) -> str:
        """
        获取可复制的纯文本格式
        
        用于 UI 的复制功能
        """
        return injection_info.get_display_text()
