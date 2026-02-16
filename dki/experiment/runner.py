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
from dki.database.repository import (
    ExperimentRepository, DemoUserRepository, UserPreferenceRepository, SessionRepository
)
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
    
    # ========================================================================
    # Experiment User & Preference Management
    # ========================================================================
    
    def setup_experiment_users(
        self,
        users: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        为实验创建用户并写入偏好到数据库。
        
        这是保证偏好注入可靠性的关键步骤:
        - DKISystem.chat() 会通过 _load_user_preferences_from_db(user_id) 加载偏好
        - 偏好必须存在于 user_preferences 表中才能被加载
        - 之前实验只通过 add_memory 写入 memories 表，偏好表为空
        
        Args:
            users: 用户列表，每个用户包含:
                - username: 用户名
                - display_name: 显示名称 (可选)
                - preferences: 偏好列表，每项包含:
                    - text: 偏好文本
                    - type: 偏好类型 (general/style/domain 等)
                    - priority: 优先级 (0-10)
                    - category: 分类 (可选)
                
        Returns:
            Dict[username, user_id] 映射
        
        Example:
            users = [
                {
                    "username": "exp_vegetarian",
                    "display_name": "素食实验用户",
                    "preferences": [
                        {"text": "我是素食主义者，不吃肉类", "type": "general", "priority": 9},
                        {"text": "我对海鲜过敏", "type": "general", "priority": 10},
                    ]
                },
            ]
            user_map = runner.setup_experiment_users(users)
        """
        if users is None:
            users = self._get_default_experiment_users()
        
        user_map = {}  # username -> user_id
        
        with self.db_manager.session_scope() as db:
            user_repo = DemoUserRepository(db)
            pref_repo = UserPreferenceRepository(db)
            
            for user_data in users:
                username = user_data["username"]
                display_name = user_data.get("display_name", username)
                
                # 创建或获取用户
                user, created = user_repo.get_or_create(
                    username=username,
                    display_name=display_name,
                )
                user_id = user.id
                user_map[username] = user_id
                
                if created:
                    logger.info(f"Created experiment user: {username} (id={user_id})")
                else:
                    logger.info(f"Found existing experiment user: {username} (id={user_id})")
                
                # 写入偏好 (先清除旧的实验偏好，再写入新的)
                preferences = user_data.get("preferences", [])
                if preferences:
                    # 软删除该用户的旧偏好
                    existing_prefs = pref_repo.get_by_user(user_id)
                    for old_pref in existing_prefs:
                        pref_repo.delete(old_pref.id)
                    
                    # 写入新偏好
                    for pref_data in preferences:
                        pref_repo.create(
                            user_id=user_id,
                            preference_text=pref_data["text"],
                            preference_type=pref_data.get("type", "general"),
                            priority=pref_data.get("priority", 5),
                            category=pref_data.get("category"),
                        )
                    
                    logger.info(
                        f"  Written {len(preferences)} preferences for {username}"
                    )
        
        self._experiment_user_map = user_map
        logger.info(f"Experiment users setup complete: {len(user_map)} users")
        return user_map
    
    def _get_default_experiment_users(self) -> List[Dict[str, Any]]:
        """
        获取默认实验用户配置。
        
        从 config.yaml 的 experiment.users 节读取，
        如果没有配置则使用内置默认值。
        """
        # 尝试从配置文件读取
        exp_config = getattr(self.config, 'experiment', None)
        if exp_config and hasattr(exp_config, 'users'):
            config_users = exp_config.users
            if config_users:
                return config_users
        
        # 内置默认实验用户
        return [
            {
                "username": "exp_user_vegetarian",
                "display_name": "素食实验用户",
                "preferences": [
                    {"text": "我是素食主义者，不吃任何肉类和海鲜", "type": "general", "priority": 10},
                    {"text": "我对海鲜过敏，请不要推荐任何海鲜相关的食物", "type": "general", "priority": 10},
                    {"text": "我住在北京海淀区", "type": "general", "priority": 7},
                ],
            },
            {
                "username": "exp_user_outdoor",
                "display_name": "户外运动实验用户",
                "preferences": [
                    {"text": "我喜欢户外运动，特别是徒步和骑行", "type": "general", "priority": 9},
                    {"text": "我住在上海浦东", "type": "general", "priority": 7},
                    {"text": "我养了一只金毛犬叫小白", "type": "general", "priority": 6},
                ],
            },
            {
                "username": "exp_user_tech",
                "display_name": "技术实验用户",
                "preferences": [
                    {"text": "我是一名数据科学家，擅长Python和机器学习", "type": "technical", "priority": 9},
                    {"text": "我对人工智能和深度学习很感兴趣", "type": "domain", "priority": 8},
                    {"text": "我喜欢阅读科幻小说", "type": "general", "priority": 5},
                ],
            },
            {
                "username": "exp_user_music",
                "display_name": "音乐爱好实验用户",
                "preferences": [
                    {"text": "我是古典音乐的爱好者，特别喜欢贝多芬和莫扎特", "type": "general", "priority": 9},
                    {"text": "我正在学弹吉他", "type": "general", "priority": 7},
                    {"text": "我对辣椒过敏，不能吃辣的食物", "type": "general", "priority": 10},
                    {"text": "我在北京工作", "type": "general", "priority": 6},
                ],
            },
        ]
    
    def _get_experiment_user_id(self, item: Dict[str, Any], default: str = "experiment_user") -> str:
        """
        从数据项中获取实验用户 ID。
        
        优先级:
        1. item 中显式指定的 user_id
        2. 通过 item 的 username 从 _experiment_user_map 查找
        3. 通过 personas 匹配最佳实验用户
        4. 使用默认值
        """
        # 1. 显式指定
        if 'user_id' in item:
            return item['user_id']
        
        # 2. 通过 username 查找
        if hasattr(self, '_experiment_user_map'):
            username = item.get('experiment_user', item.get('username'))
            if username and username in self._experiment_user_map:
                return self._experiment_user_map[username]
        
        # 3. 通过 personas 匹配 (简单关键词匹配)
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            personas = item.get('personas', [])
            if personas:
                return self._match_user_by_personas(personas)
        
        return default
    
    def _match_user_by_personas(self, personas: List[str]) -> str:
        """
        根据 personas 关键词匹配最佳实验用户。
        
        简单策略: 计算每个实验用户偏好与 personas 的关键词重叠度。
        """
        if not hasattr(self, '_experiment_user_map') or not self._experiment_user_map:
            return "experiment_user"
        
        # 获取默认用户配置用于匹配
        default_users = self._get_default_experiment_users()
        personas_text = " ".join(personas).lower()
        
        best_user = None
        best_score = 0
        
        for user_data in default_users:
            username = user_data["username"]
            if username not in self._experiment_user_map:
                continue
            
            score = 0
            for pref in user_data.get("preferences", []):
                pref_text = pref["text"].lower()
                # 简单的关键词重叠计数
                for word in pref_text:
                    if word in personas_text:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_user = username
        
        if best_user:
            return self._experiment_user_map[best_user]
        
        # 返回第一个实验用户作为默认
        first_username = list(self._experiment_user_map.keys())[0]
        return self._experiment_user_map[first_username]
    
    def _write_session_preferences(self, user_id: str, personas: List[str]) -> None:
        """
        为特定 session 的 personas 写入用户偏好表。
        
        用于多轮连贯性实验等场景，每个 session 有不同的 personas，
        需要动态更新该用户的偏好以确保 DKI 偏好注入与当前 session 匹配。
        
        注意: 这会覆盖该用户的现有偏好。
        """
        if not personas:
            return
        
        try:
            with self.db_manager.session_scope() as db:
                pref_repo = UserPreferenceRepository(db)
                
                # 软删除旧偏好
                existing = pref_repo.get_by_user(user_id)
                for old_pref in existing:
                    pref_repo.delete(old_pref.id)
                
                # 写入新偏好
                for idx, persona in enumerate(personas):
                    pref_repo.create(
                        user_id=user_id,
                        preference_text=persona,
                        preference_type="general",
                        priority=10 - idx,  # 越靠前优先级越高
                    )
                
                # 清除 DKI 系统的内存缓存，强制下次从数据库重新加载
                if self.dki_system and hasattr(self.dki_system, '_user_preferences'):
                    self.dki_system._user_preferences.pop(user_id, None)
                    
        except Exception as e:
            logger.warning(f"Failed to write session preferences for {user_id}: {e}")
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        data_path: Optional[str] = None,
        setup_users: bool = True,
        experiment_users: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full experiment.
        
        Args:
            config: Experiment configuration
            data_path: Path to experiment data (JSON file)
            setup_users: Whether to setup experiment users and preferences in DB
            experiment_users: Custom experiment user definitions (optional)
            
        Returns:
            Experiment results dict
        """
        self._ensure_systems()
        
        # 设置实验用户和偏好 (写入数据库，确保偏好注入可靠)
        if setup_users:
            self.setup_experiment_users(experiment_users)
        
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
        """Run experiment for a specific mode.
        
        Each sample uses its own session_id to prevent history accumulation
        across unrelated samples, which would cause prompt length overflow
        (decoder prompt > max_model_len).
        
        改进: 每个 sample 使用对应的实验用户 ID，确保偏好从数据库正确加载。
        """
        samples = data[:config.max_samples]
        results = []
        
        base_ts = int(time.time())
        
        # Run queries — each sample gets its own session to avoid cross-contamination
        for idx, item in enumerate(tqdm(samples, desc=f"Running {mode}")):
            # Per-sample session: prevents history from accumulating across samples
            session_id = f"exp_{mode}_{base_ts}_{idx}"
            
            # 获取该 sample 对应的实验用户 ID (从数据库中匹配)
            user_id = self._get_experiment_user_id(item)
            
            # Add memories for this sample only
            memories = item.get('personas', []) + item.get('supporting_facts', [])
            if 'memory' in item:
                memories.append(item['memory'])
            
            for mem in memories:
                if mode == 'dki':
                    self.dki_system.add_memory(session_id, mem)
                elif mode == 'rag':
                    self.rag_system.add_memory(session_id, mem)
            
            queries = self._extract_queries(item)
            
            for query in queries:
                result = self._run_single_query(
                    mode=mode,
                    query=query,
                    session_id=session_id,
                    item=item,
                    config=config,
                    user_id=user_id,
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
        import numpy as np
        
        latencies = [r.latency_ms for r in results]
        
        # Filter out error results
        valid_results = [r for r in results if not r.response.startswith("ERROR:")]
        error_count = len(results) - len(valid_results)
        
        metrics = {
            'count': len(results),
            'valid_count': len(valid_results),
            'error_count': error_count,
            'latency': self.metrics.compute_latency_stats(latencies),
            'memory_usage': {
                'total_memories_used': sum(len(r.memories_used) for r in results),
                'avg_memories_per_query': sum(len(r.memories_used) for r in results) / max(len(results), 1),
            },
        }
        
        # Alpha stats for DKI
        alphas = [r.alpha for r in results if r.alpha is not None]
        if alphas:
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
        
        # Memory recall: check if response references the memories
        recall_scores = []
        for r in valid_results:
            if r.memories_used:
                recall, _ = self.metrics.compute_memory_recall(
                    expected_memories=r.memories_used,
                    response=r.response,
                    threshold=0.3,
                )
                recall_scores.append(recall)
        if recall_scores:
            metrics['memory_recall'] = {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores)),
            }
        
        # Hallucination rate (heuristic)
        hallucination_rates = []
        for r in valid_results:
            if r.memories_used:
                h_rate, _ = self.metrics.compute_hallucination_rate(
                    response=r.response,
                    grounding_texts=r.memories_used,
                )
                hallucination_rates.append(h_rate)
        if hallucination_rates:
            metrics['hallucination'] = {
                'mean_rate': float(np.mean(hallucination_rates)),
                'std_rate': float(np.std(hallucination_rates)),
            }
        
        # Response length statistics
        response_lengths = [len(r.response) for r in valid_results]
        if response_lengths:
            metrics['response_length'] = {
                'mean': float(np.mean(response_lengths)),
                'std': float(np.std(response_lengths)),
                'min': int(np.min(response_lengths)),
                'max': int(np.max(response_lengths)),
            }
        
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
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """
        Run α sensitivity analysis.
        
        Tests DKI performance across different α values.
        
        改进: 使用数据库中的实验用户偏好，确保偏好 K/V 注入在不同 α 下生效。
        """
        self._ensure_systems()
        
        # 设置实验用户
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
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
        
        # 使用第一个实验用户 (alpha 实验关注注入强度，用户一致即可)
        user_id = "experiment_user"
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            first_username = list(self._experiment_user_map.keys())[0]
            user_id = self._experiment_user_map[first_username]
        
        # 每个 alpha 值使用独立 session，避免历史累积
        base_ts = int(time.time())
        
        # Test each alpha
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_results = []
            session_id = f"alpha_exp_{base_ts}_{alpha_idx}"
            
            # Add memories for this alpha session
            for item in data[:50]:
                if 'memory' in item:
                    self.dki_system.add_memory(session_id, item['memory'])
            
            for item in tqdm(data[:50], desc=f"α={alpha}"):
                query = item.get('query', '')
                if not query:
                    continue
                
                response = self.dki_system.chat(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
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
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """
        Run latency comparison between first turn and subsequent turns.
        
        Tests session cache effectiveness.
        
        改进: 使用数据库中的实验用户偏好。
        """
        self._ensure_systems()
        
        # 设置实验用户
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info(f"Running latency comparison with {n_turns} turns")
        
        session_id = f"latency_exp_{int(time.time())}"
        
        # 获取实验用户 ID
        user_id = "experiment_user"
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            first_username = list(self._experiment_user_map.keys())[0]
            user_id = self._experiment_user_map[first_username]
        
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
            response = self.dki_system.chat(
                query=query,
                session_id=session_id,
                user_id=user_id,
            )
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
            response = self.rag_system.chat(
                query=query,
                session_id=rag_session_id,
                user_id=user_id,
            )
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


    def run_multi_turn_coherence(
        self,
        data_path: Optional[str] = None,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """
        运行多轮连贯性实验
        
        测试 DKI 和 RAG 在多轮对话中的记忆保持能力。
        每个会话有明确的期望记忆回忆，可以精确衡量记忆召回率。
        
        改进: 使用数据库中的实验用户偏好，personas 同时写入偏好表。
        """
        self._ensure_systems()
        
        # 设置实验用户
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info("Running multi-turn coherence experiment")
        
        # Load data
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data_file = Path("./data/multi_turn_coherence.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning("No multi-turn coherence data found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                data = gen.generate_multi_turn_coherence()
        
        results = {
            'dki': {'sessions': [], 'per_turn_recall': {}},
            'rag': {'sessions': [], 'per_turn_recall': {}},
        }
        
        for mode in ['dki', 'rag']:
            system = self.dki_system if mode == 'dki' else self.rag_system
            
            for session_data in tqdm(data[:20], desc=f"Coherence ({mode})"):
                session_id = f"coherence_{mode}_{session_data['session_id']}"
                
                # 获取该 session 对应的实验用户 ID
                user_id = self._get_experiment_user_id(session_data)
                
                # 为该 session 动态写入 personas 作为偏好 (确保 DKI 偏好注入生效)
                if mode == 'dki':
                    self._write_session_preferences(user_id, session_data.get('personas', []))
                
                # Add memories
                for mem in session_data['personas']:
                    system.add_memory(session_id, mem)
                
                session_results = []
                
                for turn_idx, turn in enumerate(session_data['turns']):
                    query = turn['query']
                    
                    if mode == 'dki':
                        response = self.dki_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                    else:
                        response = self.rag_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                    
                    # Compute recall for turns that test memory
                    recall_score = 0.0
                    if turn.get('tests_memory') and turn.get('expected_recall'):
                        expected = turn['expected_recall']
                        response_lower = response_text.lower()
                        hits = sum(1 for kw in expected if kw.lower() in response_lower)
                        recall_score = hits / len(expected) if expected else 0.0
                    
                    turn_result = {
                        'turn_idx': turn_idx,
                        'query': query,
                        'response': response_text,
                        'tests_memory': turn.get('tests_memory', False),
                        'expected_recall': turn.get('expected_recall', []),
                        'recall_score': recall_score,
                    }
                    session_results.append(turn_result)
                    
                    # Aggregate per-turn recall
                    turn_key = f"turn_{turn_idx}"
                    if turn_key not in results[mode]['per_turn_recall']:
                        results[mode]['per_turn_recall'][turn_key] = []
                    if turn.get('tests_memory'):
                        results[mode]['per_turn_recall'][turn_key].append(recall_score)
                
                results[mode]['sessions'].append({
                    'session_id': session_data['session_id'],
                    'turns': session_results,
                })
        
        # Compute summary
        import numpy as np
        for mode in ['dki', 'rag']:
            per_turn = results[mode]['per_turn_recall']
            results[mode]['per_turn_summary'] = {
                turn_key: {
                    'mean_recall': float(np.mean(scores)) if scores else 0.0,
                    'count': len(scores),
                }
                for turn_key, scores in per_turn.items()
            }
            
            # Overall recall
            all_recalls = [
                s for scores in per_turn.values() for s in scores
            ]
            results[mode]['overall_recall'] = float(np.mean(all_recalls)) if all_recalls else 0.0
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"multi_turn_coherence_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Multi-turn coherence results saved to {filepath}")
        logger.info(
            f"Overall Recall - DKI: {results['dki']['overall_recall']:.3f}, "
            f"RAG: {results['rag']['overall_recall']:.3f}"
        )
        return results
    
    def run_ablation_study(
        self,
        data_path: Optional[str] = None,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """
        运行消融实验
        
        测试 DKI 各组件的独立贡献:
        - full_dki: 完整 DKI (偏好 K/V + 历史后缀 + 门控)
        - no_gating: 无门控 (固定 α=1.0)
        - rag_baseline: RAG 对照
        - no_memory: 无记忆基线
        
        改进: 使用数据库中的实验用户偏好。
        """
        self._ensure_systems()
        
        # 设置实验用户
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info("Running ablation study")
        
        # Load data
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data_file = Path("./data/ablation.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning("No ablation data found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                data = gen.generate_ablation_data()
        
        ablation_configs = {
            'full_dki': {'system': 'dki', 'force_alpha': None, 'use_memory': True},
            'no_gating': {'system': 'dki', 'force_alpha': 1.0, 'use_memory': True},
            'rag_baseline': {'system': 'rag', 'force_alpha': None, 'use_memory': True},
            'no_memory': {'system': 'dki', 'force_alpha': None, 'use_memory': False},
        }
        
        results = {mode: {'samples': [], 'latencies': []} for mode in ablation_configs}
        
        # 获取实验用户 ID
        user_id = "experiment_user"
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            first_username = list(self._experiment_user_map.keys())[0]
            user_id = self._experiment_user_map[first_username]
        
        for ablation_mode, config in ablation_configs.items():
            logger.info(f"Running ablation: {ablation_mode}")
            
            session_id = f"ablation_{ablation_mode}_{int(time.time())}"
            system = self.dki_system if config['system'] == 'dki' else self.rag_system
            
            # Add memories if applicable
            if config['use_memory']:
                for item in data[:30]:
                    if 'memory' in item:
                        system.add_memory(session_id, item['memory'])
            
            for item in tqdm(data[:30], desc=f"Ablation ({ablation_mode})"):
                query = item.get('query', '')
                if not query:
                    continue
                
                try:
                    if config['system'] == 'dki':
                        response = self.dki_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                            force_alpha=config.get('force_alpha'),
                            allow_injection=config['use_memory'],
                        )
                        response_text = response.text
                        latency = response.latency_ms
                    else:
                        response = self.rag_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                        latency = response.latency_ms
                    
                    # Check memory recall
                    relevant = item.get('relevant_memories', [])
                    recall_score = 0.0
                    if relevant:
                        recall_score, _ = self.metrics.compute_memory_recall(
                            expected_memories=relevant,
                            response=response_text,
                            threshold=0.3,
                        )
                    
                    results[ablation_mode]['samples'].append({
                        'query': query,
                        'response': response_text,
                        'latency_ms': latency,
                        'memory_recall': recall_score,
                    })
                    results[ablation_mode]['latencies'].append(latency)
                    
                except Exception as e:
                    logger.error(f"Ablation query failed ({ablation_mode}): {e}")
        
        # Compute summaries
        import numpy as np
        summary = {}
        for mode, mode_results in results.items():
            recalls = [s['memory_recall'] for s in mode_results['samples']]
            latencies = mode_results['latencies']
            
            summary[mode] = {
                'sample_count': len(mode_results['samples']),
                'mean_recall': float(np.mean(recalls)) if recalls else 0.0,
                'mean_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
                'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0.0,
            }
        
        results['summary'] = summary
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"ablation_study_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ablation study results saved to {filepath}")
        for mode, s in summary.items():
            logger.info(f"  {mode}: recall={s['mean_recall']:.3f}, latency={s['mean_latency_ms']:.1f}ms")
        
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
