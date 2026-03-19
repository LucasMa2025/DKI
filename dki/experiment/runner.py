"""
Experiment Runner for DKI System — v9.1 Refactored
Runs comparison experiments between RAG and DKI

重构说明 (v9.1):
- 使用独立的 dki.experiment.store (从 demo.store 复制, 仅 SQLite)
- 使用独立的 dki.db 数据库 (不与 demo.db 共享)
- 使用 dki.integration.create_plugin 标准集成 DKIPlugin
- 通过 ConfigDrivenAdapter 映射 demo_* 表
- 注入明文通过 InjectionMetadata 正式字段获取
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger
from tqdm import tqdm

from dki.core.rag_system import RAGSystem, RAGResponse
from dki.experiment.metrics import MetricsCalculator
from dki.config.config_loader import ConfigLoader

# DKI Plugin (标准集成)
import asyncio
from dki.core.dki_plugin import DKIPlugin, DKIPluginResponse, InjectionMetadata

# Experiment Store (独立持久化层, 使用 dki.db)
from dki.experiment.store import (
    IChatStore,
    SQLiteChatStore,
    ExperimentDBConfig,
    DemoUser,
    DemoSession,
    DemoMessage,
    DemoPreference,
    create_experiment_store,
)

# Experiment Bridge (ConfigDrivenAdapter 配置生成)
from dki.experiment.dki_bridge import build_experiment_adapter_config

# Old database (仅用于 experiments / experiment_results 表)
from dki.database.connection import DatabaseManager
from dki.database.repository import ExperimentRepository


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str = ""
    modes: List[str] = field(default_factory=lambda: ["rag", "dki", "baseline"])
    datasets: List[str] = field(default_factory=lambda: ["persona_chat", "memory_qa"])
    max_samples: int = 100
    max_new_tokens: int = 2048
    temperature: float = 0.7
    alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    force_alpha: float = 0.4
    fact_retrieve_method: str = "auto"
    
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
            'force_alpha': self.force_alpha,
            'fact_retrieve_method': self.fact_retrieve_method,
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
    rag_context: Optional[str] = None
    
    # 最终发送给模型的输入
    final_input: str = ""
    
    # 注入参数
    alpha: float = 0.0
    
    # Entropy-Gated 元认知检索信息
    fact_retrieve_method: str = "post_hoc"
    entropy_triggered: bool = False
    entropy_probe_tokens: int = 0
    entropy_grounding_facts: List[str] = field(default_factory=list)
    entropy_stages: int = 1
    entropy_spike_position: int = -1
    entropy_max_value: float = 0.0
    
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
            'fact_retrieve_method': self.fact_retrieve_method,
            'entropy_triggered': self.entropy_triggered,
            'entropy_probe_tokens': self.entropy_probe_tokens,
            'entropy_grounding_facts': self.entropy_grounding_facts,
            'entropy_stages': self.entropy_stages,
            'entropy_spike_position': self.entropy_spike_position,
            'entropy_max_value': self.entropy_max_value,
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
            lines.append(f"【事实检索方法】{self.fact_retrieve_method}")
            lines.append(f"")
            
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
            
            if self.entropy_triggered:
                lines.append(f"【Entropy-Gated 元认知检索】")
                lines.append(f"───────────────────────────────────────────────────────")
                lines.append(f"  触发: ✅ (Stage {self.entropy_stages})")
                lines.append(f"  探测 tokens: {self.entropy_probe_tokens}")
                lines.append(f"  尖峰位置: token #{self.entropy_spike_position}")
                lines.append(f"  最大熵值: {self.entropy_max_value:.3f} nats")
                if self.entropy_grounding_facts:
                    lines.append(f"  检索事实 ({len(self.entropy_grounding_facts)} 条):")
                    for i, fact in enumerate(self.entropy_grounding_facts, 1):
                        lines.append(f"    [{i}] {fact[:150]}{'...' if len(fact) > 150 else ''}")
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
    
    v9.1 重构:
    - 使用独立的 dki.experiment.store (从 demo.store 复制, 仅 SQLite)
    - 使用独立的 dki.db 数据库 (默认 ./data/dki.db, 不与 demo.db 共享)
    - 使用 dki.integration.create_plugin + ConfigDrivenAdapter 集成 DKIPlugin
    - 注入明文通过 InjectionMetadata 正式字段获取
    
    架构:
      ExperimentRunner (上层应用)
        ├── SQLiteChatStore (读写 demo_* 表, 独立 dki.db)
        │     ├── demo_users
        │     ├── demo_sessions
        │     ├── demo_messages
        │     └── demo_preferences
        └── DKIPlugin (通过 ConfigDrivenAdapter 只读 demo_* 表)
              └── InjectionPlanner → 完整注入计划
    """
    
    def __init__(
        self,
        dki_plugin: Optional[DKIPlugin] = None,
        rag_system: Optional[RAGSystem] = None,
        model_adapter: Optional[Any] = None,
        output_dir: str = "./experiment_results",
        db_path: Optional[str] = None,
    ):
        """
        初始化实验运行器。
        
        Args:
            dki_plugin: 外部 DKI 插件实例 (忽略, v9.1 始终自行创建)
            rag_system: RAG 系统实例 (可选，默认自动创建)
            model_adapter: LLM 模型适配器 (用于创建 DKIPlugin 和 baseline)
            output_dir: 实验结果输出目录
            db_path: 实验数据库路径 (可选, 默认 ./data/dki.db)
        """
        self.config = ConfigLoader().config
        
        # v9.1: 忽略外部 dki_plugin, 始终通过 create_plugin 自行创建
        self._dki_plugin: Optional[DKIPlugin] = None
        self.rag_system = rag_system
        self._model_adapter = model_adapter
        self.metrics = MetricsCalculator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验数据库路径 (独立 dki.db, 不与 demo.db 共享)
        self._db_path = db_path or "./data/dki.db"
        
        # v9.1: Experiment Store (独立持久化层)
        self._store: Optional[IChatStore] = None
        self._db_config: Optional[ExperimentDBConfig] = None
        
        # Old database manager (仅用于 experiments / experiment_results 表)
        self.db_manager = DatabaseManager(db_path=self.config.database.path)
    
    def _ensure_systems(self):
        """
        确保 DKI Plugin / RAG / Store 全部初始化。
        
        v9.1 架构:
        1. 创建 SQLiteChatStore (操作 demo_* 表, 使用独立 dki.db)
        2. 创建 ConfigDrivenAdapter 配置 (映射 demo_* 表)
        3. 通过 create_plugin(adapter_config=...) 创建 DKIPlugin
        4. DKIPlugin 通过 ConfigDrivenAdapter 只读 demo_* 表
        """
        # ============ Step 1: 创建 Store ============
        if self._store is None:
            self._db_config = ExperimentDBConfig(
                backend="sqlite",
                sqlite_path=self._db_path,
            )
            self._store = create_experiment_store(self._db_config)
            logger.info(f"Experiment store created: SQLiteChatStore ({self._db_path})")
        
        # ============ Step 2: 创建 DKIPlugin ============
        if self._dki_plugin is None:
            # Model adapter
            if self._model_adapter is None:
                from dki.models.factory import ModelFactory
                self._model_adapter = ModelFactory.get_or_create()
            
            # Build ConfigDrivenAdapter config (映射 demo_* 表)
            adapter_config = build_experiment_adapter_config(self._db_config)
            
            # 使用标准 create_plugin 工厂 (与 demo 一致)
            self._dki_plugin = self._run_async_safe(
                self._create_plugin_async(adapter_config)
            )
            
            logger.info("Experiment DKIPlugin created via create_plugin (standard integration)")
        
        # ============ Step 3: 创建 RAG ============
        if self.rag_system is None:
            self.rag_system = RAGSystem(model_adapter=self._model_adapter)
    
    async def _create_plugin_async(self, adapter_config: Dict[str, Any]) -> DKIPlugin:
        """异步创建 DKIPlugin (使用标准 create_plugin 工厂)"""
        from dki.integration import create_plugin
        plugin = await create_plugin(
            adapter_config=adapter_config,
            model_adapter=self._model_adapter,
            language="cn",
        )
        return plugin

    @staticmethod
    def _run_async_safe(coro):
        """
        在任意线程/事件循环环境下安全运行协程，避免 ThreadPoolExecutor 死锁。

        策略:
        1. 若当前线程没有运行中的 event loop → 直接 asyncio.run()
        2. 若当前线程有运行中的 event loop (如 Jupyter/FastAPI) →
           在独立线程中创建全新 event loop 运行，彻底隔离，不共享任何 loop 状态。
        """
        import threading

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # 无运行中的 loop，直接运行
            return asyncio.run(coro)

        # 有运行中的 loop：在独立线程里创建新 loop 运行，避免死锁
        result_holder = [None]
        exc_holder = [None]

        def _thread_target():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result_holder[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                exc_holder[0] = e
            finally:
                new_loop.close()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        t.join()

        if exc_holder[0] is not None:
            raise exc_holder[0]
        return result_holder[0]

    def _run_plugin_chat(self, **kwargs) -> DKIPluginResponse:
        """同步包装 DKIPlugin.chat() 异步调用。"""
        return self._run_async_safe(self._dki_plugin.chat(**kwargs))
    
    @property
    def model(self):
        """获取模型适配器 (用于 baseline 模式)。"""
        if self._model_adapter:
            return self._model_adapter
        if self._dki_plugin:
            return self._dki_plugin.model
        raise RuntimeError("No model adapter available. Call _ensure_systems() first.")
    
    # ========================================================================
    # Store Helper Methods (v9.0: 复用 demo store API)
    # ========================================================================
    
    def _store_add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向 store 添加消息 (与 demo/api/chat.py 一致)"""
        try:
            self._store.add_message(
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to store message (non-critical): {e}")
    
    def _store_ensure_session(self, session_id: str, user_id: str, title: str = "Experiment") -> None:
        """确保 session 存在"""
        existing = self._store.get_session(session_id)
        if not existing:
            self._store.create_session(
                user_id=user_id,
                title=title,
                session_id=session_id,
            )
    
    def _extract_injection_info_from_meta(
        self,
        meta: InjectionMetadata,
        query: str,
    ) -> InjectionInfo:
        """从 InjectionMetadata 提取注入信息 (v9.0: 正式字段, 无 hack)"""
        return InjectionInfo(
            mode='dki',
            original_query=query,
            preference_text=meta.preference_text,
            preference_tokens=meta.preference_tokens,
            history_suffix=meta.history_suffix_text,
            history_tokens=meta.history_tokens,
            history_messages=meta.history_messages or [],
            final_input=meta.final_input or query,
            alpha=meta.alpha,
        )
    
    # ========================================================================
    # Experiment User & Preference Management
    # ========================================================================
    
    def _get_first_experiment_user_id(self) -> str:
        """获取第一个实验用户 ID"""
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            first_username = list(self._experiment_user_map.keys())[0]
            return self._experiment_user_map[first_username]
        return "experiment_user"
    
    def setup_experiment_users(
        self,
        users: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        """
        为实验创建用户并写入偏好到数据库 (v9.0: 使用 demo store API)。
        
        使用 IChatStore.get_or_create_user() 和 IChatStore.add_preference()，
        数据写入 demo_users 和 demo_preferences 表，
        DKIPlugin 通过 ConfigDrivenAdapter 读取。
        """
        if users is None:
            users = self._get_default_experiment_users()
        
        user_map = {}  # username -> user_id
        
        for user_data in users:
            username = user_data["username"]
            display_name = user_data.get("display_name", username)
            
            # 创建或获取用户 (demo_users 表)
            user, created = self._store.get_or_create_user(
                username=username,
                display_name=display_name,
            )
            user_id = user.id
            user_map[username] = user_id
            
            if created:
                logger.info(f"Created experiment user: {username} (id={user_id})")
            else:
                logger.info(f"Found existing experiment user: {username} (id={user_id})")
            
            # 写入偏好 (先清除旧的, 再写入新的)
            preferences = user_data.get("preferences", [])
            if preferences:
                # 软删除该用户的旧偏好
                existing_prefs = self._store.get_preferences(user_id)
                for old_pref in existing_prefs:
                    self._store.delete_preference(old_pref.id)
                
                # 写入新偏好 (demo_preferences 表)
                for pref_data in preferences:
                    self._store.add_preference(
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
        """获取默认实验用户配置。"""
        exp_config = getattr(self.config, 'experiment', None)
        if exp_config and hasattr(exp_config, 'users'):
            config_users = exp_config.users
            if config_users:
                return config_users
        
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
        """从数据项中获取实验用户 ID。"""
        if 'user_id' in item:
            return item['user_id']
        
        if hasattr(self, '_experiment_user_map'):
            username = item.get('experiment_user', item.get('username'))
            if username and username in self._experiment_user_map:
                return self._experiment_user_map[username]
        
        if hasattr(self, '_experiment_user_map') and self._experiment_user_map:
            personas = item.get('personas', [])
            if personas:
                return self._match_user_by_personas(personas)
        
        return default
    
    def _match_user_by_personas(self, personas: List[str]) -> str:
        """根据 personas 关键词匹配最佳实验用户。"""
        if not hasattr(self, '_experiment_user_map') or not self._experiment_user_map:
            return self._get_first_experiment_user_id()
        
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
                pref_words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', pref_text)
                for word in pref_words:
                    if len(word) >= 2 and word in personas_text:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_user = username
        
        if best_user:
            return self._experiment_user_map[best_user]
        
        first_username = list(self._experiment_user_map.keys())[0]
        return self._experiment_user_map[first_username]
    
    def _write_session_preferences(self, user_id: str, personas: List[str]) -> None:
        """
        为特定 session 的 personas 写入用户偏好表 (v9.0: 使用 demo store API)。
        """
        if not personas:
            return
        
        try:
            # 软删除旧偏好
            existing = self._store.get_preferences(user_id)
            for old_pref in existing:
                self._store.delete_preference(old_pref.id)
            
            # 写入新偏好 (demo_preferences 表)
            for idx, persona in enumerate(personas):
                self._store.add_preference(
                    user_id=user_id,
                    preference_text=persona,
                    preference_type="general",
                    priority=10 - idx,
                )
            
            # 清除 DKIPlugin 的偏好文本缓存
            if self._dki_plugin is not None:
                self._dki_plugin.invalidate_preference_text_cache(user_id)
                logger.debug(f"Invalidated preference cache for user {user_id}")
                    
        except Exception as e:
            logger.warning(f"Failed to write session preferences for {user_id}: {e}")
    
    # ========================================================================
    # Core Experiment Methods
    # ========================================================================
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        data_path: Optional[str] = None,
        setup_users: bool = True,
        experiment_users: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a full experiment."""
        self._ensure_systems()
        
        if setup_users:
            self.setup_experiment_users(experiment_users)
        
        logger.info(f"Starting experiment: {config.name}")
        
        # Create experiment record (using old db_manager for experiments table)
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
            if data_path:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = self._load_default_data(config.datasets)
            
            for mode in config.modes:
                logger.info(f"Running mode: {mode}")
                mode_results = self._run_mode(mode, data, config)
                results['results_by_mode'][mode] = mode_results
            
            results['aggregated_metrics'] = self._aggregate_metrics(results['results_by_mode'])
            results['completed_at'] = datetime.now().isoformat()
            
            with self.db_manager.session_scope() as db:
                exp_repo = ExperimentRepository(db)
                exp_repo.update_status(experiment_id, 'completed')
                
                for mode, mode_results in results['results_by_mode'].items():
                    exp_repo.add_result(
                        experiment_id=experiment_id,
                        mode=mode,
                        dataset='combined',
                        metrics=mode_results.get('metrics', {}),
                        sample_count=len(mode_results.get('samples', [])),
                    )
            
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
        
        base_ts = int(time.time())
        
        for idx, item in enumerate(tqdm(samples, desc=f"Running {mode}")):
            session_id = f"exp_{mode}_{base_ts}_{idx}"
            user_id = self._get_experiment_user_id(item)
            
            # Ensure user and session exist in store
            self._store.get_or_create_user(username=f"exp_{user_id}", display_name=f"Experiment User")
            self._store_ensure_session(session_id, user_id, title=f"Experiment {mode}")
            
            # Add memories as messages (DKI reads from demo_messages via ConfigDrivenAdapter)
            memories = item.get('personas', []) + item.get('supporting_facts', [])
            if 'memory' in item:
                memories.append(item['memory'])
            
            for mem in memories:
                if mode == 'dki':
                    # 写入 demo_messages 表 (DKIPlugin 通过 ConfigDrivenAdapter 读取)
                    self._store_add_message(session_id, user_id, 'user', mem)
                elif mode == 'rag':
                    self.rag_system.add_memory(session_id, mem)
            
            # 写入偏好
            if item.get('personas'):
                self._write_session_preferences(user_id, item['personas'])
            
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
                
                # v9.0: 存储对话到 demo_messages 表 (与 demo/api/chat.py 一致)
                if mode == 'dki':
                    self._store_add_message(session_id, user_id, 'user', query)
                    self._store_add_message(
                        session_id, user_id, 'assistant', result.response,
                        metadata={'injection_mode': 'dki', 'alpha': result.alpha},
                    )
        
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
        user_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a single query and capture injection info."""
        user_id = user_id or self._get_first_experiment_user_id()
        try:
            if mode == 'dki':
                exp_force_alpha = getattr(config, 'force_alpha', 0.4)
                
                response = self._run_plugin_chat(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    force_alpha=exp_force_alpha,
                )
                
                meta = response.metadata
                
                # v9.0: 从 InjectionMetadata 正式字段提取注入信息
                injection_info = self._extract_injection_info_from_meta(meta, query)
                
                memories_used_ids = []
                if meta.preferences_count > 0:
                    memories_used_ids.append(f"prefs:{meta.preferences_count}")
                if meta.relevant_history_count > 0:
                    memories_used_ids.append(f"history:{meta.relevant_history_count}")
                
                return ExperimentResult(
                    mode=mode,
                    dataset=item.get('_dataset', 'unknown'),
                    sample_id=item.get('id', item.get('session_id', '')),
                    query=query,
                    response=response.text,
                    latency_ms=meta.latency_ms,
                    memories_used=memories_used_ids,
                    alpha=meta.alpha,
                    cache_hit=meta.preference_cache_hit,
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
                output = self.model.generate(
                    prompt=query,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                
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
        
        alphas = [r.alpha for r in results if r.alpha is not None]
        if alphas:
            metrics['alpha'] = {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'min': float(np.min(alphas)),
                'max': float(np.max(alphas)),
            }
        
        cache_hits = [r.cache_hit for r in results]
        if any(cache_hits):
            metrics['cache_hit_rate'] = sum(cache_hits) / len(cache_hits)
        
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
        
        fabricated_rates = []
        irrelevant_rates = []
        total_halluc_rates = []
        for r in valid_results:
            if r.memories_used:
                decomposed = self.metrics.compute_hallucination_decomposed(
                    response=r.response,
                    grounding_texts=r.memories_used,
                    query=r.query,
                )
                fabricated_rates.append(decomposed['fabricated_rate'])
                irrelevant_rates.append(decomposed['irrelevant_rate'])
                total_halluc_rates.append(decomposed['total_rate'])
        if total_halluc_rates:
            metrics['hallucination'] = {
                'mean_rate': float(np.mean(total_halluc_rates)),
                'std_rate': float(np.std(total_halluc_rates)),
                'fabricated_detail': {
                    'mean_rate': float(np.mean(fabricated_rates)),
                    'std_rate': float(np.std(fabricated_rates)),
                },
                'irrelevant_offtopic': {
                    'mean_rate': float(np.mean(irrelevant_rates)),
                    'std_rate': float(np.std(irrelevant_rates)),
                },
            }
        
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
    
    # ========================================================================
    # α Sensitivity Analysis
    # ========================================================================
    
    def run_alpha_sensitivity(
        self,
        data_path: Optional[str] = None,
        alpha_values: Optional[List[float]] = None,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """Run α sensitivity analysis."""
        self._ensure_systems()
        
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        alpha_values = alpha_values or [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        
        logger.info(f"Running α sensitivity analysis with values: {alpha_values}")
        
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
        
        user_id = self._get_first_experiment_user_id()
        base_ts = int(time.time())
        
        for alpha_idx, alpha in enumerate(alpha_values):
            alpha_results = []
            session_id = f"alpha_exp_{base_ts}_{alpha_idx}"
            
            # Ensure session
            self._store.get_or_create_user(username=f"exp_{user_id}", display_name="Alpha Exp User")
            self._store_ensure_session(session_id, user_id, title=f"Alpha {alpha}")
            
            # Add memories as messages
            for item in data[:50]:
                if 'memory' in item:
                    self._store_add_message(session_id, user_id, 'user', item['memory'])
            
            for item in tqdm(data[:50], desc=f"α={alpha}"):
                query = item.get('query', '')
                if not query:
                    continue
                
                try:
                    response = self._run_plugin_chat(
                        query=query,
                        session_id=session_id,
                        user_id=user_id,
                        force_alpha=alpha,
                    )
                    
                    response_text = response.text
                    
                    relevant = item.get('relevant_memories', item.get('personas', []))
                    recall_score = 0.0
                    if relevant:
                        recall_score, _ = self.metrics.compute_memory_recall(
                            expected_memories=relevant,
                            response=response_text,
                            threshold=0.3,
                        )
                    
                    grounding = relevant if relevant else [query]
                    if 'memory' in item:
                        grounding = grounding + [item['memory']]
                    halluc = self.metrics.compute_hallucination_decomposed(
                        response=response_text,
                        grounding_texts=grounding,
                        query=query,
                    )
                    
                    reference = item.get('reference_answer', '')
                    bleu = 0.0
                    rouge_l = 0.0
                    if reference:
                        bleu = self.metrics.compute_bleu(reference, response_text)
                        rouge_scores = self.metrics.compute_rouge(reference, response_text)
                        rouge_l = rouge_scores.get('rougeL', 0.0)
                    
                    _actual_alpha = response.metadata.alpha
                    alpha_results.append({
                        'query': query,
                        'response': response_text[:300],
                        'latency_ms': response.metadata.latency_ms,
                        'actual_alpha': _actual_alpha,
                        'memory_recall': recall_score,
                        'fabricated_halluc': halluc['fabricated_rate'],
                        'irrelevant_halluc': halluc['irrelevant_rate'],
                        'bleu4': bleu,
                        'rouge_l': rouge_l,
                    })
                except Exception as e:
                    logger.error(f"Alpha sensitivity query failed (α={alpha}): {e}")
            
            import numpy as np
            latencies = [r['latency_ms'] for r in alpha_results]
            recalls = [r['memory_recall'] for r in alpha_results]
            fab_halluc = [r['fabricated_halluc'] for r in alpha_results]
            bleu_scores = [r['bleu4'] for r in alpha_results]
            rouge_scores = [r['rouge_l'] for r in alpha_results]
            
            results['results_by_alpha'][str(alpha)] = {
                'samples': alpha_results,
                'latency_stats': self.metrics.compute_latency_stats(latencies),
                'bleu4_mean': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
                'rouge_l_mean': float(np.mean(rouge_scores)) if rouge_scores else 0.0,
                'memory_recall_mean': float(np.mean(recalls)) if recalls else 0.0,
                'fabricated_halluc_mean': float(np.mean(fab_halluc)) if fab_halluc else 0.0,
            }
        
        summary_table = []
        for alpha in alpha_values:
            key = str(alpha)
            if key not in results['results_by_alpha']:
                continue
            r = results['results_by_alpha'][key]
            summary_table.append({
                'alpha': alpha,
                'bleu4': r['bleu4_mean'],
                'rouge_l': r['rouge_l_mean'],
                'memory_recall': r['memory_recall_mean'],
                'fabricated_halluc': r['fabricated_halluc_mean'],
                'latency_p50': r['latency_stats'].get('p50', 0),
            })
        results['summary_table'] = summary_table
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"alpha_sensitivity_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"α sensitivity results saved to {filepath}")
        for row in summary_table:
            logger.info(
                f"  α={row['alpha']}: BLEU4={row['bleu4']:.3f}, "
                f"ROUGE-L={row['rouge_l']:.3f}, "
                f"Recall={row['memory_recall']:.3f}, "
                f"FabHalluc={row['fabricated_halluc']:.3f}"
            )
        return results
    
    # ========================================================================
    # Latency Comparison
    # ========================================================================
    
    def run_latency_comparison(
        self,
        n_turns: int = 10,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """Run latency comparison between first turn and subsequent turns."""
        self._ensure_systems()
        
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info(f"Running latency comparison with {n_turns} turns")
        
        session_id = f"latency_exp_{int(time.time())}"
        user_id = self._get_first_experiment_user_id()
        
        # Ensure session
        self._store_ensure_session(session_id, user_id, title="Latency Experiment")
        
        # Add some memories as messages
        memories = [
            "User prefers vegetarian food.",
            "User lives in Beijing.",
            "User enjoys hiking.",
        ]
        for mem in memories:
            self._store_add_message(session_id, user_id, 'user', mem)
        
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
            response = self._run_plugin_chat(
                query=query,
                session_id=session_id,
                user_id=user_id,
            )
            results['dki_latencies'].append({
                'turn': i + 1,
                'latency_ms': response.metadata.latency_ms,
                'cache_hit': response.metadata.preference_cache_hit,
            })
            # Store conversation
            self._store_add_message(session_id, user_id, 'user', query)
            self._store_add_message(session_id, user_id, 'assistant', response.text)
        
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
        
        import numpy as np
        dki_first = results['dki_latencies'][0]['latency_ms'] if results['dki_latencies'] else 0
        dki_subsequent = [r['latency_ms'] for r in results['dki_latencies'][1:]]
        rag_all = [r['latency_ms'] for r in results['rag_latencies']]
        
        results['summary'] = {
            'dki_first_turn': dki_first,
            'dki_subsequent_mean': float(np.mean(dki_subsequent)) if dki_subsequent else 0,
            'rag_mean': float(np.mean(rag_all)) if rag_all else 0,
            'speedup_subsequent': (
                float(np.mean(rag_all)) / float(np.mean(dki_subsequent))
                if dki_subsequent and np.mean(dki_subsequent) > 0 else 0
            ),
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"latency_comparison_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Latency comparison results saved to {filepath}")
        return results
    
    # ========================================================================
    # Multi-Turn Coherence
    # ========================================================================
    
    def run_multi_turn_coherence(
        self,
        data_path: Optional[str] = None,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """运行多轮连贯性实验"""
        self._ensure_systems()
        
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info("Running multi-turn coherence experiment")
        
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
            for session_data in tqdm(data[:20], desc=f"Coherence ({mode})"):
                session_id = f"coherence_{mode}_{session_data['session_id']}"
                user_id = self._get_experiment_user_id(session_data)
                
                self._write_session_preferences(user_id, session_data.get('personas', []))
                
                # Ensure session
                self._store_ensure_session(session_id, user_id, title=f"Coherence {mode}")
                
                # Add personas as memories
                for mem in session_data['personas']:
                    if mode == 'dki':
                        self._store_add_message(session_id, user_id, 'user', mem)
                    else:
                        self.rag_system.add_memory(session_id, mem)
                
                session_results = []
                
                for turn_idx, turn in enumerate(session_data['turns']):
                    query = turn['query']
                    
                    if mode == 'dki':
                        response = self._run_plugin_chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                        
                        # Store conversation (v9.0: demo store API)
                        self._store_add_message(session_id, user_id, 'user', query)
                        self._store_add_message(
                            session_id, user_id, 'assistant', response_text,
                            metadata={'injection_mode': 'dki', 'alpha': response.metadata.alpha},
                        )
                    else:
                        response = self.rag_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                    
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
                    
                    turn_key = f"turn_{turn_idx}"
                    if turn_key not in results[mode]['per_turn_recall']:
                        results[mode]['per_turn_recall'][turn_key] = []
                    if turn.get('tests_memory'):
                        results[mode]['per_turn_recall'][turn_key].append(recall_score)
                
                results[mode]['sessions'].append({
                    'session_id': session_data['session_id'],
                    'turns': session_results,
                })
        
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
            
            all_recalls = [
                s for scores in per_turn.values() for s in scores
            ]
            results[mode]['overall_recall'] = float(np.mean(all_recalls)) if all_recalls else 0.0
        
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
    
    # ========================================================================
    # PersonaChat Experiment
    # ========================================================================
    
    def run_persona_chat_experiment(
        self,
        data_path: Optional[str] = None,
        include_long_sessions: bool = True,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """运行 PersonaChat 实验 (短会话 + 长会话)"""
        self._ensure_systems()
        
        if setup_users:
            self.setup_experiment_users()
        
        logger.info("Running PersonaChat experiment (short + long sessions)")
        
        short_data = []
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                short_data = json.load(f)
        else:
            data_file = Path("./data/persona_chat.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    short_data = json.load(f)
            else:
                logger.warning("No persona_chat data found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                short_data = gen.generate_persona_chat()
        
        long_data = []
        if include_long_sessions:
            long_data_file = Path("./data/long_session_persona_chat.json")
            if long_data_file.exists():
                with open(long_data_file, 'r', encoding='utf-8') as f:
                    long_data = json.load(f)
            else:
                logger.warning("No long session data found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                long_data = gen.generate_long_session_persona_chat()
        
        results = {
            'short_sessions': {'dki': [], 'rag': []},
            'long_sessions': {'dki': [], 'rag': []},
            'summary': {},
        }
        
        logger.info(f"Running short sessions ({len(short_data[:20])} sessions)")
        for mode in ['dki', 'rag']:
            for session_data in tqdm(short_data[:20], desc=f"Short ({mode})"):
                session_result = self._run_session(mode, session_data, session_type='short')
                results['short_sessions'][mode].append(session_result)
        
        if long_data:
            logger.info(f"Running long sessions ({len(long_data[:10])} sessions)")
            for mode in ['dki', 'rag']:
                for session_data in tqdm(long_data[:10], desc=f"Long ({mode})"):
                    session_result = self._run_session(mode, session_data, session_type='long')
                    results['long_sessions'][mode].append(session_result)
        
        import numpy as np
        for session_type in ['short_sessions', 'long_sessions']:
            for mode in ['dki', 'rag']:
                sessions = results[session_type][mode]
                if not sessions:
                    continue
                
                all_latencies = []
                all_recall_scores = []
                total_turns = 0
                
                for s in sessions:
                    for t in s.get('turns', []):
                        all_latencies.append(t.get('latency_ms', 0))
                        if t.get('recall_score') is not None:
                            all_recall_scores.append(t['recall_score'])
                        total_turns += 1
                
                key = f"{session_type}_{mode}"
                results['summary'][key] = {
                    'session_count': len(sessions),
                    'total_turns': total_turns,
                    'mean_latency_ms': float(np.mean(all_latencies)) if all_latencies else 0,
                    'p95_latency_ms': float(np.percentile(all_latencies, 95)) if all_latencies else 0,
                    'mean_recall': float(np.mean(all_recall_scores)) if all_recall_scores else 0,
                }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"persona_chat_experiment_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"PersonaChat experiment results saved to {filepath}")
        for key, summary in results['summary'].items():
            logger.info(
                f"  {key}: sessions={summary['session_count']}, "
                f"turns={summary['total_turns']}, "
                f"latency={summary['mean_latency_ms']:.1f}ms, "
                f"recall={summary['mean_recall']:.3f}"
            )
        
        return results
    
    def _run_session(
        self,
        mode: str,
        session_data: Dict[str, Any],
        session_type: str = 'short',
    ) -> Dict[str, Any]:
        """运行单个会话 (短或长) — v9.0: 使用 demo store API"""
        session_id = f"exp_{mode}_{session_type}_{session_data.get('session_id', int(time.time()))}"
        user_id = self._get_experiment_user_id(session_data)
        
        # Ensure session
        self._store_ensure_session(session_id, user_id, title=f"PersonaChat {session_type}")
        
        self._write_session_preferences(user_id, session_data.get('personas', []))
        
        for mem in session_data.get('personas', []):
            if mode == 'dki':
                self._store_add_message(session_id, user_id, 'user', mem)
            else:
                self.rag_system.add_memory(session_id, mem)
        
        turn_results = []
        
        for turn_idx, turn_data in enumerate(session_data.get('turns', [])):
            query = turn_data.get('query', '')
            if not query:
                continue
            
            try:
                start_time = time.time()
                
                if mode == 'dki':
                    response = self._run_plugin_chat(
                        query=query,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    response_text = response.text
                    latency = response.metadata.latency_ms
                    
                    meta = response.metadata
                    injection_info = self._extract_injection_info_from_meta(meta, query)
                    
                    # Store conversation
                    self._store_add_message(session_id, user_id, 'user', query)
                    self._store_add_message(
                        session_id, user_id, 'assistant', response_text,
                        metadata={'injection_mode': 'dki', 'alpha': meta.alpha},
                    )
                else:
                    response = self.rag_system.chat(
                        query=query,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    response_text = response.text
                    latency = response.latency_ms
                    
                    prompt_info = response.prompt_info
                    injection_info = InjectionInfo(
                        mode='rag',
                        original_query=query,
                        rag_context=prompt_info.retrieved_context if prompt_info else None,
                        rag_prompt=prompt_info.final_prompt if prompt_info else None,
                        final_input=prompt_info.final_prompt if prompt_info else query,
                    )
                
                recall_score = None
                expected_keywords = turn_data.get('expected_keywords', [])
                if expected_keywords:
                    response_lower = response_text.lower()
                    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
                    recall_score = hits / len(expected_keywords) if expected_keywords else 0.0
                
                turn_results.append({
                    'turn_idx': turn_idx,
                    'query': query,
                    'response': response_text[:500],
                    'latency_ms': latency,
                    'recall_score': recall_score,
                    'expected_keywords': expected_keywords,
                    'injection_info': injection_info.to_dict(),
                })
                
            except Exception as e:
                logger.error(f"Session turn failed: {e}")
                turn_results.append({
                    'turn_idx': turn_idx,
                    'query': query,
                    'response': f"ERROR: {e}",
                    'latency_ms': 0,
                    'recall_score': None,
                })
        
        return {
            'session_id': session_id,
            'session_type': session_type,
            'user_id': user_id,
            'experiment_user': session_data.get('experiment_user', ''),
            'personas': session_data.get('personas', []),
            'turns': turn_results,
            'turn_count': len(turn_results),
        }
    
    # ========================================================================
    # Ablation Study
    # ========================================================================
    
    def run_ablation_study(
        self,
        data_path: Optional[str] = None,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """运行消融实验 — 对齐论文 Table 3"""
        self._ensure_systems()
        
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        logger.info("Running ablation study (7 variants aligned with paper Table 3)")
        
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
            'full_dki': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'entropy_gated',
                'use_kv_injection': True,
            },
            'wo_fact_call': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'post_hoc',
                'use_kv_injection': True,
            },
            'wo_multi_signal': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'entropy_gated',
                'recall_mode': 'vector_only',
                'use_kv_injection': True,
            },
            'wo_kv_injection': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'entropy_gated',
                'use_kv_injection': False,
            },
            'stable_fallback_only': {
                'system': 'dki', 'force_alpha': 0.4, 'use_memory': True,
                'fact_retrieve_method': 'post_hoc',
                'recall_mode': 'stable',
                'use_kv_injection': True,
            },
            'rag_baseline': {
                'system': 'rag', 'force_alpha': None, 'use_memory': True,
                'use_kv_injection': False,
            },
            'vanilla_llm': {
                'system': 'baseline', 'force_alpha': None, 'use_memory': False,
                'use_kv_injection': False,
            },
        }
        
        results = {mode: {'samples': [], 'latencies': []} for mode in ablation_configs}
        
        user_id = self._get_first_experiment_user_id()
        
        for ablation_mode, config in ablation_configs.items():
            logger.info(f"Running ablation: {ablation_mode}")
            
            session_id = f"ablation_{ablation_mode}_{int(time.time())}"
            self._store_ensure_session(session_id, user_id, title=f"Ablation {ablation_mode}")
            
            if config.get('use_memory'):
                if config['system'] == 'rag':
                    for item in data[:30]:
                        if 'memory' in item:
                            self.rag_system.add_memory(session_id, item['memory'])
                elif config['system'] == 'dki':
                    for item in data[:30]:
                        if 'memory' in item:
                            self._store_add_message(session_id, user_id, 'user', item['memory'])
            
            for item in tqdm(data[:30], desc=f"Ablation ({ablation_mode})"):
                query = item.get('query', '')
                if not query:
                    continue
                
                try:
                    response_text = ""
                    latency = 0.0
                    
                    if config['system'] == 'dki':
                        dki_kwargs = {
                            'query': query,
                            'session_id': session_id,
                            'user_id': user_id,
                            'force_alpha': config.get('force_alpha'),
                        }
                        if not config.get('use_kv_injection'):
                            dki_kwargs['force_alpha'] = 0.0
                        
                        if config.get('fact_retrieve_method'):
                            dki_kwargs['fact_retrieve_method'] = config['fact_retrieve_method']
                        
                        response = self._run_plugin_chat(**dki_kwargs)
                        response_text = response.text
                        latency = response.metadata.latency_ms
                        
                    elif config['system'] == 'rag':
                        response = self.rag_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                        )
                        response_text = response.text
                        latency = response.latency_ms
                        
                    else:  # baseline
                        output = self.model.generate(
                            prompt=query,
                            max_new_tokens=2048,
                            temperature=0.7,
                        )
                        response_text = output.text
                        latency = output.latency_ms
                    
                    relevant = item.get('relevant_memories', [])
                    recall_score = 0.0
                    if relevant:
                        recall_score, _ = self.metrics.compute_memory_recall(
                            expected_memories=relevant,
                            response=response_text,
                            threshold=0.3,
                        )
                    
                    grounding = item.get('relevant_memories', [])
                    if 'memory' in item:
                        grounding = grounding + [item['memory']]
                    halluc = self.metrics.compute_hallucination_decomposed(
                        response=response_text,
                        grounding_texts=grounding if grounding else [query],
                        query=query,
                    )
                    
                    reference = item.get('reference_answer', '')
                    bleu = 0.0
                    rouge_l = 0.0
                    if reference:
                        bleu = self.metrics.compute_bleu(reference, response_text)
                        rouge_scores = self.metrics.compute_rouge(reference, response_text)
                        rouge_l = rouge_scores.get('rougeL', 0.0)
                    
                    sample_record = {
                        'query': query,
                        'response': response_text[:500],
                        'latency_ms': latency,
                        'memory_recall': recall_score,
                        'fabricated_halluc': halluc['fabricated_rate'],
                        'irrelevant_halluc': halluc['irrelevant_rate'],
                        'total_halluc': halluc['total_rate'],
                        'bleu4': bleu,
                        'rouge_l': rouge_l,
                    }
                    
                    results[ablation_mode]['samples'].append(sample_record)
                    results[ablation_mode]['latencies'].append(latency)
                    
                except Exception as e:
                    logger.error(f"Ablation query failed ({ablation_mode}): {e}")
        
        import numpy as np
        summary = {}
        for mode, mode_results in results.items():
            if mode == 'summary':
                continue
            samples = mode_results['samples']
            latencies = mode_results['latencies']
            
            recalls = [s['memory_recall'] for s in samples]
            fab_rates = [s['fabricated_halluc'] for s in samples]
            irr_rates = [s['irrelevant_halluc'] for s in samples]
            total_halluc = [s['total_halluc'] for s in samples]
            bleu_scores = [s['bleu4'] for s in samples]
            rouge_scores = [s['rouge_l'] for s in samples]
            
            summary[mode] = {
                'sample_count': len(samples),
                'memory_recall': float(np.mean(recalls)) if recalls else 0.0,
                'fabricated_halluc_rate': float(np.mean(fab_rates)) if fab_rates else 0.0,
                'irrelevant_halluc_rate': float(np.mean(irr_rates)) if irr_rates else 0.0,
                'total_halluc_rate': float(np.mean(total_halluc)) if total_halluc else 0.0,
                'bleu4_mean': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
                'rouge_l_mean': float(np.mean(rouge_scores)) if rouge_scores else 0.0,
                'mean_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
                'p95_latency_ms': float(np.percentile(latencies, 95)) if latencies else 0.0,
            }
        
        results['summary'] = summary
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"ablation_study_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ablation study results saved to {filepath}")
        for mode, s in summary.items():
            logger.info(
                f"  {mode}: recall={s['memory_recall']:.3f}, "
                f"fab_halluc={s['fabricated_halluc_rate']:.3f}, "
                f"latency={s['mean_latency_ms']:.1f}ms"
            )
        
        return results
    
    # ========================================================================
    # LongMemEval Benchmark
    # ========================================================================
    
    def run_longmemeval(
        self,
        modes: Optional[List[str]] = None,
        longmemeval_modes: Optional[List[str]] = None,
        max_samples: int = 50,
        max_new_tokens: int = 2048,
        force_alpha: float = 0.4,
        setup_users: bool = True,
        auto_generate: bool = True,
        longmemeval_source: str = "../longmem/longmemeval_s_cleaned.json",
    ) -> Dict[str, Any]:
        """运行 LongMemEval 基准测试"""
        self._ensure_systems()
        
        if setup_users:
            longmem_users = self._get_default_experiment_users() + [{
                "username": "exp_user_longmem",
                "display_name": "LongMemEval Test User",
                "preferences": [
                    {"text": "I am a user participating in long-term memory evaluation", "type": "general", "priority": 5},
                ],
            }]
            self.setup_experiment_users(longmem_users)
        
        if modes is None:
            modes = ["dki", "rag", "baseline"]
        if longmemeval_modes is None:
            longmemeval_modes = ["multi_turn", "needle"]
        
        logger.info(
            f"Running LongMemEval benchmark: modes={modes}, "
            f"longmem_modes={longmemeval_modes}, max_samples={max_samples}"
        )
        
        results = {
            'benchmark': 'longmemeval',
            'started_at': datetime.now().isoformat(),
            'config': {
                'modes': modes,
                'longmemeval_modes': longmemeval_modes,
                'max_samples': max_samples,
                'max_new_tokens': max_new_tokens,
                'force_alpha': force_alpha,
            },
            'results_by_dataset': {},
            'summary': {},
        }
        
        for lm_mode in longmemeval_modes:
            dataset_name = f"longmemeval_{lm_mode}"
            data_file = Path(f"./data/{dataset_name}.json")
            
            if not data_file.exists() and auto_generate:
                logger.info(f"Data file {data_file} not found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                gen.generate_longmemeval(
                    source_path=longmemeval_source,
                    mode=lm_mode,
                    output_name=dataset_name,
                )
            
            if not data_file.exists():
                logger.error(f"Data file {data_file} not found, skipping {lm_mode}")
                continue
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} items for LongMemEval-{lm_mode}")
            samples = data[:max_samples]
            
            dataset_results = {}
            
            for mode in modes:
                logger.info(f"  Running {mode} on {dataset_name} ({len(samples)} samples)")
                mode_results = self._run_longmemeval_mode(
                    mode=mode,
                    samples=samples,
                    max_new_tokens=max_new_tokens,
                    force_alpha=force_alpha,
                )
                dataset_results[mode] = mode_results
            
            results['results_by_dataset'][dataset_name] = dataset_results
        
        results['summary'] = self._summarize_longmemeval(results['results_by_dataset'])
        results['completed_at'] = datetime.now().isoformat()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"longmemeval_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"LongMemEval results saved to {filepath}")
        self._print_longmemeval_summary(results['summary'])
        
        return results
    
    def _run_longmemeval_mode(
        self,
        mode: str,
        samples: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
        force_alpha: float = 0.4,
    ) -> Dict[str, Any]:
        """运行 LongMemEval 单模式评估。"""
        eval_results = []
        base_ts = int(time.time())
        
        for idx, item in enumerate(tqdm(samples, desc=f"LongMemEval ({mode})")):
            session_id = f"longmem_{mode}_{base_ts}_{idx}"
            user_id = self._get_experiment_user_id(item)
            
            # Ensure session
            self._store_ensure_session(session_id, user_id, title=f"LongMemEval {mode}")
            
            personas = item.get('personas', [])
            if personas:
                self._write_session_preferences(user_id, personas)
            
            # Add personas as messages
            if mode == 'dki':
                for mem in personas:
                    self._store_add_message(session_id, user_id, 'user', mem)
            elif mode == 'rag':
                for mem in personas:
                    self.rag_system.add_memory(session_id, mem)
            
            turns = item.get('turns', [])
            eval_turn = None
            history_turns = []
            
            for t in turns:
                if t.get('is_eval_query'):
                    eval_turn = t
                else:
                    history_turns.append(t)
            
            if not eval_turn:
                logger.warning(f"Sample {idx} has no eval query, skipping")
                continue
            
            # Write history turns to DB
            history_injected = 0
            for h_turn in history_turns:
                query = h_turn.get('query', '')
                expected_resp = h_turn.get('expected_response', '')
                if not query:
                    continue
                
                try:
                    if mode == 'dki':
                        self._store_add_message(session_id, user_id, 'user', query)
                        if expected_resp:
                            self._store_add_message(session_id, user_id, 'assistant', expected_resp)
                        history_injected += 1
                    elif mode == 'rag':
                        resp = self.rag_system.chat(
                            query=query,
                            session_id=session_id,
                            user_id=user_id,
                            max_new_tokens=32,
                            temperature=0.1,
                        )
                        history_injected += 1
                except Exception as e:
                    logger.debug(f"History turn failed: {e}")
            
            # Send eval question
            eval_query = eval_turn['query']
            expected_answer = eval_turn.get('expected_answer', '')
            expected_keywords = eval_turn.get('expected_keywords', [])
            
            try:
                injection_info = {}
                
                if mode == 'dki':
                    response = self._run_plugin_chat(
                        query=eval_query,
                        session_id=session_id,
                        user_id=user_id,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        force_alpha=force_alpha,
                    )
                    response_text = response.text
                    latency = response.metadata.latency_ms
                    pref_alpha = response.metadata.alpha
                    
                    meta = response.metadata
                    injection_info = {
                        'injection_enabled': meta.injection_enabled,
                        'injection_strategy': meta.injection_strategy,
                        'preferences_count': meta.preferences_count,
                        'preference_tokens': meta.preference_tokens,
                        'preference_text': meta.preference_text or '',
                        'relevant_history_count': meta.relevant_history_count,
                        'history_tokens': meta.history_tokens,
                        'history_suffix_text': meta.history_suffix_text or '',
                        'history_messages': meta.history_messages or [],
                        'final_input': meta.final_input or eval_query,
                        'total_tokens': meta.total_tokens,
                        'retrieval_mode': meta.retrieval_mode,
                        'alpha': meta.alpha,
                        'preference_cache_hit': meta.preference_cache_hit,
                        'preference_cache_tier': meta.preference_cache_tier,
                        'adapter_latency_ms': meta.adapter_latency_ms,
                        'inference_latency_ms': meta.inference_latency_ms,
                    }
                    
                    # Store eval conversation
                    self._store_add_message(session_id, user_id, 'user', eval_query)
                    self._store_add_message(session_id, user_id, 'assistant', response_text)
                    
                elif mode == 'rag':
                    response = self.rag_system.chat(
                        query=eval_query,
                        session_id=session_id,
                        user_id=user_id,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                    )
                    response_text = response.text
                    latency = response.latency_ms
                    pref_alpha = 0.0
                    
                else:  # baseline
                    output = self.model.generate(
                        prompt=eval_query,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                    )
                    response_text = output.text
                    latency = output.latency_ms
                    pref_alpha = 0.0
                
                resp_lower = response_text.lower()
                kw_hits = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
                keyword_recall = kw_hits / len(expected_keywords) if expected_keywords else 0.0
                
                answer_match = 0.0
                if expected_answer:
                    answer_words = re.findall(r'\b\w+\b', expected_answer.lower())
                    answer_words = [w for w in answer_words if len(w) > 2]
                    if answer_words:
                        hits = sum(1 for w in answer_words if w in resp_lower)
                        answer_match = hits / len(answer_words)
                
                rouge_l = 0.0
                if expected_answer:
                    try:
                        rouge_scores = self.metrics.compute_rouge(expected_answer, response_text)
                        rouge_l = rouge_scores.get('rougeL', 0.0)
                    except Exception:
                        pass
                
                eval_results.append({
                    'sample_idx': idx,
                    'session_id': item.get('session_id', ''),
                    'question_type': item.get('metadata', {}).get('question_type', ''),
                    'question_id': item.get('metadata', {}).get('question_id', ''),
                    'eval_query': eval_query,
                    'expected_answer': expected_answer,
                    'response': response_text[:500],
                    'latency_ms': latency,
                    'keyword_recall': keyword_recall,
                    'answer_match': answer_match,
                    'rouge_l': rouge_l,
                    'alpha': pref_alpha,
                    'history_turns_played': history_injected,
                    'total_turns': len(turns),
                    'injection_info': injection_info,
                })
                
            except Exception as e:
                logger.error(f"LongMemEval eval query failed: {e}")
                eval_results.append({
                    'sample_idx': idx,
                    'session_id': item.get('session_id', ''),
                    'question_type': item.get('metadata', {}).get('question_type', ''),
                    'eval_query': eval_query,
                    'expected_answer': expected_answer,
                    'response': f"ERROR: {e}",
                    'latency_ms': 0,
                    'keyword_recall': 0.0,
                    'answer_match': 0.0,
                    'rouge_l': 0.0,
                    'alpha': 0.0,
                    'history_turns_played': history_injected,
                    'total_turns': len(turns),
                })
        
        import numpy as np
        valid = [r for r in eval_results if not r['response'].startswith('ERROR:')]
        
        metrics = {
            'total_samples': len(eval_results),
            'valid_samples': len(valid),
            'error_count': len(eval_results) - len(valid),
        }
        
        if valid:
            metrics.update({
                'keyword_recall_mean': float(np.mean([r['keyword_recall'] for r in valid])),
                'answer_match_mean': float(np.mean([r['answer_match'] for r in valid])),
                'rouge_l_mean': float(np.mean([r['rouge_l'] for r in valid])),
                'latency_mean_ms': float(np.mean([r['latency_ms'] for r in valid])),
                'latency_p50_ms': float(np.median([r['latency_ms'] for r in valid])),
                'latency_p95_ms': float(np.percentile([r['latency_ms'] for r in valid], 95)),
                'avg_history_turns': float(np.mean([r['history_turns_played'] for r in valid])),
            })
            
            by_type = {}
            for r in valid:
                qt = r.get('question_type', 'unknown')
                if qt not in by_type:
                    by_type[qt] = []
                by_type[qt].append(r)
            
            metrics['by_question_type'] = {}
            for qt, items in by_type.items():
                metrics['by_question_type'][qt] = {
                    'count': len(items),
                    'keyword_recall': float(np.mean([r['keyword_recall'] for r in items])),
                    'answer_match': float(np.mean([r['answer_match'] for r in items])),
                    'rouge_l': float(np.mean([r['rouge_l'] for r in items])),
                }
        
        return {
            'mode': mode if mode != 'baseline' else 'baseline',
            'samples': eval_results,
            'metrics': metrics,
        }
    
    def _summarize_longmemeval(
        self,
        results_by_dataset: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """汇总 LongMemEval 所有数据集和模式的结果。"""
        summary = {}
        
        for dataset_name, mode_results in results_by_dataset.items():
            summary[dataset_name] = {}
            for mode, result in mode_results.items():
                m = result.get('metrics', {})
                summary[dataset_name][mode] = {
                    'keyword_recall': m.get('keyword_recall_mean', 0.0),
                    'answer_match': m.get('answer_match_mean', 0.0),
                    'rouge_l': m.get('rouge_l_mean', 0.0),
                    'latency_p50': m.get('latency_p50_ms', 0.0),
                    'valid_samples': m.get('valid_samples', 0),
                }
        
        return summary
    
    def _print_longmemeval_summary(self, summary: Dict[str, Any]):
        """打印 LongMemEval 汇总结果。"""
        for dataset_name, modes in summary.items():
            logger.info(f"\n=== {dataset_name} ===")
            for mode, metrics in modes.items():
                logger.info(
                    f"  {mode:>10s}: "
                    f"kw_recall={metrics['keyword_recall']:.3f}, "
                    f"ans_match={metrics['answer_match']:.3f}, "
                    f"rouge_l={metrics['rouge_l']:.3f}, "
                    f"latency_p50={metrics['latency_p50']:.0f}ms, "
                    f"n={metrics['valid_samples']}"
                )
    
    # ========================================================================
    # Context-Constrained Experiment
    # ========================================================================
    
    def run_context_constrained(
        self,
        data_path: Optional[str] = None,
        memory_lengths: Optional[List[int]] = None,
        context_budget: int = 4096,
        setup_users: bool = True,
    ) -> Dict[str, Any]:
        """运行上下文受限实验 — 对齐论文 Table 2"""
        self._ensure_systems()
        
        if setup_users and not hasattr(self, '_experiment_user_map'):
            self.setup_experiment_users()
        
        if memory_lengths is None:
            memory_lengths = [500, 1000, 1500, 2000, 2500, 3000, 3500]
        
        logger.info(
            f"Running context-constrained experiment "
            f"(budget={context_budget}, lengths={memory_lengths})"
        )
        
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data_file = Path("./data/context_constrained.json")
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning("No context-constrained data found, generating...")
                from dki.experiment.data_generator import ExperimentDataGenerator
                gen = ExperimentDataGenerator("./data")
                data = gen.generate_context_constrained_data(memory_lengths=memory_lengths)
        
        user_id = self._get_first_experiment_user_id()
        
        results = {
            'context_budget': context_budget,
            'memory_lengths': memory_lengths,
            'results_by_length': {},
        }
        
        for mem_length in memory_lengths:
            length_samples = [
                d for d in data
                if d.get('memory_length_tokens') == mem_length
            ]
            
            if not length_samples:
                logger.warning(f"No samples for memory_length={mem_length}, skipping")
                continue
            
            length_results = {'dki': [], 'rag': []}
            
            for mode in ['dki', 'rag']:
                for sample in tqdm(
                    length_samples[:30],
                    desc=f"Ctx-{mem_length} ({mode})",
                ):
                    session_id = f"ctx_{mode}_{mem_length}_{sample['id']}_{int(time.time())}"
                    self._store_ensure_session(session_id, user_id, title=f"Ctx {mem_length}")
                    
                    for frag in sample.get('memory_fragments', []):
                        if mode == 'dki':
                            self._store_add_message(session_id, user_id, 'user', frag)
                        else:
                            self.rag_system.add_memory(session_id, frag)
                    
                    prefs = sample.get('memory_fragments', [])[:5]
                    self._write_session_preferences(user_id, prefs)
                    
                    query = sample['query']
                    expected_kw = sample.get('expected_keywords', [])
                    
                    try:
                        if mode == 'dki':
                            response = self._run_plugin_chat(
                                query=query,
                                session_id=session_id,
                                user_id=user_id,
                                force_alpha=0.5,
                                max_new_tokens=1024,
                            )
                            response_text = response.text
                            latency = response.metadata.latency_ms
                        else:
                            response = self.rag_system.chat(
                                query=query,
                                session_id=session_id,
                                user_id=user_id,
                                max_new_tokens=1024,
                            )
                            response_text = response.text
                            latency = response.latency_ms
                        
                        resp_lower = response_text.lower()
                        kw_hits = sum(1 for kw in expected_kw if kw.lower() in resp_lower)
                        task_success = kw_hits / len(expected_kw) if expected_kw else 0.0
                        
                        mem_frags = sample.get('memory_fragments', [])
                        recall, _ = self.metrics.compute_memory_recall(
                            expected_memories=mem_frags[:5],
                            response=response_text,
                            threshold=0.3,
                        )
                        
                        halluc = self.metrics.compute_hallucination_decomposed(
                            response=response_text,
                            grounding_texts=mem_frags,
                            query=query,
                        )
                        
                        length_results[mode].append({
                            'sample_id': sample['id'],
                            'query': query,
                            'response': response_text[:300],
                            'latency_ms': latency,
                            'task_success': task_success,
                            'memory_recall': recall,
                            'fabricated_halluc': halluc['fabricated_rate'],
                            'irrelevant_halluc': halluc['irrelevant_rate'],
                        })
                        
                    except Exception as e:
                        logger.error(f"Context-constrained query failed: {e}")
            
            import numpy as np
            length_summary = {}
            for mode in ['dki', 'rag']:
                samples = length_results[mode]
                if not samples:
                    length_summary[mode] = {}
                    continue
                length_summary[mode] = {
                    'sample_count': len(samples),
                    'task_success': float(np.mean([s['task_success'] for s in samples])),
                    'memory_recall': float(np.mean([s['memory_recall'] for s in samples])),
                    'fabricated_halluc': float(np.mean([s['fabricated_halluc'] for s in samples])),
                    'irrelevant_halluc': float(np.mean([s['irrelevant_halluc'] for s in samples])),
                    'mean_latency_ms': float(np.mean([s['latency_ms'] for s in samples])),
                }
            
            results['results_by_length'][str(mem_length)] = {
                'samples': length_results,
                'summary': length_summary,
            }
        
        import numpy as np
        table_rows = []
        for mem_length in memory_lengths:
            key = str(mem_length)
            if key not in results['results_by_length']:
                continue
            s = results['results_by_length'][key]['summary']
            table_rows.append({
                'memory_length': mem_length,
                'rag_success': s.get('rag', {}).get('task_success', 0.0),
                'dki_success': s.get('dki', {}).get('task_success', 0.0),
                'delta': (
                    s.get('dki', {}).get('task_success', 0.0)
                    - s.get('rag', {}).get('task_success', 0.0)
                ),
            })
        results['comparison_table'] = table_rows
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"context_constrained_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Context-constrained results saved to {filepath}")
        for row in table_rows:
            logger.info(
                f"  mem={row['memory_length']}: "
                f"RAG={row['rag_success']:.3f}, "
                f"DKI={row['dki_success']:.3f}, "
                f"Δ={row['delta']:+.3f}"
            )
        
        return results


class InjectionInfoViewer:
    """注入信息查看器 - 用于显示和比较 DKI/RAG 的注入内容"""
    
    def __init__(self, output_dir: str = "./injection_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._injection_history: List[InjectionInfo] = []
    
    def record(self, injection_info: InjectionInfo) -> None:
        self._injection_history.append(injection_info)
    
    def get_latest(self, n: int = 10) -> List[InjectionInfo]:
        return self._injection_history[-n:]
    
    def display(self, injection_info: InjectionInfo) -> str:
        return injection_info.get_display_text()
    
    def compare(self, dki_info: InjectionInfo, rag_info: InjectionInfo) -> str:
        lines = []
        lines.append("╔═══════════════════════════════════════════╗")
        lines.append("║     DKI vs RAG 注入信息对比                 ║")
        lines.append("╚═══════════════════════════════════════════╝")
        lines.append("")
        lines.append(f"【原始查询】{dki_info.original_query}")
        lines.append("")
        
        lines.append("── DKI 注入 ──")
        if dki_info.preference_text:
            lines.append(f"  偏好: α={dki_info.alpha:.2f}, {dki_info.preference_tokens} tokens")
            lines.append(f"  {dki_info.preference_text[:80]}...")
        if dki_info.history_messages:
            lines.append(f"  历史: {len(dki_info.history_messages)} 条, {dki_info.history_tokens} tokens")
        
        lines.append("")
        lines.append("── RAG 注入 ──")
        if rag_info.rag_context:
            lines.append(f"  上下文: {rag_info.rag_context[:80]}...")
        if rag_info.history_messages:
            lines.append(f"  历史: {len(rag_info.history_messages)} 条")
        
        return "\n".join(lines)
    
    def save_to_file(self, injection_info: InjectionInfo, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"injection_{injection_info.mode}_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.display(injection_info))
        
        logger.info(f"Injection info saved to {filepath}")
        return str(filepath)
    
    def export_json(self, injection_info: InjectionInfo) -> Dict[str, Any]:
        return {
            **injection_info.to_dict(),
            'display_text': injection_info.get_display_text(),
        }