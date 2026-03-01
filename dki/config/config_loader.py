"""
Configuration Loader for DKI System
Loads and manages configuration from YAML files
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from loguru import logger


class MISConfig(BaseModel):
    """Memory Influence Scaling configuration."""
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_default: float = 0.5


class ProjectionConfig(BaseModel):
    """Query-Conditioned Projection configuration."""
    rank: int = 64
    dropout: float = 0.1


class GatingConfig(BaseModel):
    """Dual-Factor Gating configuration."""
    entropy_threshold: float = 0.5
    relevance_threshold: float = 0.7
    use_margin: bool = True
    margin_weight: float = 0.3


class CacheConfig(BaseModel):
    """Session KV Cache configuration."""
    enabled: bool = True
    max_size: int = 100
    strategy: str = "weighted"
    ttl_seconds: int = 3600


class TieredCacheConfig(BaseModel):
    """
    Tiered KV Cache configuration.
    
    Implements the memory hierarchy from Paper Section 7.4:
    - L1 (GPU HBM): Hot memories, uncompressed
    - L2 (CPU RAM): Warm memories, compressed
    - L3 (SSD): Cold memories, quantized + compressed
    - L4 (Recompute): Text only, recompute on demand
    """
    enabled: bool = True
    l1_max_size: int = 10  # GPU HBM: 5-10 memories per session
    l2_max_size: int = 100  # CPU RAM: 50-100 memories per session
    l3_path: str = "./data/kv_cache"  # SSD storage path
    enable_l3: bool = True
    enable_l4: bool = True  # Text-only fallback with recompute
    ttl_seconds: int = 3600
    
    # Compression settings
    l2_compression: str = "simple"  # simple | gear (when available)
    l3_quantization: bool = True  # INT8 quantization for L3
    
    # Selective layer caching (Paper Section 7.4.2)
    selective_layers: bool = False
    cache_layers: List[int] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Middle layers


class PositionConfig(BaseModel):
    """Position Encoding configuration."""
    strategy: str = "virtual_prefix"


class IsolationConfig(BaseModel):
    """User isolation configuration (v3.1)."""
    enabled: bool = True
    cache_key_secret: str = ""
    security_level: str = "strict"  # strict | standard | relaxed
    verify_session_ownership: bool = True
    audit_enabled: bool = True
    audit_max_entries: int = 10000
    enable_inference_guard: bool = True


class PreferenceStoreConfig(BaseModel):
    """Preference store configuration."""
    max_users: int = 1000
    ttl_seconds: int = 3600


class ExecutorCacheConfig(BaseModel):
    """Executor cache configuration."""
    max_total: int = 10000
    max_per_user: int = 100


class HybridPreferenceConfig(BaseModel):
    """Hybrid injection preference configuration."""
    alpha: float = 0.4           # 偏好注入强度 (K/V)
    enabled: bool = True


class HybridHistoryConfig(BaseModel):
    """Hybrid injection history configuration."""
    max_turns: int = 10          # 最大历史轮数
    enabled: bool = True


class HybridInjectionConfigModel(BaseModel):
    """Hybrid injection configuration (v3.2)."""
    enabled: bool = True
    preference: HybridPreferenceConfig = Field(default_factory=HybridPreferenceConfig)
    history: HybridHistoryConfig = Field(default_factory=HybridHistoryConfig)


class RecallConfigModel(BaseModel):
    """Recall v4 configuration (v4.0).
    
    透传到 dki.core.recall.recall_config.RecallConfig.from_dict()。
    使用 Dict[str, Any] 子字段以兼容 RecallConfig dataclass 的嵌套结构。
    """
    enabled: bool = True
    strategy: str = "summary_with_fact_call"
    signals: Dict[str, Any] = Field(default_factory=dict)
    score_weights: Dict[str, Any] = Field(default_factory=dict)
    budget: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    fact_call: Dict[str, Any] = Field(default_factory=dict)
    prompt_formatter: str = "auto"
    epistemic_modes: Dict[str, Any] = Field(default_factory=dict)
    signal_gating: Dict[str, Any] = Field(default_factory=dict)


class DKIConfig(BaseModel):
    """DKI module configuration."""
    enabled: bool = True
    mis: MISConfig = Field(default_factory=MISConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tiered_cache: TieredCacheConfig = Field(default_factory=TieredCacheConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    
    # Use tiered cache instead of simple cache
    use_tiered_cache: bool = True
    
    # Injection strategy (v3.2: only recall_v4 supported)
    injection_strategy: str = "recall_v4"
    
    # Recall v4 configuration (v4.0: 多信号召回 + Summary + Fact Call)
    recall: RecallConfigModel = Field(default_factory=RecallConfigModel)
    
    # Hybrid injection configuration (v3.2)
    hybrid_injection: HybridInjectionConfigModel = Field(default_factory=HybridInjectionConfigModel)
    
    # User isolation (v3.1)
    isolation: IsolationConfig = Field(default_factory=IsolationConfig)
    
    # Preference store
    preference_store: PreferenceStoreConfig = Field(default_factory=PreferenceStoreConfig)
    
    # Executor cache
    executor_cache: ExecutorCacheConfig = Field(default_factory=ExecutorCacheConfig)


class RAGConfig(BaseModel):
    """RAG module configuration."""
    enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.5
    index_type: str = "faiss"


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    batch_size: int = 32
    normalize: bool = True


class EngineConfig(BaseModel):
    """Model engine configuration."""
    enabled: bool = True
    model_name: str
    device: str = "cuda"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    load_in_8bit: bool = False
    # v5.0: DKI 偏好注入模式 (仅 vllm 引擎生效)
    # - "auto" (推荐): vLLM 原生 KV 注入 (prefix_caching)
    # - "prompt_prefix": 等同于 auto
    # - "vllm_kv": 等同于 auto
    # - "hf_kv": 已废弃, 接受但内部走 vLLM 原生推理 (向后兼容)
    injection_mode: str = "auto"
    # v4.0: 是否启用 AGA 全局知识注入 (仅 vllm 引擎生效)
    aga_enabled: bool = False
    aga_config: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model configuration."""
    default_engine: str = "vllm"
    engines: Dict[str, EngineConfig] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    path: str = "./data/dki.db"
    pool_size: int = 5
    echo: bool = False


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    reload: bool = False


class MemoryConfig(BaseModel):
    """Memory store configuration."""
    max_memories_per_session: int = 1000
    max_memory_length: int = 2048
    embedding_dim: int = 384


class SystemConfig(BaseModel):
    """System configuration."""
    name: str = "DKI System"
    version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"


class Config(BaseModel):
    """Main configuration model."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    dki: DKIConfig = Field(default_factory=DKIConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


class ConfigLoader:
    """Configuration loader and manager."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Config] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._config is None:
            self._config_path = config_path or self._find_config_path()
            self._load_config()
    
    def _find_config_path(self) -> str:
        """Find configuration file path."""
        possible_paths = [
            os.environ.get("DKI_CONFIG_PATH", ""),
            "./config/config.yaml",
            "./config.yaml",
            "../config/config.yaml",
            str(Path(__file__).parent.parent.parent / "config" / "config.yaml"),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found")
    
    @staticmethod
    def _resolve_env_vars(obj: Any) -> Any:
        """
        递归替换配置中的环境变量占位符。
        
        支持语法:
          ${ENV_VAR}          — 必须存在, 否则保留原文
          ${ENV_VAR:-default} — 不存在时使用 default
        
        示例:
          model_name: "${DKI_MODEL_PATH:-/opt/ai-demo/models/deepseek-r1-distill-qwen-7b}"
        """
        import re
        _env_pattern = re.compile(r'\$\{([^}]+)\}')

        def _replace(match: re.Match) -> str:
            expr = match.group(1)
            if ':-' in expr:
                var_name, default_val = expr.split(':-', 1)
                return os.environ.get(var_name.strip(), default_val.strip())
            return os.environ.get(expr.strip(), match.group(0))

        if isinstance(obj, str):
            return _env_pattern.sub(_replace, obj)
        elif isinstance(obj, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigLoader._resolve_env_vars(item) for item in obj]
        return obj

    def _load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 环境变量替换 (支持 ${VAR} 和 ${VAR:-default} 语法)
            raw_config = self._resolve_env_vars(raw_config)
            
            # Parse engine configs
            if 'model' in raw_config and 'engines' in raw_config['model']:
                engines = {}
                for engine_name, engine_cfg in raw_config['model']['engines'].items():
                    engines[engine_name] = EngineConfig(**engine_cfg)
                raw_config['model']['engines'] = engines
            
            self._config = Config(**raw_config)
            logger.info(f"Configuration loaded from {self._config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            self._config = Config()
    
    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, AttributeError):
            return default
    
    def reload(self) -> None:
        """Reload configuration."""
        self._load_config()
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None
        cls._config = None


def get_config() -> Config:
    """Get global configuration instance."""
    return ConfigLoader().config
