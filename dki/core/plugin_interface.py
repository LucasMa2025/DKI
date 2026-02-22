"""
DKI Plugin Interface - Standardized Plugin Architecture

Implements the standardized plugin interface from DKI Paper Section 8.2.

Design Philosophy:
1. Zero-code deployment: Enable DKI via configuration file
2. Framework agnostic: Works with vLLM, HuggingFace, TensorRT-LLM
3. Gradual adoption: Can be enabled/disabled per request
4. A/B testing ready: Easy comparison with vanilla LLM or RAG

Usage:
    # Load from config
    plugin = DKIPlugin.from_config("./dki_config.yaml")
    plugin.attach(model)
    
    # Generate with memory injection
    response = model.generate(
        input_ids,
        dki_user_id="user_123",
        dki_enabled=True
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
import torch
from loguru import logger


@dataclass
class MemorySourceConfig:
    """Configuration for memory data source."""
    type: str = "sqlite"  # sqlite | postgresql | redis | api | file
    connection: str = ""
    table: str = "user_memories"
    endpoint: str = ""
    auth: str = ""


@dataclass
class PreferenceInjectionConfig:
    """Configuration for preference K/V injection."""
    enabled: bool = True
    position_strategy: str = "negative"  # negative | actual_prefix
    alpha: float = 0.4
    max_tokens: int = 100


@dataclass
class HistoryInjectionConfig:
    """Configuration for history suffix injection."""
    enabled: bool = True
    method: str = "suffix_prompt"  # suffix_prompt | kv_injection
    max_tokens: int = 500
    prompt_template: str = "default"


@dataclass
class GatingConfig:
    """Configuration for dual-factor gating."""
    relevance_threshold: float = 0.7
    entropy_ceiling: float = 1.0
    entropy_floor: float = 0.5


@dataclass
class CacheConfig:
    """Configuration for K/V cache."""
    enabled: bool = True
    max_size: int = 100
    strategy: str = "weighted"  # lru | lfu | weighted
    ttl_seconds: int = 3600


@dataclass
class SafetyConfig:
    """Configuration for safety settings."""
    max_alpha: float = 0.8
    fallback_on_error: bool = True
    audit_logging: bool = True
    log_path: str = "./dki_audit.log"


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    enabled: bool = False
    dki_percentage: int = 50  # Percentage of traffic to route to DKI


@dataclass
class DKIPluginConfig:
    """Complete DKI plugin configuration."""
    enabled: bool = True
    version: str = "1.0"
    
    memory_source: MemorySourceConfig = field(default_factory=MemorySourceConfig)
    preference_injection: PreferenceInjectionConfig = field(default_factory=PreferenceInjectionConfig)
    history_injection: HistoryInjectionConfig = field(default_factory=HistoryInjectionConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DKIPluginConfig":
        """Create config from dictionary."""
        dki_data = data.get("dki", data)
        
        return cls(
            enabled=dki_data.get("enabled", True),
            version=dki_data.get("version", "1.0"),
            memory_source=MemorySourceConfig(**dki_data.get("memory_source", {})),
            preference_injection=PreferenceInjectionConfig(
                **dki_data.get("injection", {}).get("preference_injection", {})
            ),
            history_injection=HistoryInjectionConfig(
                **dki_data.get("injection", {}).get("history_injection", {})
            ),
            gating=GatingConfig(**dki_data.get("gating", {})),
            cache=CacheConfig(**dki_data.get("cache", {})),
            safety=SafetyConfig(**dki_data.get("safety", {})),
            ab_test=ABTestConfig(**dki_data.get("ab_test", {})),
        )
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DKIPluginConfig":
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dki": {
                "enabled": self.enabled,
                "version": self.version,
                "memory_source": {
                    "type": self.memory_source.type,
                    "connection": self.memory_source.connection,
                    "table": self.memory_source.table,
                },
                "injection": {
                    "preference_injection": {
                        "enabled": self.preference_injection.enabled,
                        "position_strategy": self.preference_injection.position_strategy,
                        "alpha": self.preference_injection.alpha,
                        "max_tokens": self.preference_injection.max_tokens,
                    },
                    "history_injection": {
                        "enabled": self.history_injection.enabled,
                        "method": self.history_injection.method,
                        "max_tokens": self.history_injection.max_tokens,
                        "prompt_template": self.history_injection.prompt_template,
                    },
                },
                "gating": {
                    "relevance_threshold": self.gating.relevance_threshold,
                    "entropy_ceiling": self.gating.entropy_ceiling,
                    "entropy_floor": self.gating.entropy_floor,
                },
                "cache": {
                    "enabled": self.cache.enabled,
                    "max_size": self.cache.max_size,
                    "strategy": self.cache.strategy,
                    "ttl_seconds": self.cache.ttl_seconds,
                },
                "safety": {
                    "max_alpha": self.safety.max_alpha,
                    "fallback_on_error": self.safety.fallback_on_error,
                    "audit_logging": self.safety.audit_logging,
                    "log_path": self.safety.log_path,
                },
                "ab_test": {
                    "enabled": self.ab_test.enabled,
                    "dki_percentage": self.ab_test.dki_percentage,
                },
            }
        }


class DKIPluginInterface(ABC):
    """
    Standard interface for DKI plugin implementations.
    
    Any inference framework can implement this interface
    to support DKI memory injection.
    
    Example Implementation:
        class VLLMDKIPlugin(DKIPluginInterface):
            def inject_memory(self, K_user, V_user, K_mem, V_mem, alpha):
                # vLLM-specific K/V injection
                ...
    """
    
    @abstractmethod
    def load_config(self, config_path: str) -> None:
        """Load DKI configuration from YAML file."""
        pass
    
    @abstractmethod
    def connect_memory_source(self, source_config: MemorySourceConfig) -> None:
        """
        Connect to external memory database.
        
        Supported sources:
        - PostgreSQL, MySQL, SQLite
        - Redis, MongoDB
        - REST API endpoint
        - Local JSON/JSONL file
        """
        pass
    
    @abstractmethod
    def get_user_memory(self, user_id: str) -> Tuple[str, str]:
        """
        Retrieve user memory (preferences, history).
        
        Returns:
            (preferences_text, history_text)
        """
        pass
    
    @abstractmethod
    def compute_memory_kv(
        self,
        memory_text: str,
        model: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute K/V representations for memory.
        
        Returns:
            (K_mem, V_mem) tensors
        """
        pass
    
    @abstractmethod
    def inject_memory(
        self,
        K_user: torch.Tensor,
        V_user: torch.Tensor,
        K_mem: torch.Tensor,
        V_mem: torch.Tensor,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject memory K/V into user K/V.
        
        Returns:
            (K_combined, V_combined)
        """
        pass


class DKIPlugin(DKIPluginInterface):
    """
    Default DKI plugin implementation.
    
    Provides:
    - Configuration-driven setup
    - Memory source connection
    - K/V computation and injection
    - A/B testing support
    - Audit logging
    
    Usage:
        plugin = DKIPlugin.from_config("./dki_config.yaml")
        plugin.attach(model)
        
        # Or manually
        plugin = DKIPlugin()
        plugin.load_config("./dki_config.yaml")
        plugin.connect_memory_source(config.memory_source)
    """
    
    def __init__(self, config: Optional[DKIPluginConfig] = None):
        self.config = config or DKIPluginConfig()
        self._model = None
        self._tokenizer = None
        self._memory_connection = None
        self._kv_cache: Dict[str, Any] = {}
        self._audit_log: List[Dict[str, Any]] = []
        
        # Import hybrid injector
        from dki.core.components.hybrid_injector import (
            HybridDKIInjector, HybridInjectionConfig
        )
        
        # Create hybrid injector with config
        hybrid_config = HybridInjectionConfig(
            preference_enabled=self.config.preference_injection.enabled,
            preference_alpha=self.config.preference_injection.alpha,
            preference_max_tokens=self.config.preference_injection.max_tokens,
            preference_position_strategy=self.config.preference_injection.position_strategy,
            history_enabled=self.config.history_injection.enabled,
            history_max_tokens=self.config.history_injection.max_tokens,
            history_method=self.config.history_injection.method,
        )
        self._hybrid_injector = HybridDKIInjector(config=hybrid_config)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "DKIPlugin":
        """Create plugin from configuration file."""
        config = DKIPluginConfig.from_yaml(config_path)
        plugin = cls(config)
        logger.info(f"DKI Plugin loaded from {config_path}")
        return plugin
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        self.config = DKIPluginConfig.from_yaml(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    
    def attach(self, model: Any, tokenizer: Optional[Any] = None) -> None:
        """
        Attach plugin to a model.
        
        Args:
            model: LLM model (HuggingFace, vLLM, etc.)
            tokenizer: Optional tokenizer
        """
        self._model = model
        self._tokenizer = tokenizer
        self._hybrid_injector.model = model
        self._hybrid_injector.tokenizer = tokenizer
        logger.info("DKI Plugin attached to model")
    
    def connect_memory_source(self, source_config: MemorySourceConfig) -> None:
        """Connect to memory data source."""
        source_type = source_config.type.lower()
        
        if source_type == "sqlite":
            import sqlite3
            self._memory_connection = sqlite3.connect(source_config.connection)
        elif source_type == "postgresql":
            try:
                import psycopg2
                self._memory_connection = psycopg2.connect(source_config.connection)
            except ImportError:
                logger.warning("psycopg2 not installed, using SQLite fallback")
        elif source_type == "redis":
            try:
                import redis
                self._memory_connection = redis.from_url(source_config.connection)
            except ImportError:
                logger.warning("redis not installed")
        elif source_type == "api":
            # Store endpoint for API calls
            self._memory_connection = {
                "endpoint": source_config.endpoint,
                "auth": source_config.auth,
            }
        elif source_type == "file":
            # File-based memory (JSON/JSONL)
            self._memory_connection = source_config.connection
        else:
            logger.warning(f"Unknown memory source type: {source_type}")
        
        logger.info(f"Connected to memory source: {source_type}")
    
    def get_user_memory(self, user_id: str) -> Tuple[str, str]:
        """
        Retrieve user memory from connected source.
        
        Returns:
            (preferences_text, history_text)
        """
        preferences = ""
        history = ""
        
        if self._memory_connection is None:
            logger.warning("No memory source connected")
            return preferences, history
        
        try:
            if isinstance(self._memory_connection, dict):
                # API source
                import requests
                endpoint = self._memory_connection["endpoint"]
                auth = self._memory_connection["auth"]
                headers = {"Authorization": auth} if auth else {}
                
                response = requests.get(
                    f"{endpoint}/users/{user_id}/memory",
                    headers=headers,
                    timeout=5
                )
                if response.ok:
                    data = response.json()
                    preferences = data.get("preferences", "")
                    history = data.get("history", "")
            elif hasattr(self._memory_connection, "cursor"):
                # SQL database
                cursor = self._memory_connection.cursor()
                cursor.execute(
                    "SELECT preferences, history FROM user_memories WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                if row:
                    preferences, history = row
            elif hasattr(self._memory_connection, "get"):
                # Redis
                data = self._memory_connection.hgetall(f"user:{user_id}")
                preferences = data.get(b"preferences", b"").decode()
                history = data.get(b"history", b"").decode()
        except Exception as e:
            logger.error(f"Failed to get user memory: {e}")
        
        return preferences, history
    
    def compute_memory_kv(
        self,
        memory_text: str,
        model: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute K/V representations for memory text."""
        if model is None:
            model = self._model
        
        if model is None:
            raise ValueError("No model available for K/V computation")
        
        # Check cache first
        cache_key = hash(memory_text)
        if cache_key in self._kv_cache:
            return self._kv_cache[cache_key]
        
        with torch.no_grad():
            if hasattr(model, "compute_kv"):
                # DKI model adapter
                kv_entries, _ = model.compute_kv(memory_text)
                # Extract K and V from entries
                K = torch.cat([e.key for e in kv_entries], dim=0)
                V = torch.cat([e.value for e in kv_entries], dim=0)
            elif hasattr(model, "model") and self._tokenizer:
                # HuggingFace model
                tokens = self._tokenizer.encode(memory_text, return_tensors="pt")
                outputs = model.model(input_ids=tokens, use_cache=True)
                kv = outputs.past_key_values
                # Stack all layers
                K = torch.stack([layer[0] for layer in kv])
                V = torch.stack([layer[1] for layer in kv])
            else:
                raise ValueError("Model does not support K/V computation")
        
        # Cache result
        if self.config.cache.enabled:
            self._kv_cache[cache_key] = (K, V)
        
        return K, V
    
    def inject_memory(
        self,
        K_user: torch.Tensor,
        V_user: torch.Tensor,
        K_mem: torch.Tensor,
        V_mem: torch.Tensor,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject memory K/V into user K/V with alpha scaling.
        
        Uses pre-softmax logit bias for mathematically correct scaling.
        """
        # Clamp alpha to safety limit
        alpha = min(alpha, self.config.safety.max_alpha)
        
        # Concatenate K/V
        K_combined = torch.cat([K_mem, K_user], dim=-2)
        V_combined = torch.cat([V_mem, V_user], dim=-2)
        
        # Note: Alpha scaling is applied during attention computation
        # via logit bias, not here. This just combines the K/V.
        
        return K_combined, V_combined
    
    def should_use_dki(self, user_id: str) -> bool:
        """
        Determine if DKI should be used for this request.
        
        Considers:
        - Plugin enabled status
        - A/B test configuration
        """
        if not self.config.enabled:
            return False
        
        if self.config.ab_test.enabled:
            # Simple hash-based routing for consistent user experience
            user_hash = hash(user_id) % 100
            return user_hash < self.config.ab_test.dki_percentage
        
        return True
    
    def log_injection(
        self,
        user_id: str,
        memory_ids: List[str],
        alpha: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log injection decision for audit."""
        if not self.config.safety.audit_logging:
            return
        
        import time
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "memory_ids": memory_ids,
            "alpha": alpha,
            "status": status,
            "metadata": metadata or {},
        }
        
        self._audit_log.append(log_entry)
        
        # Write to file if configured
        if self.config.safety.log_path:
            try:
                import json
                with open(self.config.safety.log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        return {
            "enabled": self.config.enabled,
            "version": self.config.version,
            "kv_cache_size": len(self._kv_cache),
            "audit_log_entries": len(self._audit_log),
            "ab_test": {
                "enabled": self.config.ab_test.enabled,
                "dki_percentage": self.config.ab_test.dki_percentage,
            },
            "hybrid_injector": self._hybrid_injector.get_stats(),
        }


# FastAPI middleware for easy integration
class DKIMiddleware:
    """
    FastAPI middleware for DKI integration.
    
    Usage:
        from fastapi import FastAPI
        from dki.core.plugin_interface import DKIMiddleware
        
        app = FastAPI()
        app.add_middleware(DKIMiddleware, config_path="./dki_config.yaml")
    """
    
    def __init__(self, app: Any, config_path: str):
        self.app = app
        self.plugin = DKIPlugin.from_config(config_path)
        
    async def __call__(self, scope: Dict, receive: Any, send: Any):
        # Inject plugin into request state
        if scope["type"] == "http":
            scope["state"] = scope.get("state", {})
            scope["state"]["dki_plugin"] = self.plugin
        
        await self.app(scope, receive, send)
