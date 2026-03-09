"""
Model Factory for DKI System
Creates and manages model adapters based on configuration

P2-M3: 支持命名空间多实例, 用于 A/B 测试同时加载多个模型。
向后兼容: 默认命名空间 "default" 行为与旧版完全一致。

Usage:
    # 默认用法 (向后兼容)
    adapter = ModelFactory.get_or_create()
    
    # 多实例用法 (A/B 测试)
    factory_a = ModelFactory(namespace="model_a")
    adapter_a = factory_a.get_or_create(model_name="/path/to/model_a")
    
    factory_b = ModelFactory(namespace="model_b")
    adapter_b = factory_b.get_or_create(model_name="/path/to/model_b")
"""

from typing import Dict, Optional, Type
from loguru import logger

from dki.models.base import BaseModelAdapter
from dki.models.vllm_adapter import VLLMAdapter
from dki.models.sglang_adapter import SGLangAdapter
from dki.models.llama_adapter import LlamaAdapter
from dki.models.deepseek_adapter import DeepSeekAdapter
from dki.models.glm_adapter import GLMAdapter
from dki.models.closed_source_adapter import ClosedSourceAdapter
from dki.config.config_loader import ConfigLoader


class ModelFactory:
    """
    Factory for creating model adapters.
    
    Supports multiple engines:
    - vllm: High-performance inference with vLLM
    - sglang: High-performance inference with SGLang (原生支持 Qwen3.5 等新架构)
    - llama: LLaMA models via HuggingFace
    - deepseek: DeepSeek models via HuggingFace
    - glm: ChatGLM/GLM-4 models via HuggingFace
    
    P2-M3: 支持命名空间多实例
    - 默认命名空间 "default" 保持向后兼容
    - 不同命名空间可加载不同模型 (A/B 测试)
    - 类方法使用 "default" 命名空间 (向后兼容)
    - 实例方法使用构造时指定的命名空间
    """
    
    # Registry of available adapters (全局共享)
    _adapters: Dict[str, Type[BaseModelAdapter]] = {
        'vllm': VLLMAdapter,
        'sglang': SGLangAdapter,
        'llama': LlamaAdapter,
        'deepseek': DeepSeekAdapter,
        'glm': GLMAdapter,
        'closed_source': ClosedSourceAdapter,
    }
    
    # 全局实例池: key = "namespace:engine:model_name"
    _instances: Dict[str, BaseModelAdapter] = {}
    
    def __init__(self, namespace: str = "default"):
        """
        创建命名空间化的 ModelFactory 实例
        
        Args:
            namespace: 命名空间标识 (默认 "default")
                       不同命名空间的模型实例互相隔离
        """
        self._namespace = namespace
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseModelAdapter]) -> None:
        """Register a new adapter type."""
        cls._adapters[name] = adapter_class
        logger.info(f"Registered adapter: {name}")
    
    @classmethod
    def create(
        cls,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseModelAdapter:
        """
        Create a model adapter.
        
        Args:
            engine: Engine type (vllm, llama, deepseek, glm)
            model_name: Model name/path (optional, uses config default)
            **kwargs: Additional arguments for the adapter
            
        Returns:
            Configured model adapter
        """
        # Load config
        config = ConfigLoader().config
        
        # Determine engine
        if engine is None:
            engine = config.model.default_engine
        
        engine = engine.lower()
        
        if engine not in cls._adapters:
            available = list(cls._adapters.keys())
            raise ValueError(f"Unknown engine: {engine}. Available: {available}")
        
        # Get engine config
        engine_config = config.model.engines.get(engine)
        
        if engine_config is None:
            raise ValueError(f"Engine {engine} not configured")
        
        if not engine_config.enabled:
            raise ValueError(f"Engine {engine} is disabled in configuration")
        
        # ============ 量化配置解析 ============
        # 向后兼容: load_in_8bit=True 且 quantization 未设置时, 映射到 "8bit"
        quantization = engine_config.quantization
        if engine_config.load_in_8bit and (not quantization or quantization == "none"):
            quantization = "8bit"
        
        # Merge configuration
        adapter_kwargs = {
            'model_name': model_name or engine_config.model_name,
            'device': engine_config.device,
            'dtype': engine_config.dtype,
            'trust_remote_code': engine_config.trust_remote_code,
            'quantization': quantization,
            'quantization_config': engine_config.quantization_config,
        }
        
        # Add engine-specific config
        if engine == 'vllm':
            adapter_kwargs.update({
                'tensor_parallel_size': engine_config.tensor_parallel_size,
                'max_model_len': engine_config.max_model_len,
                'gpu_memory_utilization': engine_config.gpu_memory_utilization,
                # v4.0: DKI 偏好注入模式
                # - "prompt_prefix": 偏好作为 prompt 前缀, 完整利用 vLLM (推荐)
                # - "hf_kv": HuggingFace 模型 KV 注入 (兼容旧行为)
                'injection_mode': engine_config.injection_mode,
                # v5.1: 模型实现后端
                # - "auto": vLLM 自动选择 (默认)
                # - "transformers": 强制使用 Transformers backend (新架构兼容)
                'model_impl': engine_config.model_impl,
            })
        elif engine == 'sglang':
            adapter_kwargs.update({
                'tensor_parallel_size': engine_config.tensor_parallel_size,
                'max_model_len': engine_config.max_model_len,
                'gpu_memory_utilization': engine_config.gpu_memory_utilization,
                'injection_mode': engine_config.injection_mode,
                # SGLang 特有参数
                'mem_fraction_static': getattr(engine_config, 'mem_fraction_static', 0.88),
                'schedule_policy': getattr(engine_config, 'schedule_policy', 'lpm'),
                'chunked_prefill_size': getattr(engine_config, 'chunked_prefill_size', 8192),
            })
        elif engine == 'llama':
            adapter_kwargs['load_in_8bit'] = engine_config.load_in_8bit
        elif engine == 'closed_source':
            # 闭源模型特殊参数: api_key, api_base, timeout 等
            adapter_kwargs.update({
                'api_key': getattr(engine_config, 'api_key', None),
                'api_base': getattr(engine_config, 'api_base', None),
                'api_version': getattr(engine_config, 'api_version', None),
                'max_model_len': engine_config.max_model_len,
                'timeout': getattr(engine_config, 'timeout', 120.0),
                'max_retries': getattr(engine_config, 'max_retries', 2),
                'default_system_prompt': getattr(
                    engine_config, 'default_system_prompt', None
                ),
            })
        
        # Override with explicit kwargs
        adapter_kwargs.update(kwargs)
        
        # Create adapter
        adapter_class = cls._adapters[engine]
        adapter = adapter_class(**adapter_kwargs)
        
        logger.info(f"Created {engine} adapter: {adapter_kwargs['model_name']}")
        return adapter
    
    # ================================================================
    # 类方法 (向后兼容, 使用 "default" 命名空间)
    # ================================================================
    
    @classmethod
    def get_or_create(
        cls,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
        namespace: str = "default",
        **kwargs
    ) -> BaseModelAdapter:
        """
        Get existing adapter or create new one.
        
        Uses singleton pattern to avoid loading same model multiple times.
        P2-M3: 支持命名空间隔离, 默认 "default" 保持向后兼容。
        
        Args:
            engine: 引擎类型
            model_name: 模型名称/路径
            namespace: 命名空间 (默认 "default")
            **kwargs: 额外参数
        """
        config = ConfigLoader().config
        engine = engine or config.model.default_engine
        
        engine_config = config.model.engines.get(engine)
        model_name = model_name or (engine_config.model_name if engine_config else None)
        
        cache_key = f"{namespace}:{engine}:{model_name}"
        
        if cache_key in cls._instances:
            adapter = cls._instances[cache_key]
            if adapter.is_loaded:
                return adapter
        
        # Create new adapter
        adapter = cls.create(engine=engine, model_name=model_name, **kwargs)
        adapter.load()
        cls._instances[cache_key] = adapter
        
        return adapter
    
    # ================================================================
    # 实例方法 (P2-M3: 命名空间化)
    # ================================================================
    
    def get_or_create_instance(
        self,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseModelAdapter:
        """
        P2-M3: 命名空间化的 get_or_create
        
        使用构造时指定的命名空间, 适用于 A/B 测试场景。
        
        Usage:
            factory_a = ModelFactory(namespace="model_a")
            adapter_a = factory_a.get_or_create_instance(
                model_name="/path/to/model_a"
            )
        """
        return self.get_or_create(
            engine=engine,
            model_name=model_name,
            namespace=self._namespace,
            **kwargs,
        )
    
    @classmethod
    def unload(
        cls,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """
        Unload a model adapter.
        
        P2-M3: 支持按命名空间卸载。
        
        Args:
            engine: 引擎类型 (None = 全部)
            model_name: 模型名称 (None = 全部)
            namespace: 命名空间 (None = 全部命名空间)
        """
        if engine is None and model_name is None and namespace is None:
            # Unload all
            for adapter in cls._instances.values():
                adapter.unload()
            cls._instances.clear()
            logger.info("Unloaded all models")
            return
        
        if namespace is not None and engine is None and model_name is None:
            # 卸载指定命名空间的所有模型
            keys_to_remove = [
                k for k in cls._instances if k.startswith(f"{namespace}:")
            ]
            for key in keys_to_remove:
                cls._instances[key].unload()
                del cls._instances[key]
            if keys_to_remove:
                logger.info(f"Unloaded {len(keys_to_remove)} models in namespace '{namespace}'")
            return
        
        # 向后兼容: 尝试 "default" 命名空间
        ns = namespace or "default"
        cache_key = f"{ns}:{engine}:{model_name}"
        if cache_key in cls._instances:
            cls._instances[cache_key].unload()
            del cls._instances[cache_key]
            logger.info(f"Unloaded model: {cache_key}")
        else:
            # 向后兼容: 旧格式 "engine:model_name" (无命名空间前缀)
            legacy_key = f"{engine}:{model_name}"
            if legacy_key in cls._instances:
                cls._instances[legacy_key].unload()
                del cls._instances[legacy_key]
                logger.info(f"Unloaded model (legacy key): {legacy_key}")
    
    def unload_instance(
        self,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """P2-M3: 卸载当前命名空间的模型"""
        self.unload(engine=engine, model_name=model_name, namespace=self._namespace)
    
    @classmethod
    def list_available(cls) -> Dict[str, bool]:
        """List available engines and their status."""
        config = ConfigLoader().config
        result = {}
        
        for engine in cls._adapters:
            engine_config = config.model.engines.get(engine)
            result[engine] = engine_config.enabled if engine_config else False
        
        return result
    
    @classmethod
    def list_loaded(cls, namespace: Optional[str] = None) -> Dict[str, dict]:
        """
        List currently loaded models.
        
        Args:
            namespace: 过滤指定命名空间 (None = 全部)
        """
        items = cls._instances.items()
        if namespace is not None:
            items = [(k, v) for k, v in items if k.startswith(f"{namespace}:")]
        return {
            key: adapter.get_model_info()
            for key, adapter in items
        }
    
    @classmethod
    def list_namespaces(cls) -> Dict[str, int]:
        """P2-M3: 列出所有命名空间及其模型数量"""
        namespaces: Dict[str, int] = {}
        for key in cls._instances:
            ns = key.split(":", 1)[0]
            namespaces[ns] = namespaces.get(ns, 0) + 1
        return namespaces

    @classmethod
    def is_closed_source_engine(cls, engine: Optional[str] = None) -> bool:
        """
        检查指定引擎 (或默认引擎) 是否为闭源模型

        用途:
        - 路由层据此自动强制 RAG 路由 (闭源模型无法 K/V 注入)
        - integration factory 据此设置 force_rag=True

        Args:
            engine: 引擎名称, None 时使用 config 默认引擎

        Returns:
            True 如果引擎是闭源模型
        """
        config = ConfigLoader().config
        engine = engine or config.model.default_engine
        return engine.lower() == "closed_source"

    @classmethod
    def get_adapter_is_closed_source(
        cls, adapter: Optional[BaseModelAdapter] = None
    ) -> bool:
        """
        检查适配器实例是否为闭源模型

        Args:
            adapter: 适配器实例, None 时检查默认实例

        Returns:
            True 如果适配器是闭源模型 (具有 is_closed_source=True 属性)
        """
        if adapter is None:
            return False
        return getattr(adapter, "is_closed_source", False)