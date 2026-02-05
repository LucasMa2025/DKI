"""
Model Factory for DKI System
Creates and manages model adapters based on configuration
"""

from typing import Dict, Optional, Type
from loguru import logger

from dki.models.base import BaseModelAdapter
from dki.models.vllm_adapter import VLLMAdapter
from dki.models.llama_adapter import LlamaAdapter
from dki.models.deepseek_adapter import DeepSeekAdapter
from dki.models.glm_adapter import GLMAdapter
from dki.config.config_loader import ConfigLoader


class ModelFactory:
    """
    Factory for creating model adapters.
    
    Supports multiple engines:
    - vllm: High-performance inference with vLLM
    - llama: LLaMA models via HuggingFace
    - deepseek: DeepSeek models via HuggingFace
    - glm: ChatGLM/GLM-4 models via HuggingFace
    """
    
    # Registry of available adapters
    _adapters: Dict[str, Type[BaseModelAdapter]] = {
        'vllm': VLLMAdapter,
        'llama': LlamaAdapter,
        'deepseek': DeepSeekAdapter,
        'glm': GLMAdapter,
    }
    
    # Singleton instances for loaded models
    _instances: Dict[str, BaseModelAdapter] = {}
    
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
        
        # Merge configuration
        adapter_kwargs = {
            'model_name': model_name or engine_config.model_name,
            'device': engine_config.device,
            'torch_dtype': engine_config.torch_dtype,
            'trust_remote_code': engine_config.trust_remote_code,
        }
        
        # Add engine-specific config
        if engine == 'vllm':
            adapter_kwargs.update({
                'tensor_parallel_size': engine_config.tensor_parallel_size,
                'max_model_len': engine_config.max_model_len,
                'gpu_memory_utilization': engine_config.gpu_memory_utilization,
            })
        elif engine == 'llama':
            adapter_kwargs['load_in_8bit'] = engine_config.load_in_8bit
        
        # Override with explicit kwargs
        adapter_kwargs.update(kwargs)
        
        # Create adapter
        adapter_class = cls._adapters[engine]
        adapter = adapter_class(**adapter_kwargs)
        
        logger.info(f"Created {engine} adapter: {adapter_kwargs['model_name']}")
        return adapter
    
    @classmethod
    def get_or_create(
        cls,
        engine: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseModelAdapter:
        """
        Get existing adapter or create new one.
        
        Uses singleton pattern to avoid loading same model multiple times.
        """
        config = ConfigLoader().config
        engine = engine or config.model.default_engine
        
        engine_config = config.model.engines.get(engine)
        model_name = model_name or (engine_config.model_name if engine_config else None)
        
        cache_key = f"{engine}:{model_name}"
        
        if cache_key in cls._instances:
            adapter = cls._instances[cache_key]
            if adapter.is_loaded:
                return adapter
        
        # Create new adapter
        adapter = cls.create(engine=engine, model_name=model_name, **kwargs)
        adapter.load()
        cls._instances[cache_key] = adapter
        
        return adapter
    
    @classmethod
    def unload(cls, engine: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Unload a model adapter."""
        if engine is None and model_name is None:
            # Unload all
            for adapter in cls._instances.values():
                adapter.unload()
            cls._instances.clear()
            logger.info("Unloaded all models")
            return
        
        cache_key = f"{engine}:{model_name}"
        if cache_key in cls._instances:
            cls._instances[cache_key].unload()
            del cls._instances[cache_key]
            logger.info(f"Unloaded model: {cache_key}")
    
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
    def list_loaded(cls) -> Dict[str, dict]:
        """List currently loaded models."""
        return {
            key: adapter.get_model_info()
            for key, adapter in cls._instances.items()
        }
