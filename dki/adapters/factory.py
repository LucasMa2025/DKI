"""
Adapter Factory
Creates user data adapters based on configuration

Author: AGI Demo Project
Version: 1.0.0
"""

from typing import Dict, Optional, Type

from loguru import logger

from dki.adapters.base import (
    IUserDataAdapter,
    AdapterConfig,
    AdapterType,
)
from dki.adapters.postgresql_adapter import PostgreSQLUserDataAdapter
from dki.adapters.mysql_adapter import MySQLUserDataAdapter
from dki.adapters.mongodb_adapter import MongoDBUserDataAdapter
from dki.adapters.redis_adapter import RedisUserDataAdapter
from dki.adapters.rest_adapter import RESTAPIUserDataAdapter
from dki.adapters.memory_adapter import InMemoryUserDataAdapter


class AdapterFactory:
    """
    Factory for creating user data adapters.
    
    Supports:
    - PostgreSQL
    - MySQL
    - MongoDB
    - Redis
    - REST API
    - In-Memory (for testing)
    - Custom adapters (via registration)
    
    Example:
        # Create from config
        config = AdapterConfig(
            adapter_type=AdapterType.POSTGRESQL,
            host="localhost",
            database="chat_db",
        )
        adapter = AdapterFactory.create(config)
        
        # Or use shorthand
        adapter = AdapterFactory.create_postgresql(
            host="localhost",
            database="chat_db",
        )
    """
    
    # Registry of adapter types to classes
    _adapters: Dict[AdapterType, Type[IUserDataAdapter]] = {
        AdapterType.POSTGRESQL: PostgreSQLUserDataAdapter,
        AdapterType.MYSQL: MySQLUserDataAdapter,
        AdapterType.MONGODB: MongoDBUserDataAdapter,
        AdapterType.REDIS: RedisUserDataAdapter,
        AdapterType.REST_API: RESTAPIUserDataAdapter,
        AdapterType.MEMORY: InMemoryUserDataAdapter,
    }
    
    # Singleton instances (optional)
    _instances: Dict[str, IUserDataAdapter] = {}
    
    @classmethod
    def register(
        cls,
        adapter_type: AdapterType,
        adapter_class: Type[IUserDataAdapter],
    ) -> None:
        """
        Register a custom adapter class.
        
        Args:
            adapter_type: Type identifier
            adapter_class: Adapter class implementing IUserDataAdapter
        """
        cls._adapters[adapter_type] = adapter_class
        logger.info(f"Registered adapter: {adapter_type} -> {adapter_class.__name__}")
    
    @classmethod
    def create(
        cls,
        config: AdapterConfig,
        singleton: bool = False,
    ) -> IUserDataAdapter:
        """
        Create an adapter instance from configuration.
        
        Args:
            config: Adapter configuration
            singleton: If True, return cached instance for same config
            
        Returns:
            Adapter instance
        """
        adapter_type = config.adapter_type
        
        # Check singleton cache
        if singleton:
            cache_key = f"{adapter_type}:{config.get_connection_string()}"
            if cache_key in cls._instances:
                return cls._instances[cache_key]
        
        # Get adapter class
        adapter_class = cls._adapters.get(adapter_type)
        
        if adapter_class is None:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Create instance
        adapter = adapter_class(config)
        
        # Cache if singleton
        if singleton:
            cls._instances[cache_key] = adapter
        
        logger.debug(f"Created adapter: {adapter_type}")
        return adapter
    
    @classmethod
    def create_from_dict(
        cls,
        config_dict: Dict,
        singleton: bool = False,
    ) -> IUserDataAdapter:
        """
        Create an adapter from a dictionary configuration.
        
        Args:
            config_dict: Configuration dictionary
            singleton: If True, return cached instance
            
        Returns:
            Adapter instance
        """
        # Parse adapter type
        adapter_type_str = config_dict.get("type") or config_dict.get("adapter_type", "memory")
        
        try:
            adapter_type = AdapterType(adapter_type_str.lower())
        except ValueError:
            adapter_type = AdapterType.MEMORY
            logger.warning(f"Unknown adapter type '{adapter_type_str}', using MEMORY")
        
        # Build config
        config = AdapterConfig(
            adapter_type=adapter_type,
            host=config_dict.get("host", "localhost"),
            port=config_dict.get("port", 5432),
            database=config_dict.get("database", ""),
            username=config_dict.get("username", ""),
            password=config_dict.get("password", ""),
            connection_string=config_dict.get("connection_string") or config_dict.get("connection"),
            schema=config_dict.get("schema", "public"),
            users_table=config_dict.get("users_table", "users"),
            messages_table=config_dict.get("messages_table", "messages"),
            preferences_table=config_dict.get("preferences_table", "user_preferences"),
            base_url=config_dict.get("base_url") or config_dict.get("endpoint", ""),
            api_key=config_dict.get("api_key") or config_dict.get("auth", ""),
            timeout=config_dict.get("timeout", 30),
            pool_size=config_dict.get("pool_size", 5),
            enable_cache=config_dict.get("enable_cache", True),
            cache_ttl=config_dict.get("cache_ttl", 300),
            options=config_dict.get("options", {}),
        )
        
        return cls.create(config, singleton=singleton)
    
    # Convenience methods for common adapters
    
    @classmethod
    def create_postgresql(
        cls,
        host: str = "localhost",
        port: int = 5432,
        database: str = "",
        username: str = "",
        password: str = "",
        connection_string: Optional[str] = None,
        **kwargs,
    ) -> PostgreSQLUserDataAdapter:
        """Create PostgreSQL adapter."""
        config = AdapterConfig(
            adapter_type=AdapterType.POSTGRESQL,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            connection_string=connection_string,
            **kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_mysql(
        cls,
        host: str = "localhost",
        port: int = 3306,
        database: str = "",
        username: str = "",
        password: str = "",
        **kwargs,
    ) -> MySQLUserDataAdapter:
        """Create MySQL adapter."""
        config = AdapterConfig(
            adapter_type=AdapterType.MYSQL,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            **kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_mongodb(
        cls,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: str = "",
        password: str = "",
        **kwargs,
    ) -> MongoDBUserDataAdapter:
        """Create MongoDB adapter."""
        config = AdapterConfig(
            adapter_type=AdapterType.MONGODB,
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            **kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_redis(
        cls,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        **kwargs,
    ) -> RedisUserDataAdapter:
        """Create Redis adapter."""
        config = AdapterConfig(
            adapter_type=AdapterType.REDIS,
            host=host,
            port=port,
            password=password,
            **kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_rest_api(
        cls,
        base_url: str,
        api_key: str = "",
        timeout: int = 30,
        **kwargs,
    ) -> RESTAPIUserDataAdapter:
        """Create REST API adapter."""
        config = AdapterConfig(
            adapter_type=AdapterType.REST_API,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )
        return cls.create(config)
    
    @classmethod
    def create_memory(cls) -> InMemoryUserDataAdapter:
        """Create in-memory adapter for testing."""
        config = AdapterConfig(adapter_type=AdapterType.MEMORY)
        return cls.create(config)
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached singleton instances."""
        cls._instances.clear()
        logger.info("Cleared adapter instance cache")
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported adapter types."""
        return list(cls._adapters.keys())
