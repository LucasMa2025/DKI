"""
Example Application Service

示例应用服务层

职责:
1. 管理示例应用的数据 (用户、会话、消息、偏好)
2. 调用 DKI 插件进行增强聊天
3. 展示 DKI 的使用方式

注意:
- 这是示例应用的业务逻辑，不是 DKI 的核心功能
- 数据存储在 ExampleAdapter 中 (内存)
- 生产环境应使用真实数据库

Author: AGI Demo Project
Version: 2.0.0
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger

from dki.core.dki_plugin import DKIPlugin, DKIPluginResponse
from dki.adapters.example_adapter import ExampleAdapter
from dki.models.base import BaseModelAdapter


@dataclass
class ChatResponse:
    """聊天响应"""
    message_id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    
    # DKI 元数据 (用于展示注入效果)
    dki_enabled: bool = False
    dki_alpha: float = 0.0
    dki_preference_tokens: int = 0
    dki_history_tokens: int = 0
    dki_cache_hit: bool = False
    dki_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "dki": {
                "enabled": self.dki_enabled,
                "alpha": self.dki_alpha,
                "preference_tokens": self.dki_preference_tokens,
                "history_tokens": self.dki_history_tokens,
                "cache_hit": self.dki_cache_hit,
                "latency_ms": self.dki_latency_ms,
            },
        }


class ExampleAppService:
    """
    示例应用服务
    
    演示如何使用 DKI 插件:
    1. 管理用户和会话 (示例数据存储)
    2. 调用 DKI 插件进行增强聊天
    3. 展示 DKI 注入效果
    
    使用方式:
    ```python
    # 创建服务
    service = ExampleAppService(model_adapter=vllm_adapter)
    
    # 注册用户
    user = service.register_user("user_001", "张三")
    
    # 设置偏好
    service.add_preference("user_001", "dietary", "素食主义者")
    
    # 创建会话
    session = service.create_session("user_001")
    
    # 聊天 (使用 DKI 增强)
    response = await service.chat(
        session_id=session["session_id"],
        user_id="user_001",
        message="推荐一家餐厅",
    )
    ```
    """
    
    def __init__(
        self,
        model_adapter: Optional[BaseModelAdapter] = None,
        language: str = "cn",
    ):
        """
        初始化示例应用服务
        
        Args:
            model_adapter: LLM 模型适配器
            language: 语言 ("en" | "cn")
        """
        # 创建示例适配器 (内存存储)
        self.adapter = ExampleAdapter()
        
        # 创建 DKI 插件
        self.dki_plugin: Optional[DKIPlugin] = None
        if model_adapter:
            self.dki_plugin = DKIPlugin(
                model_adapter=model_adapter,
                user_data_adapter=self.adapter,
                language=language,
            )
        
        self.language = language
        
        logger.info("ExampleAppService initialized")
    
    async def initialize(self):
        """初始化服务"""
        await self.adapter.connect()
        logger.info("ExampleAppService adapter connected")
    
    async def shutdown(self):
        """关闭服务"""
        await self.adapter.disconnect()
        logger.info("ExampleAppService adapter disconnected")
    
    # ============ 用户管理 (示例应用功能) ============
    
    def register_user(
        self,
        user_id: str,
        username: str,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        注册用户
        
        这是示例应用的功能，不是 DKI 的职责
        """
        user = self.adapter.add_user(
            user_id=user_id,
            username=username,
            display_name=display_name,
        )
        return user.to_dict()
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        user = self.adapter._store.users.get(user_id)
        return user.to_dict() if user else None
    
    # ============ 偏好管理 (示例应用功能) ============
    
    def add_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_text: str,
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        添加用户偏好
        
        这是示例应用的功能，DKI 会通过适配器读取这些偏好
        """
        pref = self.adapter.add_preference(
            user_id=user_id,
            preference_type=preference_type,
            preference_text=preference_text,
            priority=priority,
        )
        return pref.to_dict()
    
    async def get_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户偏好"""
        preferences = await self.adapter.get_user_preferences(user_id)
        return [p.to_dict() for p in preferences]
    
    def update_preference(
        self,
        user_id: str,
        preference_id: str,
        preference_text: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> bool:
        """更新偏好"""
        return self.adapter.update_preference(
            user_id=user_id,
            preference_id=preference_id,
            preference_text=preference_text,
            priority=priority,
        )
    
    def delete_preference(self, user_id: str, preference_id: str) -> bool:
        """删除偏好"""
        return self.adapter.delete_preference(user_id, preference_id)
    
    # ============ 会话管理 (示例应用功能) ============
    
    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        创建会话
        
        这是示例应用的功能，不是 DKI 的职责
        """
        return self.adapter.create_session(
            user_id=user_id,
            title=title,
        )
    
    async def get_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户会话列表"""
        return await self.adapter.get_user_sessions(user_id, limit=limit)
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取会话历史"""
        messages = await self.adapter.get_session_history(session_id, limit=limit)
        return [m.to_dict() for m in messages]
    
    # ============ 聊天 (调用 DKI 插件) ============
    
    async def chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
        use_dki: bool = True,
        force_alpha: Optional[float] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """
        聊天
        
        核心流程:
        1. 保存用户消息 (示例应用功能)
        2. 调用 DKI 插件 (DKI 会通过适配器读取偏好和历史)
        3. 保存助手响应 (示例应用功能)
        4. 返回响应 (包含 DKI 元数据)
        
        Args:
            session_id: 会话 ID
            user_id: 用户 ID
            message: 用户消息 (原始文本，不含任何 prompt)
            use_dki: 是否使用 DKI 增强
            force_alpha: 强制 alpha 值
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            
        Returns:
            ChatResponse 包含响应和 DKI 元数据
        """
        # Step 1: 保存用户消息 (示例应用功能)
        user_msg = self.adapter.add_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=message,
        )
        
        # Step 2: 调用 DKI 插件
        if use_dki and self.dki_plugin:
            # DKI 插件会:
            # 1. 通过适配器读取用户偏好
            # 2. 通过适配器检索相关历史
            # 3. 执行 K/V 注入
            # 4. 调用 LLM 推理
            dki_response = await self.dki_plugin.chat(
                query=message,  # 原始用户输入
                user_id=user_id,
                session_id=session_id,
                force_alpha=force_alpha,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            response_text = dki_response.text
            dki_metadata = dki_response.metadata
        else:
            # 不使用 DKI，直接调用模型
            if self.dki_plugin and self.dki_plugin.model:
                output = self.dki_plugin.model.generate(
                    prompt=message,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                response_text = output.text
            else:
                response_text = "[模型未加载]"
            
            dki_metadata = None
        
        # Step 3: 保存助手响应 (示例应用功能)
        assistant_msg = self.adapter.add_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=response_text,
            metadata={
                "dki_enabled": use_dki and dki_metadata is not None,
                "dki_alpha": dki_metadata.alpha if dki_metadata else 0.0,
            },
        )
        
        # Step 4: 构造响应
        return ChatResponse(
            message_id=assistant_msg.message_id,
            session_id=session_id,
            role="assistant",
            content=response_text,
            timestamp=assistant_msg.timestamp,
            dki_enabled=dki_metadata.injection_enabled if dki_metadata else False,
            dki_alpha=dki_metadata.alpha if dki_metadata else 0.0,
            dki_preference_tokens=dki_metadata.preference_tokens if dki_metadata else 0,
            dki_history_tokens=dki_metadata.history_tokens if dki_metadata else 0,
            dki_cache_hit=dki_metadata.preference_cache_hit if dki_metadata else False,
            dki_latency_ms=dki_metadata.latency_ms if dki_metadata else 0.0,
        )
    
    # ============ 统计 ============
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        adapter_stats = self.adapter.get_stats()
        dki_stats = self.dki_plugin.get_stats() if self.dki_plugin else {}
        
        return {
            "adapter": adapter_stats,
            "dki": dki_stats,
        }
