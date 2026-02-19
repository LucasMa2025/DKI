"""
Example Application Entry Point

示例应用入口

启动示例应用，演示如何使用 DKI 插件

Usage:
    python -m dki.example_app.main
    python -m dki.example_app.main --host 0.0.0.0 --port 8080
    python -m dki.example_app.main --model-url http://localhost:8001/v1

Author: AGI Demo Project
Version: 2.0.0
"""

import argparse
import uvicorn
from loguru import logger

from dki.example_app.app import create_example_app
from dki.example_app.service import ExampleAppService


def create_mock_model_adapter():
    """
    创建模拟模型适配器
    
    用于演示，不需要真实模型
    """
    from dki.models.base import BaseModelAdapter, ModelOutput, KVCacheEntry
    from typing import List, Optional
    
    class MockModelAdapter(BaseModelAdapter):
        """模拟模型适配器"""
        
        def __init__(self):
            self._hidden_dim = 4096
        
        @property
        def hidden_dim(self) -> int:
            return self._hidden_dim
        
        def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            **kwargs,
        ) -> ModelOutput:
            """模拟生成"""
            # 简单的模拟响应
            response = f"[模拟响应] 收到您的问题: {prompt[:50]}..."
            return ModelOutput(
                text=response,
                input_tokens=len(prompt.split()),
                output_tokens=len(response.split()),
            )
        
        def forward_with_kv_injection(
            self,
            prompt: str,
            injected_kv: List[KVCacheEntry],
            alpha: float = 0.5,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            **kwargs,
        ) -> ModelOutput:
            """模拟带 K/V 注入的生成"""
            # 简单的模拟响应，显示注入信息
            response = f"[DKI 增强响应] (alpha={alpha:.2f}) 收到您的问题: {prompt[:50]}..."
            return ModelOutput(
                text=response,
                input_tokens=len(prompt.split()),
                output_tokens=len(response.split()),
            )
        
        def compute_kv(self, text: str) -> tuple:
            """模拟 K/V 计算"""
            import torch
            
            # 创建模拟 K/V
            seq_len = len(text.split())
            num_layers = 32
            num_heads = 32
            head_dim = self._hidden_dim // num_heads
            
            kv_entries = []
            for layer_idx in range(num_layers):
                k = torch.randn(1, num_heads, seq_len, head_dim)
                v = torch.randn(1, num_heads, seq_len, head_dim)
                kv_entries.append(KVCacheEntry(
                    layer_idx=layer_idx,
                    key=k,
                    value=v,
                ))
            
            return kv_entries, seq_len
    
    return MockModelAdapter()


def main():
    parser = argparse.ArgumentParser(description="DKI Example Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--model-url", default=None, help="Model API URL (optional)")
    parser.add_argument("--mock", action="store_true", help="Use mock model adapter")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # 创建模型适配器
    model_adapter = None
    if args.mock or args.model_url is None:
        logger.info("Using mock model adapter (for demonstration)")
        model_adapter = create_mock_model_adapter()
    else:
        # TODO: 创建真实模型适配器
        logger.info(f"Connecting to model at {args.model_url}")
        model_adapter = create_mock_model_adapter()  # 暂时使用模拟
    
    # 创建服务
    service = ExampleAppService(
        model_adapter=model_adapter,
        language="cn",
    )
    
    # 添加示例数据
    service.adapter.add_user("demo_user", "演示用户", "Demo User")
    service.adapter.add_preference("demo_user", "dietary", "素食主义者，不吃辣")
    service.adapter.add_preference("demo_user", "communication", "喜欢简洁的回答")
    service.adapter.create_session("demo_user", "demo_session", "演示会话")
    
    logger.info("Demo user and session created")
    logger.info("  User ID: demo_user")
    logger.info("  Session ID: demo_session")
    
    # 创建应用
    app = create_example_app(service=service)
    
    # 启动
    logger.info(f"Starting DKI Example Application at http://{args.host}:{args.port}")
    logger.info("API Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
