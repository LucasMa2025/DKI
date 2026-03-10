"""
Closed-Source Model Adapter for DKI System

闭源模型适配器 — 通过 OpenAI 兼容 API 调用闭源 LLM

适用于:
- OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
- DeepSeek API (deepseek-chat, deepseek-reasoner)
- GLM API (glm-4, glm-4-flash)
- Moonshot (moonshot-v1-8k, moonshot-v1-32k)
- 任何兼容 OpenAI Chat Completions API 的服务

核心设计:
- 继承 BaseModelAdapter 但禁用 K/V 注入相关方法 (闭源模型无法访问内部)
- generate / async_generate / async_stream_generate 走 HTTP API
- compute_kv / forward_with_kv_injection 抛 NotImplementedError
  → 路由层检测到此适配器时自动走 RAG 路由 (prompt 拼接)
- 不依赖 torch / transformers (闭源模型无需本地 GPU)

配置方式:
```yaml
model:
    default_engine: "closed_source"
    engines:
        closed_source:
            enabled: true
            model_name: "deepseek-chat"
            api_key: "${DEEPSEEK_API_KEY}"
            api_base: "https://api.deepseek.com/v1"
            max_model_len: 32768
```

Author: AGI Demo Project
Version: 1.0.0
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from loguru import logger

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

# ModelOutput — 复用 base.py 的 dataclass (RAG 生成路径需要此类型)
try:
    from dki.models.base import ModelOutput
except ImportError:
    ModelOutput = None  # type: ignore


class ClosedSourceAdapter:
    """
    闭源模型适配器 — OpenAI 兼容 API

    与 BaseModelAdapter 接口兼容, 但:
    1. 不继承 BaseModelAdapter (避免 torch 硬依赖)
    2. generate / async_generate / async_stream_generate 实现完整
    3. compute_kv / forward_with_kv_injection / embed 明确不可用
    4. is_closed_source = True (路由层可据此判断)

    使用方式:
    ```python
    adapter = ClosedSourceAdapter(
        model_name="deepseek-chat",
        api_key="sk-xxx",
        api_base="https://api.deepseek.com/v1",
    )
    adapter.load()  # 验证连接 (可选)
    output = await adapter.async_generate("你好")
    ```
    """

    # 标记: 这是闭源模型适配器 (路由层可据此判断)
    is_closed_source: bool = True

    # 与 BaseModelAdapter 兼容的属性
    SUPPORTED_QUANTIZATIONS = ("none",)

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_model_len: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 2,
        # system prompt (可选, 用于 RAG 场景)
        default_system_prompt: Optional[str] = None,
        # 兼容 BaseModelAdapter 的参数 (忽略但不报错)
        device: str = "cpu",
        dtype: str = "float16",
        trust_remote_code: bool = True,
        quantization: str = "none",
        quantization_config: Optional[Dict[str, Any]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        injection_mode: str = "auto",
        model_impl: str = "auto",
        load_in_8bit: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
        self.api_version = api_version
        self.max_model_len = max_model_len
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_system_prompt = default_system_prompt

        # 兼容 BaseModelAdapter 的属性
        self.device = "cpu"
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.quantization = "none"
        self.quantization_config = {}

        # 架构信息 (闭源模型不暴露这些)
        self.hidden_dim: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.head_dim: int = 0

        # FlashAttention (不适用于闭源模型)
        self._flash_attn_config = None
        self._flash_attn_backend = None
        self._kv_injection_optimizer = None

        # HTTP 客户端 (延迟初始化)
        self._http_client = None
        self._async_http_client = None

        logger.info(
            f"Initializing ClosedSourceAdapter: "
            f"model={model_name}, api_base={self.api_base}"
        )

    # ================================================================
    # 生命周期
    # ================================================================

    def load(self) -> None:
        """
        加载 (验证 API 连接)

        对于闭源模型, load() 主要是:
        1. 验证 API key 是否设置
        2. 初始化 HTTP 客户端
        3. (可选) 发送一个简单请求验证连接
        """
        if not self.api_key:
            import os
            # 尝试从环境变量获取
            env_keys = [
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GLM_API_KEY",
                "ZHIPUAI_API_KEY",
                "MOONSHOT_API_KEY",
                "CLOSED_SOURCE_API_KEY",
            ]
            for env_key in env_keys:
                val = os.environ.get(env_key)
                if val:
                    self.api_key = val
                    logger.info(f"API key loaded from env: {env_key}")
                    break

        if not self.api_key:
            raise ValueError(
                "API key not provided. Set api_key parameter or one of "
                "OPENAI_API_KEY / DEEPSEEK_API_KEY / GLM_API_KEY environment variables."
            )

        # 初始化 HTTP 客户端
        try:
            import httpx
            self._http_client = httpx.Client(
                timeout=self.timeout,
                headers=self._build_headers(),
            )
            self._async_http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._build_headers(),
            )
        except ImportError:
            logger.info(
                "httpx not installed, will use aiohttp/requests as fallback. "
                "For best experience: pip install httpx"
            )

        self._is_loaded = True
        logger.info(
            f"ClosedSourceAdapter loaded: model={self.model_name}, "
            f"api_base={self.api_base}"
        )

    def unload(self) -> None:
        """释放资源"""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        if self._async_http_client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_http_client.aclose())
                else:
                    loop.run_until_complete(self._async_http_client.aclose())
            except Exception:
                pass
            self._async_http_client = None
        self._is_loaded = False
        logger.info(f"ClosedSourceAdapter unloaded: {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # ================================================================
    # HTTP 辅助
    # ================================================================

    def _build_headers(self) -> Dict[str, str]:
        """构建 API 请求头"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.api_version:
            headers["api-version"] = self.api_version
        return headers

    def _build_chat_request(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        构建 Chat Completions API 请求体

        支持两种 prompt 格式:
        1. 纯文本 → 自动包装为 messages
        2. 已是 ChatML/messages 格式 → 解析后使用
        """
        messages = self._parse_prompt_to_messages(prompt)

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        # 透传额外参数 (如 presence_penalty, frequency_penalty 等)
        for key in ("presence_penalty", "frequency_penalty", "stop", "n",
                     "response_format", "tools", "tool_choice", "seed"):
            if key in kwargs:
                request_body[key] = kwargs[key]

        return request_body

    def _parse_prompt_to_messages(
        self, prompt: str
    ) -> List[Dict[str, str]]:
        """
        解析 prompt 为 messages 列表

        支持:
        1. ChatML 格式: <|im_start|>role\ncontent<|im_end|>
        2. 纯文本: 直接作为 user message
        """
        # 检测是否为 ChatML 格式
        if "<|im_start|>" in prompt:
            return self._parse_chatml(prompt)

        # 检测是否为 Llama 3 格式
        if "<|begin_of_text|>" in prompt or "<|start_header_id|>" in prompt:
            return self._parse_llama3_format(prompt)

        # 纯文本 → user message
        messages = []
        if self.default_system_prompt:
            messages.append({
                "role": "system",
                "content": self.default_system_prompt,
            })
        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_chatml(self, prompt: str) -> List[Dict[str, str]]:
        """解析 ChatML 格式为 messages"""
        messages = []
        parts = prompt.split("<|im_start|>")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 移除 <|im_end|> 及其后面的内容
            if "<|im_end|>" in part:
                part = part.split("<|im_end|>")[0]

            # 分离 role 和 content
            lines = part.split("\n", 1)
            if len(lines) >= 2:
                role = lines[0].strip()
                content = lines[1].strip()
                if role in ("system", "user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            elif len(lines) == 1:
                role = lines[0].strip()
                if role == "assistant":
                    # 生成提示, 不添加消息
                    continue

        return messages if messages else [{"role": "user", "content": prompt}]

    def _parse_llama3_format(self, prompt: str) -> List[Dict[str, str]]:
        """解析 Llama 3 chat template 格式"""
        messages = []
        import re
        pattern = r"<\|start_header_id\|>(.*?)<\|end_header_id\|>\s*(.*?)(?=<\|eot_id\|>|<\|start_header_id\|>|$)"
        matches = re.findall(pattern, prompt, re.DOTALL)

        for role, content in matches:
            role = role.strip()
            content = content.strip()
            if role in ("system", "user", "assistant") and content:
                messages.append({"role": role, "content": content})

        return messages if messages else [{"role": "user", "content": prompt}]

    def _get_api_url(self) -> str:
        """获取 API URL"""
        base = self.api_base.rstrip("/")
        if not base.endswith("/chat/completions"):
            return f"{base}/chat/completions"
        return base

    # ================================================================
    # 生成方法 (核心)
    # ================================================================

    def _make_output(
        self,
        text: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        构造与 ModelOutput 兼容的输出对象

        优先使用 base.ModelOutput (RAG 路径需要 .text / .input_tokens 等字段),
        如果 import 失败则返回轻量 SimpleNamespace。
        """
        if ModelOutput is not None:
            return ModelOutput(
                text=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=metadata or {},
            )
        # fallback
        from types import SimpleNamespace
        return SimpleNamespace(
            text=text,
            tokens=None,
            logits=None,
            hidden_states=None,
            attentions=None,
            kv_cache=None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata or {},
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ):
        """
        同步生成 (通过 HTTP API)

        Returns:
            ModelOutput (与 BaseModelAdapter.generate 返回值兼容)
        """
        start_time = time.time()
        request_body = self._build_chat_request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            **kwargs,
        )

        response_data = self._sync_api_call(request_body)

        # 解析响应
        text = ""
        input_tokens = 0
        output_tokens = 0

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            text = choice.get("message", {}).get("content", "")

        if "usage" in response_data:
            usage = response_data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        latency_ms = (time.time() - start_time) * 1000

        return self._make_output(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "model": response_data.get("model", self.model_name),
                "finish_reason": (
                    response_data.get("choices", [{}])[0].get("finish_reason")
                    if response_data.get("choices")
                    else None
                ),
            },
        )

    async def async_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ):
        """
        异步生成 (通过 HTTP API)

        Returns:
            ModelOutput (与 BaseModelAdapter.async_generate 返回值兼容)
        """
        start_time = time.time()
        request_body = self._build_chat_request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            **kwargs,
        )

        response_data = await self._async_api_call(request_body)

        text = ""
        input_tokens = 0
        output_tokens = 0

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            text = choice.get("message", {}).get("content", "")

        if "usage" in response_data:
            usage = response_data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        latency_ms = (time.time() - start_time) * 1000

        return self._make_output(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "model": response_data.get("model", self.model_name),
                "finish_reason": (
                    response_data.get("choices", [{}])[0].get("finish_reason")
                    if response_data.get("choices")
                    else None
                ),
            },
        )

    async def async_stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        异步流式生成 (SSE)

        Yields:
            str: 文本片段 (逐 token)
        """
        request_body = self._build_chat_request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

        async for chunk_text in self._async_stream_api_call(request_body):
            yield chunk_text

    # ================================================================
    # Native Tool Calls (v7.1: 闭源模型原生 function calling)
    # ================================================================

    async def async_generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        带 tools 的异步生成 (原生 function calling)

        直接传入 messages 列表 (不需要 prompt 解析),
        支持 tools 和 tool_choice 参数。

        Returns:
            ModelOutput, metadata 中包含 raw_message 和 finish_reason
        """
        start_time = time.time()

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = tool_choice

        response_data = await self._async_api_call(request_body)

        text = ""
        input_tokens = 0
        output_tokens = 0
        raw_message = {}
        finish_reason = "stop"

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            raw_message = choice.get("message", {})
            text = raw_message.get("content", "") or ""
            finish_reason = choice.get("finish_reason", "stop")

        if "usage" in response_data:
            usage = response_data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

        latency_ms = (time.time() - start_time) * 1000

        return self._make_output(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "model": response_data.get("model", self.model_name),
                "finish_reason": finish_reason,
                "raw_message": raw_message,  # 包含 tool_calls
            },
        )

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ):
        """
        同步流式生成

        Yields:
            str: 文本片段
        """
        request_body = self._build_chat_request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

        for chunk_text in self._sync_stream_api_call(request_body):
            yield chunk_text

    # ================================================================
    # K/V 注入 (闭源模型不支持)
    # ================================================================

    def compute_kv(self, text: str, return_hidden: bool = False):
        """闭源模型不支持 K/V 计算"""
        raise NotImplementedError(
            f"compute_kv is not available for closed-source model "
            f"'{self.model_name}'. Use RAG route instead."
        )

    def forward_with_kv_injection(
        self, prompt: str, injected_kv: Any, alpha: float = 1.0,
        max_new_tokens: int = 2048, **kwargs,
    ):
        """闭源模型不支持 K/V 注入"""
        raise NotImplementedError(
            f"K/V injection is not available for closed-source model "
            f"'{self.model_name}'. Use RAG route instead."
        )

    def embed(self, text: str):
        """闭源模型不支持本地 embedding (可以调 Embedding API, 但本方法不实现)"""
        raise NotImplementedError(
            f"embed is not available for closed-source model "
            f"'{self.model_name}'. Use a separate embedding service."
        )

    def compute_prefill_entropy(self, text: str, layer_idx: int = 3) -> float:
        """闭源模型不支持 prefill entropy 计算"""
        raise NotImplementedError(
            f"compute_prefill_entropy is not available for closed-source model "
            f"'{self.model_name}'."
        )

    def tokenize(self, text: str) -> Dict[str, Any]:
        """闭源模型不支持本地 tokenize (返回估算值)"""
        # 简单估算: 中文约 2 char/token, 英文约 4 char/token
        mixed_ratio = sum(1 for c in text if '\u4e00' <= c <= '\u9fff') / max(len(text), 1)
        est_tokens = int(len(text) / (2.0 * mixed_ratio + 4.0 * (1 - mixed_ratio)))
        return {"input_ids": list(range(est_tokens))}

    def decode(self, tokens) -> str:
        """闭源模型不支持本地 decode"""
        raise NotImplementedError(
            f"decode is not available for closed-source model "
            f"'{self.model_name}'."
        )

    # ================================================================
    # 兼容 BaseModelAdapter 接口
    # ================================================================

    @property
    def is_quantized(self) -> bool:
        return False

    @property
    def is_4bit(self) -> bool:
        return False

    @property
    def is_8bit(self) -> bool:
        return False

    @property
    def is_fp8(self) -> bool:
        return False

    @property
    def flash_attn_enabled(self) -> bool:
        return False

    @property
    def flash_attn_backend(self) -> Optional[str]:
        return None

    def enable_flash_attention(self, config=None) -> str:
        """闭源模型不支持 FlashAttention"""
        logger.warning("FlashAttention not applicable to closed-source models")
        return "none"

    def disable_flash_attention(self):
        pass

    def get_flash_attn_stats(self) -> Dict[str, Any]:
        return {"enabled": False, "reason": "closed_source_model"}

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": "api",
            "dtype": "api",
            "quantization": "none",
            "is_quantized": False,
            "is_loaded": self._is_loaded,
            "is_closed_source": True,
            "api_base": self.api_base,
            "max_model_len": self.max_model_len,
            "hidden_dim": 0,
            "num_layers": 0,
            "num_heads": 0,
            "head_dim": 0,
            "flash_attn_enabled": False,
            "flash_attn_backend": None,
        }

    def __repr__(self) -> str:
        return (
            f"ClosedSourceAdapter(model={self.model_name}, "
            f"api_base={self.api_base})"
        )

    # ================================================================
    # HTTP 调用实现 (支持 httpx 和 fallback)
    # ================================================================

    def _sync_api_call(
        self, request_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """同步 API 调用"""
        url = self._get_api_url()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if self._http_client:
                    # httpx
                    response = self._http_client.post(
                        url, json=request_body,
                    )
                    response.raise_for_status()
                    return response.json()
                else:
                    # fallback: requests
                    import requests
                    response = requests.post(
                        url,
                        json=request_body,
                        headers=self._build_headers(),
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = (attempt + 1) * 1.0
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}), retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"API call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    async def _async_api_call(
        self, request_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步 API 调用"""
        url = self._get_api_url()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                if self._async_http_client:
                    # httpx
                    response = await self._async_http_client.post(
                        url, json=request_body,
                    )
                    response.raise_for_status()
                    return response.json()
                else:
                    # fallback: aiohttp
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                url,
                                json=request_body,
                                headers=self._build_headers(),
                                timeout=aiohttp.ClientTimeout(
                                    total=self.timeout
                                ),
                            ) as resp:
                                resp.raise_for_status()
                                return await resp.json()
                    except ImportError:
                        # 最终 fallback: 在线程池中执行同步调用
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(
                            None,
                            lambda: self._sync_api_call(request_body),
                        )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = (attempt + 1) * 1.0
                    logger.warning(
                        f"Async API call failed (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}), retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"Async API call failed after {self.max_retries + 1} "
            f"attempts: {last_error}"
        )

    async def _async_stream_api_call(
        self, request_body: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """异步流式 API 调用 (SSE)"""
        url = self._get_api_url()

        if self._async_http_client:
            # httpx streaming
            import httpx
            async with self._async_http_client.stream(
                "POST", url, json=request_body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    text = self._parse_sse_line(line)
                    if text is not None:
                        yield text
        else:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=request_body,
                        headers=self._build_headers(),
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.content:
                            line_str = line.decode("utf-8", errors="ignore").strip()
                            text = self._parse_sse_line(line_str)
                            if text is not None:
                                yield text
            except ImportError:
                # fallback: 非流式调用, 一次性返回
                logger.warning(
                    "No async HTTP library available for streaming, "
                    "falling back to non-streaming"
                )
                request_body["stream"] = False
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: self._sync_api_call(request_body)
                )
                text = ""
                if "choices" in result and result["choices"]:
                    text = result["choices"][0].get("message", {}).get(
                        "content", ""
                    )
                if text:
                    yield text

    def _sync_stream_api_call(
        self, request_body: Dict[str, Any]
    ):
        """同步流式 API 调用 (SSE)"""
        url = self._get_api_url()

        if self._http_client:
            import httpx
            with self._http_client.stream(
                "POST", url, json=request_body,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    text = self._parse_sse_line(line)
                    if text is not None:
                        yield text
        else:
            import requests
            response = requests.post(
                url,
                json=request_body,
                headers=self._build_headers(),
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8", errors="ignore")
                    text = self._parse_sse_line(line_str)
                    if text is not None:
                        yield text

    @staticmethod
    def _parse_sse_line(line: str) -> Optional[str]:
        """
        解析 SSE 行, 提取 delta content

        标准 SSE 格式:
            data: {"choices": [{"delta": {"content": "Hello"}}]}
            data: [DONE]
        """
        if not line:
            return None

        if line.startswith("data: "):
            data = line[6:]
        elif line.startswith("data:"):
            data = line[5:]
        else:
            return None

        data = data.strip()

        if data == "[DONE]":
            return None

        try:
            parsed = json.loads(data)
            if "choices" in parsed and parsed["choices"]:
                delta = parsed["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    return content
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        return None
