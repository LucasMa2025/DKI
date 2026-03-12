# -*- coding: utf-8 -*-
"""
单元测试: vLLM / SGLang Adapter Logprobs 支持 (v8.0 熵门控)

测试范围:
- VLLMAdapter._parse_vllm_logprobs: vLLM logprobs 解析
- SGLangAdapter._parse_sglang_logprobs: SGLang logprobs 解析
- VLLMAdapter.generate() logprobs 参数传递
- SGLangAdapter.generate() logprobs 参数传递
- SGLangAdapter.async_generate() logprobs 参数传递
- ModelOutput.logprobs 字段正确传播

不依赖真实模型加载, 使用 Mock 模拟所有引擎组件。

Author: AGI Demo Project
"""

import asyncio
import math
import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# ============================================================
# 添加项目路径
# ============================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# Mock vLLM Logprob 对象
# ============================================================

class MockVLLMLogprob:
    """模拟 vLLM 的 Logprob 对象 (NamedTuple-like)"""
    def __init__(self, logprob: float, rank: int = 1, decoded_token: str = ""):
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = decoded_token


# ============================================================
# 测试 VLLMAdapter._parse_vllm_logprobs
# ============================================================

class TestParseVLLMLogprobs:
    """测试 vLLM logprobs 解析"""
    
    @pytest.fixture
    def adapter_cls(self):
        """获取 VLLMAdapter 类 (不实例化)"""
        from dki.models.vllm_adapter import VLLMAdapter
        return VLLMAdapter
    
    def test_parse_logprob_objects(self, adapter_cls):
        """正常情况: vLLM Logprob 对象"""
        raw = [
            {101: MockVLLMLogprob(-0.5), 102: MockVLLMLogprob(-1.2), 103: MockVLLMLogprob(-2.0)},
            {201: MockVLLMLogprob(-0.1), 202: MockVLLMLogprob(-3.0)},
        ]
        result = adapter_cls._parse_vllm_logprobs(raw)
        assert len(result) == 2
        assert result[0] == [-0.5, -1.2, -2.0]
        assert result[1] == [-0.1, -3.0]
    
    def test_parse_float_values(self, adapter_cls):
        """旧版 vLLM: logprob 值直接是 float"""
        raw = [
            {101: -0.5, 102: -1.2},
            {201: -0.1},
        ]
        result = adapter_cls._parse_vllm_logprobs(raw)
        assert len(result) == 2
        assert result[0] == [-0.5, -1.2]
        assert result[1] == [-0.1]
    
    def test_parse_with_none_tokens(self, adapter_cls):
        """某些 token 的 logprobs 为 None (如 prompt tokens)"""
        raw = [
            None,
            {101: MockVLLMLogprob(-0.5)},
            None,
            {201: MockVLLMLogprob(-1.0), 202: MockVLLMLogprob(-2.0)},
        ]
        result = adapter_cls._parse_vllm_logprobs(raw)
        assert len(result) == 2
        assert result[0] == [-0.5]
        assert result[1] == [-1.0, -2.0]
    
    def test_parse_empty_list(self, adapter_cls):
        """空列表"""
        result = adapter_cls._parse_vllm_logprobs([])
        assert result == []
    
    def test_parse_mixed_types(self, adapter_cls):
        """混合类型: 部分是对象, 部分是 float"""
        raw = [
            {101: MockVLLMLogprob(-0.5), 102: -1.2},
        ]
        result = adapter_cls._parse_vllm_logprobs(raw)
        assert len(result) == 1
        assert result[0] == [-0.5, -1.2]


# ============================================================
# 测试 SGLangAdapter._parse_sglang_logprobs
# ============================================================

class TestParseSGLangLogprobs:
    """测试 SGLang logprobs 解析"""
    
    @pytest.fixture
    def adapter_cls(self):
        """获取 SGLangAdapter 类 (不实例化)"""
        from dki.models.sglang_adapter import SGLangAdapter
        return SGLangAdapter
    
    def test_parse_top_logprobs_dict(self, adapter_cls):
        """正常情况: output_top_logprobs 是 List[Dict[int, float]]"""
        meta_info = {
            "output_top_logprobs": [
                {101: -0.5, 102: -1.2, 103: -2.0},
                {201: -0.1, 202: -3.0},
            ]
        }
        result = adapter_cls._parse_sglang_logprobs(meta_info)
        assert result is not None
        assert len(result) == 2
        assert sorted(result[0]) == sorted([-0.5, -1.2, -2.0])
        assert sorted(result[1]) == sorted([-0.1, -3.0])
    
    def test_parse_top_logprobs_list_format(self, adapter_cls):
        """output_top_logprobs 内部元素已经是 list"""
        meta_info = {
            "output_top_logprobs": [
                [-0.5, -1.2],
                [-0.1, -3.0],
            ]
        }
        result = adapter_cls._parse_sglang_logprobs(meta_info)
        assert result is not None
        assert len(result) == 2
        assert result[0] == [-0.5, -1.2]
        assert result[1] == [-0.1, -3.0]
    
    def test_parse_fallback_token_logprobs(self, adapter_cls):
        """回退: 只有 output_token_logprobs (无 top-k)"""
        meta_info = {
            "output_token_logprobs": [-0.5, -1.2, -0.1],
        }
        result = adapter_cls._parse_sglang_logprobs(meta_info)
        assert result is not None
        assert len(result) == 3
        assert result[0] == [-0.5]
        assert result[1] == [-1.2]
        assert result[2] == [-0.1]
    
    def test_parse_empty_meta_info(self, adapter_cls):
        """空 meta_info"""
        result = adapter_cls._parse_sglang_logprobs({})
        assert result is None
    
    def test_parse_none_meta_info(self, adapter_cls):
        """None meta_info"""
        result = adapter_cls._parse_sglang_logprobs(None)
        assert result is None
    
    def test_parse_no_logprob_keys(self, adapter_cls):
        """meta_info 中没有 logprob 相关键"""
        meta_info = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        result = adapter_cls._parse_sglang_logprobs(meta_info)
        assert result is None


# ============================================================
# 测试 VLLMAdapter.generate() logprobs 传递
# ============================================================

class TestVLLMAdapterGenerateLogprobs:
    """测试 VLLMAdapter.generate() 的 logprobs 参数传递"""
    
    @pytest.fixture
    def mock_adapter(self):
        """创建一个 mock 的 VLLMAdapter"""
        with patch.dict(sys.modules, {
            'vllm': MagicMock(),
        }):
            from dki.models.vllm_adapter import VLLMAdapter
            adapter = VLLMAdapter.__new__(VLLMAdapter)
            adapter.model_name = "test-model"
            adapter._is_loaded = True
            adapter.tokenizer = MagicMock()
            adapter.tokenizer.eos_token = "<|im_end|>"
            
            # Mock vLLM output
            mock_output_item = MagicMock()
            mock_output_item.text = "test response"
            mock_output_item.token_ids = [101, 102, 103]
            mock_output_item.logprobs = [
                {101: MockVLLMLogprob(-0.5), 102: MockVLLMLogprob(-1.2)},
                {201: MockVLLMLogprob(-0.1)},
                {301: MockVLLMLogprob(-0.8), 302: MockVLLMLogprob(-2.0)},
            ]
            
            mock_request_output = MagicMock()
            mock_request_output.outputs = [mock_output_item]
            mock_request_output.prompt_token_ids = [1, 2, 3, 4, 5]
            
            mock_llm = MagicMock()
            mock_llm.generate.return_value = [mock_request_output]
            adapter.llm = mock_llm
            
            return adapter
    
    def test_generate_with_logprobs(self, mock_adapter):
        """generate() 传入 logprobs=5 时, SamplingParams 包含 logprobs 参数"""
        with patch.dict(sys.modules, {'vllm': MagicMock()}):
            mock_sp_cls = MagicMock()
            with patch('dki.models.vllm_adapter.SamplingParams', mock_sp_cls, create=True):
                # 重新 import 以使用 mock
                import importlib
                vllm_mod = sys.modules.get('vllm', MagicMock())
                vllm_mod.SamplingParams = mock_sp_cls
                sys.modules['vllm'] = vllm_mod
                
                output = mock_adapter.generate(
                    prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
                    max_new_tokens=64,
                    logprobs=5,
                )
                
                # 验证 SamplingParams 被调用时包含 logprobs
                sp_call_kwargs = mock_sp_cls.call_args
                assert sp_call_kwargs is not None
                if sp_call_kwargs.kwargs:
                    assert sp_call_kwargs.kwargs.get('logprobs') == 5
                
                # 验证 ModelOutput 包含 parsed logprobs
                assert output.logprobs is not None
                assert len(output.logprobs) == 3
    
    def test_generate_without_logprobs(self, mock_adapter):
        """generate() 不传 logprobs 时, SamplingParams 不包含 logprobs"""
        with patch.dict(sys.modules, {'vllm': MagicMock()}):
            mock_sp_cls = MagicMock()
            vllm_mod = sys.modules.get('vllm', MagicMock())
            vllm_mod.SamplingParams = mock_sp_cls
            sys.modules['vllm'] = vllm_mod
            
            output = mock_adapter.generate(
                prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
                max_new_tokens=64,
            )
            
            # logprobs 应为 None
            assert output.logprobs is None


# ============================================================
# 测试 SGLangAdapter.generate() logprobs 传递
# ============================================================

class TestSGLangAdapterGenerateLogprobs:
    """测试 SGLangAdapter.generate() 的 logprobs 参数传递"""
    
    @pytest.fixture
    def mock_adapter(self):
        """创建一个 mock 的 SGLangAdapter"""
        from dki.models.sglang_adapter import SGLangAdapter
        adapter = SGLangAdapter.__new__(SGLangAdapter)
        adapter.model_name = "test-model"
        adapter._is_loaded = True
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.eos_token = "<|im_end|>"
        
        # Mock SGLang engine output (dict 格式)
        mock_engine = MagicMock()
        mock_engine.generate.return_value = {
            "text": "test response",
            "meta_info": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "output_ids": [101, 102, 103],
                "output_top_logprobs": [
                    {101: -0.5, 102: -1.2},
                    {201: -0.1, 202: -3.0},
                    {301: -0.8},
                ],
                "output_token_logprobs": [-0.5, -0.1, -0.8],
            },
        }
        adapter.engine = mock_engine
        
        return adapter
    
    def test_generate_with_logprobs(self, mock_adapter):
        """generate() 传入 logprobs=5 时, sampling_params 包含 return_logprob"""
        output = mock_adapter.generate(
            prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=64,
            logprobs=5,
        )
        
        # 验证 engine.generate 被调用时 sampling_params 包含 logprob 参数
        call_args = mock_adapter.engine.generate.call_args
        sampling_params = call_args[0][1]  # 第二个位置参数
        assert sampling_params.get("return_logprob") is True
        assert sampling_params.get("top_logprobs_num") == 5
        
        # 验证 ModelOutput 包含 parsed logprobs
        assert output.logprobs is not None
        assert len(output.logprobs) == 3
    
    def test_generate_without_logprobs(self, mock_adapter):
        """generate() 不传 logprobs 时, sampling_params 不包含 return_logprob"""
        # 修改 mock 返回值, 移除 logprob 相关字段
        mock_adapter.engine.generate.return_value = {
            "text": "test response",
            "meta_info": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "output_ids": [101, 102, 103],
            },
        }
        
        output = mock_adapter.generate(
            prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=64,
        )
        
        # 验证 sampling_params 不包含 logprob 参数
        call_args = mock_adapter.engine.generate.call_args
        sampling_params = call_args[0][1]
        assert "return_logprob" not in sampling_params
        assert "top_logprobs_num" not in sampling_params
        
        # logprobs 应为 None
        assert output.logprobs is None


# ============================================================
# 测试 SGLangAdapter.async_generate() logprobs 传递
# ============================================================

class TestSGLangAdapterAsyncGenerateLogprobs:
    """测试 SGLangAdapter.async_generate() 的 logprobs 参数传递"""
    
    @pytest.fixture
    def mock_adapter(self):
        """创建一个 mock 的 SGLangAdapter"""
        from dki.models.sglang_adapter import SGLangAdapter
        adapter = SGLangAdapter.__new__(SGLangAdapter)
        adapter.model_name = "test-model"
        adapter._is_loaded = True
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.eos_token = "<|im_end|>"
        
        # Mock SGLang engine with async_generate
        mock_engine = MagicMock()
        mock_engine.async_generate = AsyncMock(return_value={
            "text": "async test response",
            "meta_info": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "output_ids": [101, 102, 103],
                "output_top_logprobs": [
                    {101: -0.5, 102: -1.2},
                    {201: -0.1},
                ],
            },
        })
        adapter.engine = mock_engine
        
        return adapter
    
    @pytest.mark.asyncio
    async def test_async_generate_with_logprobs(self, mock_adapter):
        """async_generate() 传入 logprobs=5"""
        output = await mock_adapter.async_generate(
            prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=64,
            logprobs=5,
        )
        
        # 验证 async_generate 被调用时 sampling_params 包含 logprob 参数
        call_args = mock_adapter.engine.async_generate.call_args
        sampling_params = call_args[0][1]
        assert sampling_params.get("return_logprob") is True
        assert sampling_params.get("top_logprobs_num") == 5
        
        # 验证 logprobs 被正确解析
        assert output.logprobs is not None
        assert len(output.logprobs) == 2
    
    @pytest.mark.asyncio
    async def test_async_generate_without_logprobs(self, mock_adapter):
        """async_generate() 不传 logprobs"""
        mock_adapter.engine.async_generate = AsyncMock(return_value={
            "text": "async test response",
            "meta_info": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "output_ids": [101, 102],
            },
        })
        
        output = await mock_adapter.async_generate(
            prompt="<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n",
            max_new_tokens=64,
        )
        
        # 验证不传 logprob 参数
        call_args = mock_adapter.engine.async_generate.call_args
        sampling_params = call_args[0][1]
        assert "return_logprob" not in sampling_params
        
        # logprobs 应为 None
        assert output.logprobs is None


# ============================================================
# 测试 ModelOutput logprobs 字段
# ============================================================

class TestModelOutputLogprobs:
    """测试 ModelOutput 的 logprobs 字段"""
    
    def test_logprobs_field_default_none(self):
        """logprobs 默认为 None"""
        from dki.models.base import ModelOutput
        output = ModelOutput(text="test")
        assert output.logprobs is None
    
    def test_logprobs_field_with_data(self):
        """logprobs 可以设置为 List[List[float]]"""
        from dki.models.base import ModelOutput
        lps = [[-0.5, -1.2], [-0.1, -3.0]]
        output = ModelOutput(text="test", logprobs=lps)
        assert output.logprobs == lps
        assert len(output.logprobs) == 2
        assert output.logprobs[0] == [-0.5, -1.2]


# ============================================================
# 测试 Entropy 计算兼容性
# ============================================================

class TestEntropyFromLogprobs:
    """验证 adapter 返回的 logprobs 可以被 EntropyMonitor 正确消费"""
    
    def test_vllm_logprobs_entropy_calculation(self):
        """vLLM logprobs → 熵计算"""
        from dki.models.vllm_adapter import VLLMAdapter
        
        # 模拟 vLLM 返回的 logprobs (top-5)
        raw_logprobs = [
            {
                1: MockVLLMLogprob(math.log(0.6)),   # 主 token 60%
                2: MockVLLMLogprob(math.log(0.2)),   # 20%
                3: MockVLLMLogprob(math.log(0.1)),   # 10%
                4: MockVLLMLogprob(math.log(0.05)),  # 5%
                5: MockVLLMLogprob(math.log(0.05)),  # 5%
            },
        ]
        
        parsed = VLLMAdapter._parse_vllm_logprobs(raw_logprobs)
        assert len(parsed) == 1
        assert len(parsed[0]) == 5
        
        # 验证可以从 logprobs 计算 Shannon 熵
        probs = [math.exp(lp) for lp in parsed[0]]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        assert entropy > 0  # 非零熵 (分布不是 delta)
        assert entropy < math.log(5)  # 不超过均匀分布熵
    
    def test_sglang_logprobs_entropy_calculation(self):
        """SGLang logprobs → 熵计算"""
        from dki.models.sglang_adapter import SGLangAdapter
        
        meta_info = {
            "output_top_logprobs": [
                {1: math.log(0.5), 2: math.log(0.3), 3: math.log(0.2)},
            ]
        }
        
        parsed = SGLangAdapter._parse_sglang_logprobs(meta_info)
        assert parsed is not None
        assert len(parsed) == 1
        
        # 验证可以计算 Shannon 熵
        probs = [math.exp(lp) for lp in parsed[0]]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        assert entropy > 0
        assert entropy < math.log(3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
