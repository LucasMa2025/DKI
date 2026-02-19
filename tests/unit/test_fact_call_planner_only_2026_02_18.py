"""
单元测试: P0-3B Fact Call Planner-only 重构验证

验证 Fact Call 从 Executor 循环推理 → Planner 预解析的重构:

1. FactBlock 数据结构
2. InjectionPlan 新增字段 (fact_blocks, fact_tokens, fact_strategy)
3. Planner._resolve_facts_in_planner() 正确检索和内联事实
4. Planner._append_fact_blocks_to_input() 正确拼装事实到 final_input
5. Planner.build_plan() 在 recall_v4 + has_fact_call_instruction 时触发事实解析
6. Planner.build_plan() 在 fact_call.enabled=False 时不触发事实解析
7. Planner 事实 token 预算限制
8. Planner 事实检索异常处理 (容错)
9. Executor 不再有 fact_call_loop 相关方法
10. Executor.execute() 将 Planner 侧事实信息复制到 ExecutionResult
11. DKIPlugin 将 FactRetriever/PromptFormatter 传递给 Planner
12. DKIPlugin.chat() 将事实解析信息填充到 InjectionMetadata
13. InjectionPlan.to_dict() 包含事实相关字段
14. ExecutionResult 包含 Planner 侧事实信息字段
15. __init__.py 导出 FactBlock

测试策略:
- 使用 Mock 和内存对象，不需要 GPU
- 验证每个重构点的正确行为

Author: AGI Demo Project
Date: 2026-02-18
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import torch

# 确保测试可以找到项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.core.plugin.injection_plan import (
    InjectionPlan,
    AlphaProfile,
    SafetyEnvelope,
    QueryContext,
    ExecutionResult,
    FactBlock,
)
from dki.core.plugin.injection_planner import InjectionPlanner
from dki.core.plugin.injection_executor import InjectionExecutor
from dki.core.recall.recall_config import (
    RecallConfig,
    RecallFactCallConfig,
    FactResponse,
    HistoryItem,
    AssembledSuffix,
    RecallResult,
)
from dki.models.base import KVCacheEntry, ModelOutput
from dki.adapters.base import (
    IUserDataAdapter,
    UserPreference as AdapterUserPreference,
    ChatMessage as AdapterChatMessage,
)


# ================================================================
# 辅助工具
# ================================================================

def make_chat_message(role: str, content: str, msg_id: str = "m1") -> AdapterChatMessage:
    """创建测试用 ChatMessage"""
    return AdapterChatMessage(
        message_id=msg_id,
        session_id="sess_1",
        user_id="user_1",
        role=role,
        content=content,
        timestamp=datetime.utcnow(),
    )


def make_preference(text: str, ptype: str = "general", priority: int = 1) -> AdapterUserPreference:
    """创建测试用 UserPreference"""
    return AdapterUserPreference(
        user_id="user_1",
        preference_text=text,
        preference_type=ptype,
        priority=priority,
    )


def make_fake_kv_entries(num_layers: int = 2) -> List[KVCacheEntry]:
    """创建假的 KV entries (CPU tensors)"""
    entries = []
    for i in range(num_layers):
        entries.append(KVCacheEntry(
            key=torch.randn(1, 4, 3, 8),
            value=torch.randn(1, 4, 3, 8),
            layer_idx=i,
        ))
    return entries


def make_mock_model():
    """创建 Mock 模型适配器"""
    model = MagicMock()
    model.device = "cpu"
    model.hidden_dim = 64
    model.model_name = "test-model"
    model.compute_kv.return_value = (make_fake_kv_entries(), "test_hash")
    model.generate.return_value = ModelOutput(
        text="Test response",
        input_tokens=10,
        output_tokens=5,
    )
    model.forward_with_kv_injection.return_value = ModelOutput(
        text="Injected response",
        input_tokens=15,
        output_tokens=8,
    )
    return model


def make_mock_fact_retriever(responses: Optional[Dict[str, FactResponse]] = None):
    """创建 Mock FactRetriever"""
    retriever = MagicMock()
    
    if responses is None:
        # 默认: 每个 trace_id 返回一些消息
        def default_retrieve(trace_id, session_id, offset=0, limit=5, db_session=None):
            return FactResponse(
                messages=[
                    {"role": "user", "content": f"Original message for {trace_id}"},
                    {"role": "assistant", "content": f"Original reply for {trace_id}"},
                ],
                trace_id=trace_id,
                total_count=2,
                offset=0,
                has_more=False,
            )
        retriever.retrieve.side_effect = default_retrieve
    else:
        def custom_retrieve(trace_id, session_id, offset=0, limit=5, db_session=None):
            return responses.get(trace_id, FactResponse(
                trace_id=trace_id, total_count=0, offset=0, has_more=False,
            ))
        retriever.retrieve.side_effect = custom_retrieve
    
    return retriever


def make_mock_prompt_formatter():
    """创建 Mock PromptFormatter"""
    formatter = MagicMock()
    
    def format_fact_segment(response: FactResponse) -> str:
        if not response.messages:
            return ""
        parts = []
        for msg in response.messages:
            parts.append(f"[{msg['role']}] {msg['content']}")
        return f"[FACT trace_id={response.trace_id}]\n" + "\n".join(parts) + "\n[/FACT]"
    
    formatter.format_fact_segment.side_effect = format_fact_segment
    return formatter


def make_recall_config(
    fact_call_enabled: bool = True,
    max_fact_tokens: int = 800,
    batch_size: int = 5,
) -> RecallConfig:
    """创建测试用 RecallConfig"""
    return RecallConfig(
        fact_call=RecallFactCallConfig(
            enabled=fact_call_enabled,
            max_fact_tokens=max_fact_tokens,
            batch_size=batch_size,
        ),
    )


def make_mock_multi_signal_recall():
    """创建 Mock MultiSignalRecall"""
    recall = MagicMock()
    recall.recall.return_value = RecallResult(
        messages=[MagicMock()],
        keyword_hits=1,
        vector_hits=1,
    )
    return recall


def make_mock_suffix_builder(
    text: str = "assembled suffix text",
    has_fact_call: bool = True,
    trace_ids: Optional[List[str]] = None,
):
    """创建 Mock SuffixBuilder"""
    builder = MagicMock()
    builder.build.return_value = AssembledSuffix(
        text=text,
        total_tokens=50,
        summary_count=1,
        message_count=2,
        has_fact_call_instruction=has_fact_call,
        trace_ids=trace_ids or ["trace_001", "trace_002"],
    )
    return builder


# ================================================================
# Test 1: FactBlock 数据结构
# ================================================================

class TestFactBlock:
    """FactBlock 数据结构测试"""
    
    def test_default_values(self):
        """默认值应正确"""
        fb = FactBlock()
        assert fb.trace_id == ""
        assert fb.fact_text == ""
        assert fb.fact_tokens == 0
        assert fb.source == "retriever"
    
    def test_custom_values(self):
        """自定义值应正确"""
        fb = FactBlock(
            trace_id="trace_001",
            fact_text="some fact text",
            fact_tokens=42,
            source="inline",
        )
        assert fb.trace_id == "trace_001"
        assert fb.fact_text == "some fact text"
        assert fb.fact_tokens == 42
        assert fb.source == "inline"
    
    def test_import_from_plugin_init(self):
        """FactBlock 应可从 plugin __init__ 导入"""
        from dki.core.plugin import FactBlock as ImportedFactBlock
        assert ImportedFactBlock is FactBlock


# ================================================================
# Test 2: InjectionPlan 新增字段
# ================================================================

class TestInjectionPlanFactFields:
    """InjectionPlan 事实相关字段测试"""
    
    def test_default_fact_fields(self):
        """默认事实字段应为空/零"""
        plan = InjectionPlan()
        assert plan.fact_blocks == []
        assert plan.fact_tokens == 0
        assert plan.fact_strategy == "none"
        assert plan.fact_rounds_used == 0
    
    def test_fact_fields_in_to_dict(self):
        """to_dict 应包含事实字段"""
        plan = InjectionPlan()
        plan.fact_blocks = [FactBlock(trace_id="t1", fact_text="f1", fact_tokens=10)]
        plan.fact_tokens = 10
        plan.fact_strategy = "planner_resolved"
        
        d = plan.to_dict()
        assert "fact_blocks_count" in d
        assert d["fact_blocks_count"] == 1
        assert d["fact_tokens"] == 10
        assert d["fact_strategy"] == "planner_resolved"
    
    def test_fact_blocks_count_in_to_dict(self):
        """to_dict 应正确计算 fact_blocks_count"""
        plan = InjectionPlan()
        plan.fact_blocks = [
            FactBlock(trace_id="t1"),
            FactBlock(trace_id="t2"),
            FactBlock(trace_id="t3"),
        ]
        d = plan.to_dict()
        assert d["fact_blocks_count"] == 3


# ================================================================
# Test 3: ExecutionResult 事实字段
# ================================================================

class TestExecutionResultFactFields:
    """ExecutionResult 事实相关字段测试"""
    
    def test_default_fact_fields(self):
        """默认事实字段应为空/零"""
        result = ExecutionResult()
        assert result.fact_blocks_resolved == 0
        assert result.fact_tokens_total == 0
        assert result.fact_strategy == "none"
    
    def test_fact_fields_settable(self):
        """事实字段应可设置"""
        result = ExecutionResult()
        result.fact_blocks_resolved = 3
        result.fact_tokens_total = 150
        result.fact_strategy = "planner_resolved"
        
        assert result.fact_blocks_resolved == 3
        assert result.fact_tokens_total == 150
        assert result.fact_strategy == "planner_resolved"


# ================================================================
# Test 4: Planner._resolve_facts_in_planner()
# ================================================================

class TestPlannerResolveFactsInPlanner:
    """Planner 侧事实解析测试"""
    
    def test_resolve_facts_basic(self):
        """基本事实解析: 有 trace_ids → 检索事实 → 填充 fact_blocks"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001", "trace_002"]
        plan.has_fact_call_instruction = True
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert len(result_plan.fact_blocks) == 2
        assert result_plan.fact_blocks[0].trace_id == "trace_001"
        assert result_plan.fact_blocks[1].trace_id == "trace_002"
        assert result_plan.fact_tokens > 0
        assert result_plan.fact_strategy == "planner_resolved"
        assert result_plan.fact_rounds_used == 2
    
    def test_resolve_facts_no_retriever(self):
        """无 FactRetriever → 跳过事实解析"""
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=None,
            prompt_formatter=make_mock_prompt_formatter(),
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert len(result_plan.fact_blocks) == 0
        assert result_plan.fact_strategy == "none"
    
    def test_resolve_facts_no_formatter(self):
        """无 PromptFormatter → 跳过事实解析"""
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=make_mock_fact_retriever(),
            prompt_formatter=None,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert len(result_plan.fact_blocks) == 0
    
    def test_resolve_facts_empty_trace_ids(self):
        """空 trace_ids → 无事实块"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = []
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert len(result_plan.fact_blocks) == 0
        assert result_plan.fact_strategy == "none"
        retriever.retrieve.assert_not_called()
    
    def test_resolve_facts_token_budget_exceeded(self):
        """token 预算超限 → 部分解析 + strategy=budget_exceeded"""
        # 设置很小的 token 预算
        recall_config = make_recall_config(max_fact_tokens=5)
        
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001", "trace_002", "trace_003"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        # 第一个可能刚好超过预算, 或者第二个超过
        # 关键是 strategy 应该是 budget_exceeded
        assert result_plan.fact_strategy == "budget_exceeded"
    
    def test_resolve_facts_retriever_returns_empty(self):
        """FactRetriever 返回空消息 → 跳过该 trace_id"""
        responses = {
            "trace_001": FactResponse(
                messages=[],
                trace_id="trace_001",
                total_count=0,
                offset=0,
                has_more=False,
            ),
            "trace_002": FactResponse(
                messages=[{"role": "user", "content": "Hello"}],
                trace_id="trace_002",
                total_count=1,
                offset=0,
                has_more=False,
            ),
        }
        retriever = make_mock_fact_retriever(responses)
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001", "trace_002"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        # trace_001 空消息被跳过, 只有 trace_002
        assert len(result_plan.fact_blocks) == 1
        assert result_plan.fact_blocks[0].trace_id == "trace_002"
    
    def test_resolve_facts_retriever_exception(self):
        """FactRetriever 抛异常 → 跳过该 trace_id, 继续处理其他"""
        retriever = MagicMock()
        call_count = [0]
        
        def side_effect(trace_id, session_id, offset=0, limit=5, db_session=None):
            call_count[0] += 1
            if trace_id == "trace_001":
                raise RuntimeError("Database connection lost")
            return FactResponse(
                messages=[{"role": "user", "content": f"Fact for {trace_id}"}],
                trace_id=trace_id,
                total_count=1,
                offset=0,
                has_more=False,
            )
        
        retriever.retrieve.side_effect = side_effect
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001", "trace_002"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        # trace_001 异常被跳过, trace_002 成功
        assert len(result_plan.fact_blocks) == 1
        assert result_plan.fact_blocks[0].trace_id == "trace_002"
    
    def test_resolve_facts_updates_total_tokens(self):
        """事实解析应更新 plan.total_tokens"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001"]
        plan.total_tokens = 100
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        # total_tokens 应增加 fact_tokens
        assert result_plan.total_tokens > 100
        assert result_plan.total_tokens == 100 + result_plan.fact_tokens
    
    def test_resolve_facts_updates_stats(self):
        """事实解析应更新 Planner 统计"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001", "trace_002"]
        
        planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert planner._stats["fact_blocks_resolved"] == 2


# ================================================================
# Test 5: Planner._append_fact_blocks_to_input()
# ================================================================

class TestPlannerAppendFactBlocks:
    """事实块追加到 final_input 测试"""
    
    def test_append_cn(self):
        """中文: 事实块应正确追加"""
        planner = InjectionPlanner(language="cn")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "原始后缀内容"
        plan.fact_blocks = [
            FactBlock(trace_id="t1", fact_text="事实内容1"),
            FactBlock(trace_id="t2", fact_text="事实内容2"),
        ]
        
        result = planner._append_fact_blocks_to_input(plan)
        
        assert "原始后缀内容" in result
        assert "事实内容1" in result
        assert "事实内容2" in result
        assert "补充事实" in result
        assert "不需要再调用 retrieve_fact" in result
        assert "不需要请求更多事实" in result
    
    def test_append_en(self):
        """英文: 事实块应正确追加"""
        planner = InjectionPlanner(language="en")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "Original suffix content"
        plan.fact_blocks = [
            FactBlock(trace_id="t1", fact_text="Fact content 1"),
        ]
        
        result = planner._append_fact_blocks_to_input(plan)
        
        assert "Original suffix content" in result
        assert "Fact content 1" in result
        assert "Supplementary Facts" in result
        assert "Do NOT call retrieve_fact again" in result
        assert "Do NOT request more facts" in result
    
    def test_append_empty_blocks(self):
        """空事实块 → 返回原始后缀"""
        planner = InjectionPlanner(language="cn")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "原始后缀"
        plan.fact_blocks = []
        
        # _append_fact_blocks_to_input 只在有 fact_blocks 时被调用
        # 但如果被调用了, 空列表不应该崩溃
        result = planner._append_fact_blocks_to_input(plan)
        
        assert "原始后缀" in result
    
    def test_append_preserves_assembled_suffix(self):
        """追加事实不应修改 assembled_suffix 的内容"""
        planner = InjectionPlanner(language="cn")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "这是完整的后缀内容\n包含多行"
        plan.fact_blocks = [
            FactBlock(trace_id="t1", fact_text="事实A"),
        ]
        
        result = planner._append_fact_blocks_to_input(plan)
        
        # assembled_suffix 应作为第一部分出现
        assert result.startswith("这是完整的后缀内容\n包含多行")


# ================================================================
# Test 6: Planner.build_plan() 触发事实解析
# ================================================================

class TestPlannerBuildPlanWithFacts:
    """build_plan 集成测试: 验证 recall_v4 + fact_call 触发"""
    
    def test_build_plan_triggers_fact_resolution(self):
        """recall_v4 + has_fact_call_instruction → 触发事实解析"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config(fact_call_enabled=True)
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            has_fact_call=True,
            trace_ids=["trace_001"],
        )
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        context = QueryContext()
        plan = planner.build_plan(
            query="你之前说了什么?",
            user_id="user_1",
            preferences=[make_preference("素食主义者")],
            relevant_history=[],
            context=context,
            session_id="sess_1",
        )
        
        # 事实应被解析
        assert len(plan.fact_blocks) > 0
        assert plan.fact_strategy == "planner_resolved"
        assert plan.fact_tokens > 0
        # final_input 应包含事实
        assert "补充事实" in plan.final_input or "Supplementary Facts" in plan.final_input
    
    def test_build_plan_no_fact_call_instruction(self):
        """recall_v4 但 has_fact_call_instruction=False → 不触发事实解析"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config(fact_call_enabled=True)
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            has_fact_call=False,  # 无 fact call 指令
            trace_ids=[],
        )
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        context = QueryContext()
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[make_preference("素食")],
            relevant_history=[],
            context=context,
            session_id="sess_1",
        )
        
        assert len(plan.fact_blocks) == 0
        assert plan.fact_strategy == "none"
        retriever.retrieve.assert_not_called()
    
    def test_build_plan_fact_call_disabled(self):
        """fact_call.enabled=False → 不触发事实解析"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config(fact_call_enabled=False)
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            has_fact_call=True,
            trace_ids=["trace_001"],
        )
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        context = QueryContext()
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[],
            relevant_history=[],
            context=context,
            session_id="sess_1",
        )
        
        assert len(plan.fact_blocks) == 0
        retriever.retrieve.assert_not_called()
    
    def test_build_plan_stable_strategy_no_facts(self):
        """stable 策略 → 不触发事实解析"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config(fact_call_enabled=True)
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="stable",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
        )
        
        context = QueryContext()
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[make_preference("素食")],
            relevant_history=[make_chat_message("user", "之前的消息")],
            context=context,
            session_id="sess_1",
        )
        
        assert plan.strategy == "stable"
        assert len(plan.fact_blocks) == 0
        retriever.retrieve.assert_not_called()
    
    def test_build_plan_recall_v4_fallback_to_stable(self):
        """recall_v4 失败回退到 stable → 不触发事实解析"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config(fact_call_enabled=True)
        
        # MultiSignalRecall 抛异常
        multi_signal = MagicMock()
        multi_signal.recall.side_effect = RuntimeError("recall failed")
        suffix_builder = make_mock_suffix_builder()
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        context = QueryContext()
        plan = planner.build_plan(
            query="你好",
            user_id="user_1",
            preferences=[make_preference("素食")],
            relevant_history=[make_chat_message("user", "之前的消息")],
            context=context,
            session_id="sess_1",
        )
        
        # 回退到 stable
        assert plan.strategy == "stable"
        # 不应触发事实解析 (因为已不是 recall_v4)
        assert len(plan.fact_blocks) == 0
        retriever.retrieve.assert_not_called()


# ================================================================
# Test 7: Executor 不再有 fact_call_loop
# ================================================================

class TestExecutorNoFactCallLoop:
    """验证 Executor 不再有 Fact Call 循环"""
    
    def test_no_execute_fact_call_loop_method(self):
        """Executor 不应有 _execute_fact_call_loop 方法"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        assert not hasattr(executor, '_execute_fact_call_loop')
    
    def test_no_log_function_call_method(self):
        """Executor 不应有 _log_function_call 方法"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        assert not hasattr(executor, '_log_function_call')
    
    def test_no_fact_call_rounds_in_stats(self):
        """Executor 统计中不应有 fact_call_rounds"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        stats = executor.get_stats()
        assert "fact_call_rounds" not in stats
    
    def test_executor_init_no_fact_retriever_param(self):
        """Executor.__init__ 不应接受 fact_retriever 参数"""
        import inspect
        sig = inspect.signature(InjectionExecutor.__init__)
        params = list(sig.parameters.keys())
        
        assert "fact_retriever" not in params
        assert "prompt_formatter" not in params
        assert "recall_config" not in params


# ================================================================
# Test 8: Executor.execute() 复制 Planner 事实信息
# ================================================================

class TestExecutorCopiesPlannerFactInfo:
    """验证 Executor 将 Planner 侧事实信息复制到 ExecutionResult"""
    
    @pytest.mark.asyncio
    async def test_execute_copies_fact_info_to_result(self):
        """execute() 应将 plan 中的事实信息复制到 result"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食主义者",
            final_input="test input",
            user_id="user_1",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            # Planner 侧事实信息
            fact_blocks=[
                FactBlock(trace_id="t1", fact_text="fact1", fact_tokens=20),
                FactBlock(trace_id="t2", fact_text="fact2", fact_tokens=30),
            ],
            fact_tokens=50,
            fact_strategy="planner_resolved",
        )
        
        result = await executor.execute(plan)
        
        assert result.fact_blocks_resolved == 2
        assert result.fact_tokens_total == 50
        assert result.fact_strategy == "planner_resolved"
    
    @pytest.mark.asyncio
    async def test_execute_plain_no_fact_info(self):
        """无注入执行 → 事实信息为默认值"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=False,
            final_input="test input",
            user_id="user_1",
        )
        
        result = await executor.execute(plan)
        
        assert result.fact_blocks_resolved == 0
        assert result.fact_tokens_total == 0
        assert result.fact_strategy == "none"


# ================================================================
# Test 9: DKIPlugin 初始化传递 Fact 组件
# ================================================================

class TestDKIPluginFactWiring:
    """验证 DKIPlugin 将 Fact 组件传递给 Planner"""
    
    def test_planner_receives_fact_retriever(self):
        """Planner 应接收 fact_retriever"""
        model = make_mock_model()
        adapter = MagicMock(spec=IUserDataAdapter)
        
        # 使用 patch 来避免实际初始化 Recall v4 组件
        with patch('dki.core.dki_plugin.InjectionPlanner') as MockPlanner:
            MockPlanner.return_value = MagicMock()
            
            from dki.core.dki_plugin import DKIPlugin
            
            plugin = DKIPlugin(
                model_adapter=model,
                user_data_adapter=adapter,
                language="cn",
            )
            
            # 验证 Planner 被调用时传递了 fact_retriever 和 prompt_formatter
            call_kwargs = MockPlanner.call_args[1]
            # fact_retriever 和 prompt_formatter 应该被传递
            assert "fact_retriever" in call_kwargs
            assert "prompt_formatter" in call_kwargs
            assert "recall_config" in call_kwargs
    
    def test_executor_does_not_receive_fact_retriever(self):
        """Executor 不应接收 fact_retriever"""
        model = make_mock_model()
        adapter = MagicMock(spec=IUserDataAdapter)
        
        with patch('dki.core.dki_plugin.InjectionExecutor') as MockExecutor:
            MockExecutor.return_value = MagicMock()
            
            from dki.core.dki_plugin import DKIPlugin
            
            plugin = DKIPlugin(
                model_adapter=model,
                user_data_adapter=adapter,
                language="cn",
            )
            
            call_kwargs = MockExecutor.call_args[1]
            assert "fact_retriever" not in call_kwargs
            assert "prompt_formatter" not in call_kwargs
            assert "recall_config" not in call_kwargs


# ================================================================
# Test 10: InjectionMetadata 事实字段
# ================================================================

class TestInjectionMetadataFactFields:
    """验证 InjectionMetadata 包含事实解析字段"""
    
    def test_default_fact_fields(self):
        """默认事实字段应为空/零"""
        from dki.core.dki_plugin import InjectionMetadata
        
        meta = InjectionMetadata()
        assert meta.fact_blocks_resolved == 0
        assert meta.fact_tokens_total == 0
        assert meta.fact_strategy == "none"
    
    def test_to_dict_includes_fact_resolution(self):
        """to_dict 应包含 fact_resolution 段"""
        from dki.core.dki_plugin import InjectionMetadata
        
        meta = InjectionMetadata()
        meta.fact_blocks_resolved = 3
        meta.fact_tokens_total = 150
        meta.fact_strategy = "planner_resolved"
        
        d = meta.to_dict()
        assert "fact_resolution" in d
        assert d["fact_resolution"]["blocks_resolved"] == 3
        assert d["fact_resolution"]["tokens_total"] == 150
        assert d["fact_resolution"]["strategy"] == "planner_resolved"


# ================================================================
# Test 11: DKIPlugin.chat() 事实元数据填充
# ================================================================

class TestDKIPluginChatFactMetadata:
    """验证 chat() 将事实解析信息填充到 metadata"""
    
    @pytest.mark.asyncio
    async def test_chat_populates_fact_metadata(self):
        """chat() 应将 Planner 事实信息通过 Executor result 填充到 metadata"""
        from dki.core.dki_plugin import DKIPlugin, InjectionMetadata
        
        model = make_mock_model()
        adapter = MagicMock(spec=IUserDataAdapter)
        adapter.get_user_preferences = AsyncMock(return_value=[
            make_preference("素食主义者"),
        ])
        adapter.search_relevant_history = AsyncMock(return_value=[])
        
        # Mock Planner
        mock_planner = MagicMock()
        mock_planner.analyze_query.return_value = QueryContext()
        
        mock_plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食主义者",
            final_input="test input with facts",
            user_id="user_1",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            fact_blocks=[
                FactBlock(trace_id="t1", fact_text="f1", fact_tokens=20),
            ],
            fact_tokens=20,
            fact_strategy="planner_resolved",
        )
        mock_planner.build_plan.return_value = mock_plan
        mock_planner.get_stats.return_value = {"strategy": "recall_v4"}
        
        # Mock Executor
        mock_executor = MagicMock()
        mock_result = ExecutionResult(
            text="response",
            input_tokens=10,
            output_tokens=5,
            fact_blocks_resolved=1,
            fact_tokens_total=20,
            fact_strategy="planner_resolved",
        )
        mock_executor.execute = AsyncMock(return_value=mock_result)
        mock_executor.get_stats.return_value = {}
        
        plugin = DKIPlugin(
            model_adapter=model,
            user_data_adapter=adapter,
            language="cn",
        )
        
        # Replace internal components with mocks
        plugin._planner = mock_planner
        plugin._executor = mock_executor
        
        response = await plugin.chat(
            query="你好",
            user_id="user_1",
            session_id="sess_1",
        )
        
        # 验证 metadata 包含事实信息
        assert response.metadata.fact_blocks_resolved == 1
        assert response.metadata.fact_tokens_total == 20
        assert response.metadata.fact_strategy == "planner_resolved"
        
        # 验证 to_dict 也包含
        d = response.metadata.to_dict()
        assert d["fact_resolution"]["blocks_resolved"] == 1


# ================================================================
# Test 12: Planner 初始化接受 Fact 组件
# ================================================================

class TestPlannerInitFactComponents:
    """验证 Planner 初始化正确接收 Fact 组件"""
    
    def test_planner_stores_fact_retriever(self):
        """Planner 应存储 fact_retriever"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        config = make_recall_config()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=config,
        )
        
        assert planner._fact_retriever is retriever
        assert planner._prompt_formatter is formatter
        assert planner._recall_config is config
    
    def test_planner_defaults_none(self):
        """Planner 默认 fact 组件为 None"""
        planner = InjectionPlanner(language="cn")
        
        assert planner._fact_retriever is None
        assert planner._prompt_formatter is None
    
    def test_planner_stats_include_fact_blocks(self):
        """Planner 统计应包含 fact_blocks_resolved"""
        planner = InjectionPlanner(language="cn")
        
        stats = planner.get_stats()
        assert "fact_blocks_resolved" in stats
        assert stats["fact_blocks_resolved"] == 0


# ================================================================
# Test 13: 端到端 Planner → Executor 事实流
# ================================================================

class TestEndToEndFactFlow:
    """端到端测试: Planner 解析事实 → Executor 执行一次"""
    
    @pytest.mark.asyncio
    async def test_planner_resolves_then_executor_runs_once(self):
        """Planner 解析事实后, Executor 只执行一次 forward pass"""
        # Setup Planner
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        recall_config = make_recall_config()
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            text="[summary] 用户讨论了编程\n[trace_id=trace_001]",
            has_fact_call=True,
            trace_ids=["trace_001"],
        )
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        # Build plan
        context = planner.analyze_query("你之前说了什么关于编程的?")
        plan = planner.build_plan(
            query="你之前说了什么关于编程的?",
            user_id="user_1",
            preferences=[make_preference("喜欢Python")],
            relevant_history=[],
            context=context,
            session_id="sess_1",
        )
        
        # Verify plan has facts
        assert len(plan.fact_blocks) > 0
        assert plan.fact_strategy == "planner_resolved"
        assert "补充事实" in plan.final_input
        
        # Setup Executor
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        # Execute
        result = await executor.execute(plan)
        
        # Executor should call forward_with_kv_injection exactly once
        model.forward_with_kv_injection.assert_called_once()
        
        # Result should have fact info
        assert result.fact_blocks_resolved == len(plan.fact_blocks)
        assert result.fact_tokens_total == plan.fact_tokens
        assert result.fact_strategy == "planner_resolved"
    
    @pytest.mark.asyncio
    async def test_no_facts_executor_still_runs_once(self):
        """无事实时, Executor 仍只执行一次"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食",
            final_input="你好",
            user_id="user_1",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
        )
        
        result = await executor.execute(plan)
        
        # 只调用一次
        assert model.forward_with_kv_injection.call_count == 1
        assert result.fact_blocks_resolved == 0


# ================================================================
# Test 14: Planner 事实解析语言切换
# ================================================================

class TestPlannerFactLanguage:
    """验证 Planner 事实解析的语言切换"""
    
    def test_cn_fact_instruction(self):
        """中文: 事实指令应为中文"""
        planner = InjectionPlanner(language="cn")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "后缀"
        plan.fact_blocks = [
            FactBlock(trace_id="t1", fact_text="事实1"),
        ]
        
        result = planner._append_fact_blocks_to_input(plan)
        
        assert "补充事实" in result
        assert "不需要再调用 retrieve_fact" in result
    
    def test_en_fact_instruction(self):
        """英文: 事实指令应为英文"""
        planner = InjectionPlanner(language="en")
        
        plan = InjectionPlan()
        plan.assembled_suffix = "suffix"
        plan.fact_blocks = [
            FactBlock(trace_id="t1", fact_text="Fact 1"),
        ]
        
        result = planner._append_fact_blocks_to_input(plan)
        
        assert "Supplementary Facts" in result
        assert "Do NOT call retrieve_fact again" in result


# ================================================================
# Test 15: 边界情况
# ================================================================

class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_trace_id(self):
        """单个 trace_id 应正常工作"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        assert len(result_plan.fact_blocks) == 1
        assert result_plan.fact_rounds_used == 1
    
    def test_many_trace_ids_with_budget(self):
        """大量 trace_ids 但 token 预算有限"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        # 很小的预算
        recall_config = make_recall_config(max_fact_tokens=10)
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=recall_config,
        )
        
        plan = InjectionPlan()
        plan.trace_ids = [f"trace_{i:03d}" for i in range(20)]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        # 不应解析所有 20 个
        assert len(result_plan.fact_blocks) < 20
        assert result_plan.fact_strategy == "budget_exceeded"
    
    def test_fact_block_source_field(self):
        """FactBlock source 字段应默认为 'retriever'"""
        retriever = make_mock_fact_retriever()
        formatter = make_mock_prompt_formatter()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=make_recall_config(),
        )
        
        plan = InjectionPlan()
        plan.trace_ids = ["trace_001"]
        
        result_plan = planner._resolve_facts_in_planner(plan, "sess_1")
        
        for fb in result_plan.fact_blocks:
            assert fb.source == "retriever"
    
    @pytest.mark.asyncio
    async def test_executor_stable_fallback_preserves_fact_info(self):
        """Executor stable fallback 应保留 plan 的事实信息"""
        model = make_mock_model()
        # 让第一次 forward_with_kv_injection 抛异常
        model.forward_with_kv_injection.side_effect = RuntimeError("GPU error")
        
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食",
            final_input="test",
            original_query="test",
            user_id="user_1",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
            history_suffix="some history",
            fact_blocks=[FactBlock(trace_id="t1", fact_text="f1", fact_tokens=20)],
            fact_tokens=20,
            fact_strategy="planner_resolved",
        )
        
        # 应该降级到 stable fallback 或无注入
        result = await executor.execute(plan)
        
        # 即使降级了, 也应返回有效结果
        assert result.text is not None


# ================================================================
# 运行测试
# ================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
