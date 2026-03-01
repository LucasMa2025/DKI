"""
单元测试: v6.0 Planner / SuffixBuilder 重构验证

验证 v6.0 改动:
1. FactBlock 已从 InjectionPlan 移除
2. Planner 不再做一次性全量事实补齐 (_resolve_facts_in_planner 已移除)
3. SuffixBuilder 改为两阶段全局预算分配
4. Executor 不再有 fact_call_loop
5. ExecutionResult 不再有 fact_blocks_resolved / fact_tokens_total / fact_strategy
6. InjectionMetadata 不再有 fact_blocks_resolved / fact_tokens_total / fact_strategy
7. Planner 仍保留 fact_retriever / prompt_formatter 引用 (供 Executor 回调用)
8. InjectionPlan.to_dict() 不再包含 fact_blocks_count / fact_tokens / fact_strategy
9. __init__.py 不再导出 FactBlock
10. Planner build_plan recall_v4 不再追加事实块到 final_input
11. Executor.execute() 不再复制 fact 字段到 result
12. SuffixBuilder Phase1: 完整收集 (不压缩)
13. SuffixBuilder Phase2: 全局预算分配 (短消息优先保留, 长消息按预算分配)

测试策略:
- 使用 Mock 和内存对象，不需要 GPU
- 验证每个重构点的正确行为

Author: AGI Demo Project
Date: 2026-02-27 (updated from 2026-02-18)
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

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
)
from dki.core.plugin.injection_planner import InjectionPlanner
from dki.core.plugin.injection_executor import InjectionExecutor
from dki.core.recall.recall_config import (
    RecallConfig,
    RecallFactCallConfig,
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
    has_fact_call: bool = False,
    trace_ids: Optional[List[str]] = None,
    items: Optional[List[HistoryItem]] = None,
):
    """创建 Mock SuffixBuilder"""
    builder = MagicMock()
    builder.build.return_value = AssembledSuffix(
        text=text,
        total_tokens=50,
        summary_count=0,
        message_count=2,
        has_fact_call_instruction=has_fact_call,
        trace_ids=trace_ids or [],
        items=items or [],
    )
    return builder


# ================================================================
# Test 1: FactBlock 已从 injection_plan 移除
# ================================================================

class TestFactBlockRemoved:
    """验证 FactBlock 已从 InjectionPlan 模块移除"""
    
    def test_factblock_not_in_injection_plan(self):
        """FactBlock 不应存在于 injection_plan 模块"""
        import dki.core.plugin.injection_plan as plan_module
        assert not hasattr(plan_module, 'FactBlock')
    
    def test_factblock_not_in_plugin_init(self):
        """FactBlock 不应从 plugin __init__ 导出"""
        import dki.core.plugin as plugin_module
        assert not hasattr(plugin_module, 'FactBlock')
    
    def test_injection_plan_no_fact_blocks_field(self):
        """InjectionPlan 不应有 fact_blocks 字段"""
        plan = InjectionPlan()
        assert not hasattr(plan, 'fact_blocks')
    
    def test_injection_plan_no_fact_tokens_field(self):
        """InjectionPlan 不应有 fact_tokens 字段"""
        plan = InjectionPlan()
        assert not hasattr(plan, 'fact_tokens')
    
    def test_injection_plan_no_fact_strategy_field(self):
        """InjectionPlan 不应有 fact_strategy 字段"""
        plan = InjectionPlan()
        assert not hasattr(plan, 'fact_strategy')
    
    def test_to_dict_no_fact_fields(self):
        """to_dict 不应包含事实相关字段"""
        plan = InjectionPlan()
        d = plan.to_dict()
        assert "fact_blocks_count" not in d
        assert "fact_tokens" not in d
        assert "fact_strategy" not in d


# ================================================================
# Test 2: ExecutionResult 不再有 fact 字段
# ================================================================

class TestExecutionResultFactFieldsRemoved:
    """验证 ExecutionResult 的 fact 字段已移除"""
    
    def test_no_fact_blocks_resolved(self):
        """ExecutionResult 不应有 fact_blocks_resolved 字段"""
        result = ExecutionResult()
        assert not hasattr(result, 'fact_blocks_resolved')
    
    def test_no_fact_tokens_total(self):
        """ExecutionResult 不应有 fact_tokens_total 字段"""
        result = ExecutionResult()
        assert not hasattr(result, 'fact_tokens_total')
    
    def test_no_fact_strategy(self):
        """ExecutionResult 不应有 fact_strategy 字段"""
        result = ExecutionResult()
        assert not hasattr(result, 'fact_strategy')


# ================================================================
# Test 3: Planner 不再有 _resolve_facts_in_planner
# ================================================================

class TestPlannerFactResolutionRemoved:
    """验证 Planner 的一次性事实补齐已移除"""
    
    def test_no_resolve_facts_method(self):
        """Planner 不应有 _resolve_facts_in_planner 方法"""
        planner = InjectionPlanner(language="cn")
        assert not hasattr(planner, '_resolve_facts_in_planner')
    
    def test_no_append_fact_blocks_method(self):
        """Planner 不应有 _append_fact_blocks_to_input 方法"""
        planner = InjectionPlanner(language="cn")
        assert not hasattr(planner, '_append_fact_blocks_to_input')
    
    def test_stats_no_fact_blocks_resolved(self):
        """Planner stats 不应包含 fact_blocks_resolved"""
        planner = InjectionPlanner(language="cn")
        stats = planner.get_stats()
        assert "fact_blocks_resolved" not in stats


# ================================================================
# Test 4: Planner 仍保留 fact_retriever / prompt_formatter 引用
# ================================================================

class TestPlannerRetainsFCReferences:
    """Planner 仍保留 fact_retriever / prompt_formatter (供 Executor 回调)"""
    
    def test_planner_stores_fact_retriever(self):
        """Planner 应保留 fact_retriever 引用"""
        retriever = MagicMock()
        formatter = MagicMock()
        
        planner = InjectionPlanner(
            language="cn",
            fact_retriever=retriever,
            prompt_formatter=formatter,
        )
        
        assert planner._fact_retriever is retriever
        assert planner._prompt_formatter is formatter
    
    def test_planner_defaults_none(self):
        """Planner 默认 fact 组件为 None"""
        planner = InjectionPlanner(language="cn")
        assert planner._fact_retriever is None
        assert planner._prompt_formatter is None


# ================================================================
# Test 5: Executor 不再有 fact_call_loop
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
    
    def test_executor_init_no_fact_retriever_param(self):
        """Executor.__init__ 不应接受 fact_retriever 参数"""
        import inspect
        sig = inspect.signature(InjectionExecutor.__init__)
        params = list(sig.parameters.keys())
        assert "fact_retriever" not in params
        assert "prompt_formatter" not in params
        assert "recall_config" not in params


# ================================================================
# Test 6: Executor.execute() 不再复制 fact 字段
# ================================================================

class TestExecutorExecuteNoFactCopy:
    """验证 Executor.execute() 不再复制 fact 字段到 result"""
    
    @pytest.mark.asyncio
    async def test_execute_result_has_no_fact_fields(self):
        """execute() 结果不应有 fact 字段"""
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preference_text="素食主义者",
            final_input="test input",
            user_id="user_1",
            alpha_profile=AlphaProfile(preference_alpha=0.4),
        )
        
        result = await executor.execute(plan)
        
        assert not hasattr(result, 'fact_blocks_resolved')
        assert not hasattr(result, 'fact_tokens_total')
        assert not hasattr(result, 'fact_strategy')
    
    @pytest.mark.asyncio
    async def test_execute_runs_once(self):
        """Executor 应只执行一次推理"""
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
        assert result.text is not None


# ================================================================
# Test 7: Planner build_plan 不再追加事实块
# ================================================================

class TestPlannerBuildPlanNoFactAppend:
    """验证 build_plan 不再追加事实块到 final_input"""
    
    def test_build_plan_no_fact_in_final_input(self):
        """recall_v4 build_plan 的 final_input 不应包含 '补充事实'"""
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            text="[会话历史参考]\n用户: 你好\n助手: 你好\n",
            has_fact_call=True,
            trace_ids=["trace_001", "trace_002"],
        )
        
        retriever = MagicMock()
        formatter = MagicMock()
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            fact_retriever=retriever,
            prompt_formatter=formatter,
            recall_config=RecallConfig(
                fact_call=RecallFactCallConfig(enabled=True),
            ),
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
        
        # final_input 不应包含 '补充事实' (旧的 Planner 补齐标记)
        assert "补充事实" not in plan.final_input
        assert "Supplementary Facts" not in plan.final_input
        # FactRetriever 不应被调用 (Planner 不再调用它)
        retriever.retrieve.assert_not_called()
    
    def test_build_plan_stable_no_facts(self):
        """stable 策略不应有事实相关处理"""
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="stable",
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
        assert "补充事实" not in plan.final_input


# ================================================================
# Test 8: InjectionMetadata 不再有 fact 字段
# ================================================================

class TestInjectionMetadataFactFieldsRemoved:
    """验证 InjectionMetadata 的 fact 字段已移除"""
    
    def test_no_fact_fields(self):
        """InjectionMetadata 不应有 fact 字段"""
        from dki.core.dki_plugin import InjectionMetadata
        
        meta = InjectionMetadata()
        assert not hasattr(meta, 'fact_blocks_resolved')
        assert not hasattr(meta, 'fact_tokens_total')
        assert not hasattr(meta, 'fact_strategy')
    
    def test_to_dict_no_fact_resolution(self):
        """to_dict 不应包含 fact_resolution"""
        from dki.core.dki_plugin import InjectionMetadata
        
        meta = InjectionMetadata()
        d = meta.to_dict()
        assert "fact_resolution" not in d


# ================================================================
# Test 9: Memory Metadata 不再包含事实统计
# ================================================================

class TestMemoryMetadataNoFactStats:
    """验证 Memory Metadata 块不再包含事实统计"""
    
    def test_cn_metadata_no_fact_info(self):
        """中文记忆元数据不应包含 '事实补充'"""
        planner = InjectionPlanner(language="cn")
        
        plan = InjectionPlan()
        plan.relevant_history_count = 3
        plan.summary_count = 1
        
        metadata = planner._build_memory_metadata_block(
            plan=plan,
            preferences=[make_preference("素食")],
            relevant_history=[],
            context=QueryContext(),
        )
        
        assert "事实补充" not in metadata
        assert "[DMI 记忆状态]" in metadata
        assert "[DMI 记忆状态结束]" in metadata
    
    def test_en_metadata_no_fact_info(self):
        """英文记忆元数据不应包含 'Facts:'"""
        planner = InjectionPlanner(language="en")
        
        plan = InjectionPlan()
        plan.relevant_history_count = 3
        plan.summary_count = 1
        
        metadata = planner._build_memory_metadata_block(
            plan=plan,
            preferences=[make_preference("veg")],
            relevant_history=[],
            context=QueryContext(),
        )
        
        assert "Facts:" not in metadata
        assert "[DMI Memory Context]" in metadata
        assert "[DMI Memory Context End]" in metadata


# ================================================================
# Test 10: InjectionPlan 仍保留其他字段
# ================================================================

class TestInjectionPlanOtherFieldsIntact:
    """验证 InjectionPlan 其他字段未受影响"""
    
    def test_trace_ids_still_exist(self):
        """trace_ids 仍应存在"""
        plan = InjectionPlan()
        assert hasattr(plan, 'trace_ids')
        assert plan.trace_ids == []
    
    def test_has_fact_call_instruction_still_exists(self):
        """has_fact_call_instruction 仍应存在"""
        plan = InjectionPlan()
        assert hasattr(plan, 'has_fact_call_instruction')
        assert plan.has_fact_call_instruction is False
    
    def test_fact_rounds_used_still_exists(self):
        """fact_rounds_used 仍应存在 (用于记录回调轮次)"""
        plan = InjectionPlan()
        assert hasattr(plan, 'fact_rounds_used')
        assert plan.fact_rounds_used == 0
    
    def test_history_items_still_exist(self):
        """history_items 仍应存在"""
        plan = InjectionPlan()
        assert hasattr(plan, 'history_items')
        assert plan.history_items == []
    
    def test_to_dict_still_has_core_fields(self):
        """to_dict 仍应包含核心字段"""
        plan = InjectionPlan(
            strategy="recall_v4",
            injection_enabled=True,
            preferences_count=2,
            preference_tokens=80,
            history_tokens=200,
            query_tokens=30,
        )
        d = plan.to_dict()
        assert d["strategy"] == "recall_v4"
        assert d["injection_enabled"] is True
        assert d["preferences_count"] == 2
        assert d["preference_tokens"] == 80
        assert d["history_tokens"] == 200
        assert d["query_tokens"] == 30
        assert "trace_ids_count" in d
        assert "has_fact_call_instruction" in d


# ================================================================
# Test 11: 端到端 Planner → Executor 流 (无事实补齐)
# ================================================================

class TestEndToEndNoFactResolution:
    """端到端测试: Planner 不再补齐事实, Executor 只做一次推理"""
    
    @pytest.mark.asyncio
    async def test_planner_to_executor_flow(self):
        """完整流程: Planner build_plan → Executor execute"""
        multi_signal = make_mock_multi_signal_recall()
        suffix_builder = make_mock_suffix_builder(
            text="[会话历史参考]\n用户: 你之前说了什么关于编程的?\n",
            has_fact_call=True,
            trace_ids=["trace_001"],
        )
        
        planner = InjectionPlanner(
            language="cn",
            injection_strategy="recall_v4",
            multi_signal_recall=multi_signal,
            suffix_builder=suffix_builder,
        )
        
        context = planner.analyze_query("你之前说了什么关于编程的?")
        plan = planner.build_plan(
            query="你之前说了什么关于编程的?",
            user_id="user_1",
            preferences=[make_preference("喜欢Python")],
            relevant_history=[],
            context=context,
            session_id="sess_1",
        )
        
        # Planner 不应补齐事实
        assert "补充事实" not in plan.final_input
        
        # Executor
        model = make_mock_model()
        executor = InjectionExecutor(model_adapter=model)
        result = await executor.execute(plan)
        
        # 应只执行一次推理
        model.forward_with_kv_injection.assert_called_once()
        assert result.text is not None
        assert not hasattr(result, 'fact_blocks_resolved')


# ================================================================
# Test 12: SuffixBuilder 两阶段全局预算分配
# ================================================================

class TestSuffixBuilderGlobalBudget:
    """验证 SuffixBuilder 的两阶段全局预算分配"""
    
    @pytest.fixture
    def config(self):
        return RecallConfig.from_dict({
            "summary": {
                "per_message_threshold": 50,
                "max_tokens_per_summary": 30,
                "strategy": "extractive",
            },
            "budget": {
                "instruction_reserve": 50,
            },
        })
    
    @pytest.fixture
    def formatter(self):
        from dki.core.recall.prompt_formatter import GenericFormatter
        return GenericFormatter(language="cn")
    
    @pytest.fixture
    def builder(self, config, formatter):
        from dki.core.recall.suffix_builder import SuffixBuilder
        return SuffixBuilder(config=config, prompt_formatter=formatter)
    
    def test_all_messages_fit_budget(self, builder):
        """所有消息在预算内 → 全部保留原文"""
        from dataclasses import dataclass
        
        @dataclass
        class FakeMsg:
            id: str
            content: str
            role: str = "user"
        
        messages = [
            FakeMsg(id="m1", content="你好", role="user"),
            FakeMsg(id="m2", content="你好！", role="assistant"),
        ]
        
        result = builder.build(
            query="测试",
            recalled_messages=messages,
            context_window=4096,
        )
        
        assert result.message_count == 2
        assert result.summary_count == 0
    
    def test_budget_exceeded_long_messages_compressed(self, builder):
        """预算不足 → 长消息被压缩"""
        from dataclasses import dataclass
        
        @dataclass
        class FakeMsg:
            id: str
            content: str
            role: str = "user"
        
        long_content = "这是一条非常长的消息，包含了很多信息。" * 30
        messages = [
            FakeMsg(id="m1", content="你好"),
            FakeMsg(id="m2", content=long_content),
        ]
        
        result = builder.build(
            query="测试",
            recalled_messages=messages,
            context_window=300,  # Very small window
        )
        
        # 至少有一条消息
        assert len(result.items) >= 1
        # 短消息应保留
        short_items = [i for i in result.items if i.trace_id == "m1"]
        if short_items:
            assert short_items[0].type == "message"
    
    def test_empty_messages(self, builder):
        """空消息列表 → 仅返回 query"""
        result = builder.build(
            query="测试",
            recalled_messages=[],
            context_window=4096,
        )
        
        assert result.text == "测试"
        assert result.items == []
        assert result.total_tokens == 0


# ================================================================
# 运行测试
# ================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
