"""
Recall v4 集成测试

测试完整的 Recall v4 流程:
MultiSignalRecall → SuffixBuilder → FactRetriever
模拟端到端的记忆召回与后缀组装
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest

from dki.core.recall.recall_config import (
    RecallConfig,
    RecallResult,
    HistoryItem,
    AssembledSuffix,
    FactRequest,
    FactResponse,
)
from dki.core.recall.multi_signal_recall import MultiSignalRecall
from dki.core.recall.suffix_builder import SuffixBuilder
from dki.core.recall.fact_retriever import FactRetriever
from dki.core.recall.prompt_formatter import (
    GenericFormatter,
    create_formatter,
)


# ============================================================
# Mock 数据
# ============================================================

@dataclass
class FakeMessage:
    id: str
    content: str
    role: str = "user"
    created_at: str = "2025-01-01 12:00:00"
    message_id: Optional[str] = None


class FakeConversationRepo:
    def __init__(self, messages: List[FakeMessage] = None):
        self._messages = messages or []

    def get_by_session(self, session_id: str, **kwargs) -> List[FakeMessage]:
        return self._messages

    def get_recent(self, session_id: str, limit: int = 10, **kwargs) -> List[FakeMessage]:
        return self._messages[-limit:]

    def get_by_id(self, message_id: str) -> Optional[FakeMessage]:
        for msg in self._messages:
            if str(msg.id) == message_id:
                return msg
        return None


@dataclass
class FakeSearchResult:
    memory_id: str
    score: float
    content: str = ""


class FakeMemoryRouter:
    def __init__(self, results: List[FakeSearchResult] = None):
        self._results = results or []

    def search(self, query: str, top_k: int = 10) -> List[FakeSearchResult]:
        return self._results[:top_k]


# ============================================================
# 端到端测试
# ============================================================

class TestRecallV4EndToEnd:
    """Recall v4 端到端测试"""

    @pytest.fixture
    def conversation_messages(self):
        """模拟完整的会话历史"""
        return [
            FakeMessage(id="msg-001", content="你好，我想了解一下你们的会员制度", role="user"),
            FakeMessage(id="msg-002", content="您好！我们的会员分为三个等级：银卡、金卡和钻石卡。", role="assistant"),
            FakeMessage(id="msg-003", content="金卡有什么权益？", role="user"),
            FakeMessage(
                id="msg-004",
                content=(
                    "金卡会员享有以下权益：\n"
                    "1. 全场商品9折优惠\n"
                    "2. 每月赠送200积分\n"
                    "3. 生日当月双倍积分\n"
                    "4. 免费停车2小时\n"
                    "5. 专属客服热线：400-123-4567\n"
                    "6. 每季度一次免费体检\n"
                    "7. 合作酒店8折优惠\n"
                    "8. 机场贵宾厅使用权\n"
                    "9. 年度旅行保险\n"
                    "10. 积分可兑换礼品\n"
                    "详细信息请参阅会员手册第15-23页。"
                ) * 3,  # 重复 3 次使其超过阈值
                role="assistant",
            ),
            FakeMessage(id="msg-005", content="钻石卡呢？", role="user"),
            FakeMessage(id="msg-006", content="钻石卡在金卡基础上额外享有全场8折和专属管家服务。", role="assistant"),
            FakeMessage(id="msg-007", content="怎么升级到金卡？", role="user"),
            FakeMessage(id="msg-008", content="年消费满5万元即可自动升级为金卡会员。", role="assistant"),
        ]

    @pytest.fixture
    def config(self):
        return RecallConfig.from_dict({
            "enabled": True,
            "strategy": "summary_with_fact_call",
            "signals": {
                "keyword_enabled": False,  # 关闭关键词 (避免 jieba 依赖)
                "vector_enabled": True,
                "vector_threshold": 0.0,  # 接受所有向量结果
            },
            "budget": {
                "generation_reserve": 200,
                "instruction_reserve": 100,
                "min_recent_turns": 1,
            },
            "summary": {
                "per_message_threshold": 100,  # 低阈值
                "max_tokens_per_summary": 50,
                "strategy": "extractive",
            },
            "fact_call": {
                "enabled": True,
                "max_rounds": 3,
                "max_fact_tokens": 500,
                "batch_size": 5,
            },
        })

    # ============ 完整流程测试 ============

    def test_full_pipeline(self, config, conversation_messages):
        """完整流程: 召回 → 组装 → 事实检索"""
        repo = FakeConversationRepo(conversation_messages)
        router = FakeMemoryRouter([
            FakeSearchResult(memory_id="msg-004", score=0.9),
            FakeSearchResult(memory_id="msg-006", score=0.7),
        ])
        formatter = GenericFormatter(language="cn")

        # Phase 1: 多信号召回
        recaller = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        recall_result = recaller.recall(
            query="金卡会员有什么权益？",
            session_id="sess-001",
        )
        assert len(recall_result.messages) > 0

        # Phase 2: 后缀组装
        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )
        suffix = builder.build(
            query="金卡会员有什么权益？",
            recalled_messages=recall_result.messages,
            context_window=4096,
        )
        assert isinstance(suffix, AssembledSuffix)
        assert suffix.total_tokens > 0
        assert len(suffix.text) > 0
        assert "金卡会员有什么权益" in suffix.text

        # Phase 3: 事实检索 (模拟模型请求 trace_id)
        if suffix.trace_ids:
            retriever = FactRetriever(
                config=config,
                conversation_repo=repo,
            )
            fact_response = retriever.retrieve(
                trace_id=suffix.trace_ids[0],
                session_id="sess-001",
            )
            assert isinstance(fact_response, FactResponse)
            assert fact_response.trace_id == suffix.trace_ids[0]

    # ============ Summary 与原文混合测试 ============

    def test_mixed_summary_and_messages(self, config, conversation_messages):
        """长消息应生成 summary, 短消息保留原文"""
        repo = FakeConversationRepo(conversation_messages)
        router = FakeMemoryRouter([
            FakeSearchResult(memory_id="msg-004", score=0.9),  # 长消息
            FakeSearchResult(memory_id="msg-008", score=0.8),  # 短消息
        ])
        formatter = GenericFormatter(language="cn")

        recaller = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        recall_result = recaller.recall(
            query="升级条件",
            session_id="sess-001",
        )

        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )
        suffix = builder.build(
            query="升级条件",
            recalled_messages=recall_result.messages,
            context_window=4096,
        )

        # 应该有混合的 summary 和 message
        has_summary = suffix.summary_count > 0
        has_message = suffix.message_count > 0
        # 至少应该有一种类型
        assert has_summary or has_message

    # ============ Fact Call 检测测试 ============

    def test_fact_call_detection_in_suffix(self, config, conversation_messages):
        """后缀中有 summary 时应包含 retrieve_fact 指令"""
        repo = FakeConversationRepo(conversation_messages)
        router = FakeMemoryRouter([
            FakeSearchResult(memory_id="msg-004", score=0.9),
        ])
        formatter = GenericFormatter(language="cn")

        recaller = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        recall_result = recaller.recall(
            query="权益详情",
            session_id="sess-001",
        )

        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )
        suffix = builder.build(
            query="权益详情",
            recalled_messages=recall_result.messages,
            context_window=4096,
        )

        if suffix.summary_count > 0:
            assert suffix.has_fact_call_instruction is True
            assert "retrieve_fact" in suffix.text

    # ============ 模拟 Fact Call 循环测试 ============

    def test_fact_call_loop_simulation(self, config, conversation_messages):
        """模拟 Fact Call 循环"""
        repo = FakeConversationRepo(conversation_messages)
        formatter = GenericFormatter(language="cn")
        retriever = FactRetriever(config=config, conversation_repo=repo)

        # 模拟模型输出包含 function call
        model_output = '根据摘要，金卡有多项权益，但我需要确认具体信息。retrieve_fact(trace_id="msg-004", offset=0, limit=5)'

        # 检测 function call
        fact_request = formatter.detect_fact_request(model_output)
        assert fact_request is not None
        assert fact_request.trace_id == "msg-004"

        # 检索事实
        fact_response = retriever.retrieve(
            trace_id=fact_request.trace_id,
            session_id="sess-001",
            offset=fact_request.offset,
            limit=fact_request.limit,
        )
        assert fact_response.total_count > 0

        # 格式化事实段落
        fact_text = formatter.format_fact_segment(fact_response)
        assert "[FACT_SEGMENT" in fact_text
        assert 'trace_id="msg-004"' in fact_text

    # ============ 空会话测试 ============

    def test_empty_session(self, config):
        """空会话应正常返回"""
        repo = FakeConversationRepo([])
        router = FakeMemoryRouter([])
        formatter = GenericFormatter(language="cn")

        recaller = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        recall_result = recaller.recall(
            query="你好",
            session_id="sess-empty",
        )
        assert recall_result.messages == []

        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )
        suffix = builder.build(
            query="你好",
            recalled_messages=recall_result.messages,
            context_window=4096,
        )
        assert suffix.text == "你好"
        assert suffix.total_tokens == 0

    # ============ 小 Context Window 测试 ============

    def test_small_context_window(self, config, conversation_messages):
        """小 context window 应正确截断"""
        repo = FakeConversationRepo(conversation_messages)
        router = FakeMemoryRouter([
            FakeSearchResult(memory_id=f"msg-{i:03d}", score=0.9 - i * 0.1)
            for i in range(1, 9)
        ])
        formatter = GenericFormatter(language="cn")

        recaller = MultiSignalRecall(
            config=config,
            memory_router=router,
            conversation_repo=repo,
        )
        recall_result = recaller.recall(
            query="所有信息",
            session_id="sess-001",
        )

        builder = SuffixBuilder(
            config=config,
            prompt_formatter=formatter,
        )
        suffix = builder.build(
            query="所有信息",
            recalled_messages=recall_result.messages,
            context_window=500,  # 很小的窗口
        )

        # 不应包含所有消息
        total_items = suffix.message_count + suffix.summary_count
        assert total_items < len(conversation_messages)


class TestFormatterFactory:
    """格式化器工厂与不同模型的集成测试"""

    def test_create_formatter_for_deepseek(self):
        f = create_formatter(model_name="deepseek-v2-chat")
        assert isinstance(f, GenericFormatter)  # DeepSeekFormatter is a subclass
        # 验证 constraint 包含 tool definition
        constraint = f.format_constraint_instruction(["msg-001"])
        assert "TOOL_DEFINITION" in constraint

    def test_create_formatter_for_glm(self):
        f = create_formatter(model_name="chatglm3-6b")
        response = FactResponse(
            messages=[{"role": "user", "content": "test"}],
            trace_id="t1",
            total_count=1,
            offset=0,
            has_more=False,
        )
        result = f.format_fact_segment(response)
        assert "<|observation|>" in result

    def test_create_formatter_for_llama(self):
        f = create_formatter(model_name="llama-3-8b")
        assert type(f) is GenericFormatter  # 精确类型检查
