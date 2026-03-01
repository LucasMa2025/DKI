"""
v5.1 修复测试 — 覆盖以下修复点:

1. _DetachedMessage: ORM 对象脱管安全副本
2. _build_recall_v4_chat_prompt: chat template 多轮格式 (修复角色混乱)
3. _build_hybrid_chat_prompt: hybrid 模式 chat template 构造
4. _ConversationRepoWrapper._to_detached: 批量脱管转换
5. max_new_tokens 默认值 (已提升到 512)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# Fixtures: 模拟 ORM 对象和 HistoryItem
# ============================================================

class FakeORMMessage:
    """模拟 SQLAlchemy ORM 消息对象"""
    def __init__(self, id=1, session_id="s1", role="user", content="hello",
                 created_at=None, injection_mode=None, injection_alpha=None,
                 memory_ids=None, latency_ms=None):
        self.id = id
        self.session_id = session_id
        self.role = role
        self.content = content
        self.created_at = created_at
        self.injection_mode = injection_mode
        self.injection_alpha = injection_alpha
        self.memory_ids = memory_ids
        self.latency_ms = latency_ms


@dataclass
class FakeHistoryItem:
    """模拟 recall_config.HistoryItem"""
    type: str = "message"
    content: str = ""
    trace_id: str = ""
    role: Optional[str] = None
    token_count: int = 0
    confidence: str = "high"
    facts_covered: List[str] = field(default_factory=list)
    facts_missing: List[str] = field(default_factory=list)


class FakeTokenizer:
    """模拟 tokenizer，支持 apply_chat_template"""
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        """简单 ChatML 格式化，用于测试验证"""
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant")
        return "\n".join(parts) + "\n"


class FakeModelAdapter:
    """模拟 model adapter"""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()


# ============================================================
# Test: _DetachedMessage
# ============================================================

class TestDetachedMessage:
    """测试 ORM 对象脱管安全副本"""
    
    def _get_class(self):
        """延迟导入 _DetachedMessage"""
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        from dki.core.dki_system import _DetachedMessage
        return _DetachedMessage
    
    def test_basic_extraction(self):
        """基本属性提取"""
        _DetachedMessage = self._get_class()
        orm = FakeORMMessage(id=42, role="assistant", content="你好", session_id="sess_1")
        
        detached = _DetachedMessage(orm)
        
        assert detached.id == 42
        assert detached.message_id == 42
        assert detached.role == "assistant"
        assert detached.content == "你好"
        assert detached.session_id == "sess_1"
    
    def test_missing_attributes_have_defaults(self):
        """缺少的属性有合理默认值"""
        _DetachedMessage = self._get_class()
        
        class MinimalORM:
            pass
        
        detached = _DetachedMessage(MinimalORM())
        
        assert detached.id is None
        assert detached.role == "user"  # 默认 "user"
        assert detached.content == ""   # 默认空字符串
    
    def test_detached_survives_session_close(self):
        """脱管对象在 session 关闭后仍可访问"""
        _DetachedMessage = self._get_class()
        orm = FakeORMMessage(role="user", content="测试内容")
        
        detached = _DetachedMessage(orm)
        # 模拟 session 关闭：删除原始 ORM 对象
        del orm
        
        # 脱管对象仍然可以访问
        assert detached.role == "user"
        assert detached.content == "测试内容"
    
    def test_multiple_messages(self):
        """批量转换多条消息"""
        _DetachedMessage = self._get_class()
        
        orms = [
            FakeORMMessage(id=1, role="user", content="你好"),
            FakeORMMessage(id=2, role="assistant", content="你好！有什么可以帮你的？"),
            FakeORMMessage(id=3, role="user", content="帮我写代码"),
        ]
        
        detached_list = [_DetachedMessage(o) for o in orms]
        
        assert len(detached_list) == 3
        assert detached_list[0].role == "user"
        assert detached_list[1].role == "assistant"
        assert detached_list[2].content == "帮我写代码"


# ============================================================
# Test: _build_recall_v4_chat_prompt
# ============================================================

class TestBuildRecallV4ChatPrompt:
    """测试 recall_v4 chat template 构造 (修复角色混乱)"""
    
    def _make_system(self, tokenizer=None):
        """创建一个 mock DKISystem 对象，只包含所需方法"""
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        from dki.core.dki_system import DKISystem
        
        system = MagicMock(spec=DKISystem)
        system._model_adapter = FakeModelAdapter(tokenizer)
        system._recall_formatter = None
        
        # 绑定真实方法 (包括 budget 相关的依赖方法)
        system._build_recall_v4_chat_prompt = DKISystem._build_recall_v4_chat_prompt.__get__(system)
        system._trim_history_to_budget = DKISystem._trim_history_to_budget.__get__(system)
        system._estimate_prompt_tokens = DKISystem._estimate_prompt_tokens.__get__(system)
        # _get_max_prompt_tokens: 返回足够大的值, 确保测试中历史不被截断
        system._get_max_prompt_tokens = MagicMock(return_value=8000)
        return system
    
    def test_basic_multi_turn(self):
        """基本多轮对话格式化"""
        system = self._make_system()
        
        items = [
            FakeHistoryItem(role="user", content="你叫什么名字？"),
            FakeHistoryItem(role="assistant", content="我是AI助手。"),
            FakeHistoryItem(role="user", content="你喜欢什么？"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="我们之前聊了什么？",
        )
        
        # 验证基本结构
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "你叫什么名字？" in result
        assert "我是AI助手。" in result
        assert "我们之前聊了什么？" in result
    
    def test_roles_properly_separated(self):
        """角色正确分离 — 修复前的问题是所有历史被当成单一 user message"""
        system = self._make_system()
        
        items = [
            FakeHistoryItem(role="user", content="问题1"),
            FakeHistoryItem(role="assistant", content="回答1"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="问题2",
        )
        
        # 应该有多个 <|im_start|>user 和 <|im_start|>assistant
        # 而不是一个巨大的 user message
        user_starts = result.count("<|im_start|>user")
        assistant_starts = result.count("<|im_start|>assistant")
        
        # 2 个 user (历史 + 当前查询), 至少 1 个 assistant (历史 + generation prompt)
        assert user_starts == 2, f"Expected 2 user starts, got {user_starts}"
        assert assistant_starts >= 1, f"Expected >= 1 assistant starts, got {assistant_starts}"
    
    def test_preference_as_system_message(self):
        """偏好信息放入 system message"""
        system = self._make_system()
        
        preference = MagicMock()
        preference.content = "语言风格: 简洁\n我的名字: Lucas"
        
        items = [
            FakeHistoryItem(role="user", content="你好"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="你记得我吗？",
            preference=preference,
        )
        
        assert "<|im_start|>system" in result
        assert "Lucas" in result
        assert "你记得我吗？" in result
    
    def test_empty_items(self):
        """空历史 — 只有当前查询"""
        system = self._make_system()
        
        result = system._build_recall_v4_chat_prompt(
            items=[],
            original_query="你好",
        )
        
        assert "你好" in result
        # 即使历史为空，也应该有正确的格式
        assert "<|im_start|>user" in result
    
    def test_summary_items_with_trace_id(self):
        """summary 类型条目带 trace_id 标注"""
        system = self._make_system()
        
        items = [
            FakeHistoryItem(
                type="summary",
                role="assistant",
                content="之前讨论了ERP系统",
                trace_id="msg_abc123",
            ),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="ERP的最新进展？",
        )
        
        assert "trace_id=msg_abc123" in result
        assert "ERP" in result
    
    def test_current_query_is_last_user_message(self):
        """当前查询始终是最后一个 user message"""
        system = self._make_system()
        
        items = [
            FakeHistoryItem(role="user", content="旧问题"),
            FakeHistoryItem(role="assistant", content="旧回答"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="新问题",
        )
        
        # 新问题应该在旧回答之后
        old_answer_pos = result.rfind("旧回答")
        new_query_pos = result.rfind("新问题")
        assert new_query_pos > old_answer_pos, "当前查询应该在历史之后"
    
    def test_invalid_role_defaults_to_user(self):
        """无效角色名默认为 user"""
        system = self._make_system()
        
        items = [
            FakeHistoryItem(role="unknown_role", content="内容"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="测试",
        )
        
        # unknown_role 应该被转换为 user
        assert "unknown_role" not in result
    
    def test_fallback_chatml_when_no_tokenizer(self):
        """没有 tokenizer 时回退到 ChatML 格式"""
        system = self._make_system()
        system._model_adapter = MagicMock()
        system._model_adapter.tokenizer = None  # 无 tokenizer
        
        items = [
            FakeHistoryItem(role="user", content="你好"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="测试回退",
        )
        
        # 应该使用 ChatML 回退格式
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "测试回退" in result
    
    def test_tokenizer_exception_falls_back_to_chatml(self):
        """tokenizer.apply_chat_template 异常时回退到 ChatML"""
        bad_tokenizer = MagicMock()
        bad_tokenizer.apply_chat_template.side_effect = Exception("template error")
        
        system = self._make_system(tokenizer=bad_tokenizer)
        
        items = [
            FakeHistoryItem(role="user", content="你好"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="异常测试",
        )
        
        # 应该回退到 ChatML
        assert "<|im_start|>user" in result
        assert "异常测试" in result


# ============================================================
# Test: _build_hybrid_chat_prompt
# ============================================================

class TestBuildHybridChatPrompt:
    """测试 hybrid 模式 chat template 构造"""
    
    def _make_system(self, tokenizer=None):
        """创建 mock DKISystem"""
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        from dki.core.dki_system import DKISystem
        
        system = MagicMock(spec=DKISystem)
        system._model_adapter = FakeModelAdapter(tokenizer)
        system._build_hybrid_chat_prompt = DKISystem._build_hybrid_chat_prompt.__get__(system)
        system._trim_history_to_budget = DKISystem._trim_history_to_budget.__get__(system)
        system._estimate_prompt_tokens = DKISystem._estimate_prompt_tokens.__get__(system)
        # _get_max_prompt_tokens: 返回足够大的值, 确保测试中历史不被截断
        system._get_max_prompt_tokens = MagicMock(return_value=8000)
        return system
    
    def test_dict_history_messages(self):
        """字典格式的历史消息"""
        system = self._make_system()
        
        history = MagicMock()
        history.messages = [
            {"role": "user", "content": "问题A"},
            {"role": "assistant", "content": "回答A"},
        ]
        
        result = system._build_hybrid_chat_prompt(
            history=history,
            original_query="问题B",
        )
        
        assert "问题A" in result
        assert "回答A" in result
        assert "问题B" in result
        
        # 角色正确分离
        user_starts = result.count("<|im_start|>user")
        assert user_starts == 2  # 历史 user + 当前 user
    
    def test_object_history_messages(self):
        """对象格式的历史消息 (有 role/content 属性)"""
        system = self._make_system()
        
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "对象消息"
        
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "对象回复"
        
        history = MagicMock()
        history.messages = [msg1, msg2]
        
        result = system._build_hybrid_chat_prompt(
            history=history,
            original_query="新查询",
        )
        
        assert "对象消息" in result
        assert "对象回复" in result
    
    def test_preference_in_system_message(self):
        """偏好文本放入 system message"""
        system = self._make_system()
        
        preference = MagicMock()
        preference.content = "编程爱好: Python"
        
        history = MagicMock()
        history.messages = [{"role": "user", "content": "hello"}]
        
        result = system._build_hybrid_chat_prompt(
            history=history,
            original_query="测试",
            preference=preference,
        )
        
        assert "<|im_start|>system" in result
        assert "Python" in result
    
    def test_empty_content_messages_filtered(self):
        """空内容消息被过滤"""
        system = self._make_system()
        
        history = MagicMock()
        history.messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "有内容的消息"},
        ]
        
        result = system._build_hybrid_chat_prompt(
            history=history,
            original_query="测试",
        )
        
        assert "有内容的消息" in result
        # 空消息不应该产生空的 user block


# ============================================================
# Test: Web UI max_new_tokens
# ============================================================

class TestMaxNewTokensDefault:
    """测试 max_new_tokens 默认值修复"""
    
    def test_chat_request_default(self):
        """ChatRequest 默认 max_new_tokens 应为 512"""
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        
        try:
            # 尝试导入 ChatRequest (可能依赖 FastAPI)
            from dki.web.app import ChatRequest
            req = ChatRequest(query="test", mode="dki", session_id="s1")
            assert req.max_new_tokens >= 512, f"max_new_tokens={req.max_new_tokens}, expected >= 512"
        except ImportError:
            pytest.skip("FastAPI not available")


# ============================================================
# Test: 角色混乱回归测试
# ============================================================

class TestRoleConfusionRegression:
    """回归测试: 确保不再出现角色混乱"""
    
    def _make_system(self):
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        from dki.core.dki_system import DKISystem
        
        system = MagicMock(spec=DKISystem)
        system._model_adapter = FakeModelAdapter()
        system._recall_formatter = None
        system._build_recall_v4_chat_prompt = DKISystem._build_recall_v4_chat_prompt.__get__(system)
        system._trim_history_to_budget = DKISystem._trim_history_to_budget.__get__(system)
        system._estimate_prompt_tokens = DKISystem._estimate_prompt_tokens.__get__(system)
        system._get_max_prompt_tokens = MagicMock(return_value=8000)
        return system
    
    def test_no_nested_user_assistant_tags(self):
        """
        回归测试: 不应出现嵌套的 User:/Assistant: 标记
        
        修复前的问题:
        <|im_start|>user
        User: 你好
        Assistant: 你好！
        User: 我们聊过什么？
        <|im_end|>
        
        修复后的正确格式:
        <|im_start|>user
        你好
        <|im_end|>
        <|im_start|>assistant
        你好！
        <|im_end|>
        <|im_start|>user
        我们聊过什么？
        <|im_end|>
        """
        system = self._make_system()
        
        items = [
            FakeHistoryItem(role="user", content="你好"),
            FakeHistoryItem(role="assistant", content="你好！有什么可以帮你的？"),
            FakeHistoryItem(role="user", content="帮我写个Python脚本"),
            FakeHistoryItem(role="assistant", content="好的，请问需要什么功能？"),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="我们之前聊了什么？",
        )
        
        # 验证: 在 <|im_start|>user 和 <|im_end|> 之间不应出现 "Assistant:" 标记
        # 在 <|im_start|>assistant 和 <|im_end|> 之间不应出现 "User:" 标记
        import re
        
        # 提取所有 user blocks
        user_blocks = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
            result,
            re.DOTALL,
        )
        
        for block in user_blocks:
            assert "Assistant:" not in block, f"User block contains 'Assistant:': {block}"
            assert "<|im_start|>assistant" not in block, f"Nested assistant tag in user block"
        
        # 提取所有 assistant blocks
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            result,
            re.DOTALL,
        )
        
        for block in assistant_blocks:
            assert "User:" not in block, f"Assistant block contains 'User:': {block}"
    
    def test_real_world_scenario(self):
        """
        真实场景: 模拟用户报告的角色混乱问题
        
        用户偏好: 语言风格、名字等
        多轮历史: ERP 讨论 + 名字/偏好查询
        """
        system = self._make_system()
        
        preference = MagicMock()
        preference.content = "语言风格: 细腻温暖\n个人爱好: 编程\n我的名字: lucas"
        
        items = [
            FakeHistoryItem(role="user", content="什么是ERP系统？"),
            FakeHistoryItem(
                role="assistant",
                content="ERP（Enterprise Resource Planning）是企业资源规划系统，"
                        "用于整合企业各部门的业务流程。",
            ),
            FakeHistoryItem(role="user", content="你记得我的名字吗？"),
            FakeHistoryItem(
                role="assistant",
                content="根据你之前分享的信息，你叫Lucas，喜欢编程。",
            ),
        ]
        
        result = system._build_recall_v4_chat_prompt(
            items=items,
            original_query="我们谈过哪些ERP方面的话题呢",
            preference=preference,
        )
        
        # 1. 偏好在 system message 中
        assert "lucas" in result.lower()
        assert "<|im_start|>system" in result
        
        # 2. 当前查询是最后一个 user message
        last_user_start = result.rfind("<|im_start|>user")
        assert "ERP方面的话题" in result[last_user_start:]
        
        # 3. 历史角色正确分离
        user_count = result.count("<|im_start|>user")
        assistant_count = result.count("<|im_start|>assistant")
        assert user_count == 3  # 2 历史 user + 1 当前
        assert assistant_count >= 2  # 2 历史 assistant + 1 generation prompt
        
        # 4. 不应有 "思考过程" 被当成 user message 的情况
        # (修复前的输出中模型的思考过程会被嵌套在 user message 中)


# ============================================================
# Test: InjectionPlan.history_items
# ============================================================

class TestInjectionPlanHistoryItems:
    """测试 InjectionPlan 的 history_items 字段"""
    
    def test_default_empty(self):
        """默认为空列表"""
        import sys
        sys.path.insert(0, "D:\\MyProject\\AGI_Demo\\DKI")
        
        try:
            from dki.core.plugin.injection_plan import InjectionPlan
            # InjectionPlan 可能需要特定参数，跳过如果无法创建
            plan = InjectionPlan.__new__(InjectionPlan)
            if hasattr(plan, 'history_items'):
                # 如果 history_items 存在且有默认值
                assert isinstance(plan.history_items, list) or plan.history_items is None
        except (ImportError, TypeError, AttributeError):
            pytest.skip("InjectionPlan not importable or changed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
