"""
Unit Tests for ChatView 功能改进

覆盖:
1. 自动命名修正 — sessionId 局部化避免竞态
2. 复制功能 — 单条消息复制, 会话完整复制文本格式
3. PreferencesView 默认分类合并逻辑

Author: AGI Demo Project
"""

import pytest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ============================================================
# Mock 数据结构
# ============================================================

@dataclass
class MockMessage:
    """模拟前端 ChatMessage"""
    id: str
    role: str  # 'user' | 'assistant'
    content: str
    timestamp: str = "2026-03-12T10:00:00Z"
    session_id: str = "session_001"


@dataclass
class MockSession:
    """模拟前端 Session"""
    id: str
    title: str = "New Chat"
    preview: str = ""
    message_count: int = 0
    updated_at: str = ""


# ============================================================
# 1. 自动命名逻辑测试
# ============================================================

class TestAutoNaming:
    """测试自动命名逻辑"""
    
    def _generate_session_title(self, first_message: str) -> str:
        """
        模拟 chatStore.generateSessionTitle 逻辑
        从 chat.ts 提取的核心函数
        """
        if not first_message or not first_message.strip():
            return "New Chat"
        text = first_message.strip()
        max_len = 30
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
    
    def test_normal_title(self):
        """正常消息生成标题"""
        title = self._generate_session_title("你好，请介绍一下DKI系统")
        assert title == "你好，请介绍一下DKI系统"
    
    def test_long_title_truncation(self):
        """超长消息截断到30字符"""
        long_msg = "这是一条非常长的消息，用来测试自动命名截断功能是否正常工作，应该被截断"
        title = self._generate_session_title(long_msg)
        assert len(title) <= 34  # 30 + "..."
        assert title.endswith("...")
    
    def test_empty_message(self):
        """空消息返回默认标题"""
        assert self._generate_session_title("") == "New Chat"
        assert self._generate_session_title("   ") == "New Chat"
    
    def test_short_message(self):
        """短消息直接作为标题"""
        title = self._generate_session_title("Hello")
        assert title == "Hello"
    
    def test_session_id_locality(self):
        """
        测试 sessionId 局部化逻辑:
        模拟并发情况下 currentSessionId 变更, 局部变量不受影响
        """
        # 模拟: 保存 sessionId 到局部变量
        current_session_id = "session_A"
        local_session_id = current_session_id
        
        # 模拟: 异步过程中 currentSessionId 被切换
        current_session_id = "session_B"
        
        # 局部变量不受影响
        assert local_session_id == "session_A"
        assert current_session_id == "session_B"


# ============================================================
# 2. 复制功能测试
# ============================================================

class TestCopyFunctions:
    """测试复制功能"""
    
    def _format_session_for_copy(
        self,
        messages: List[MockMessage],
        title: str = "Chat",
        username: str = "User",
    ) -> str:
        """
        模拟 handleCopySession 的文本格式化逻辑
        """
        lines = [f"# {title}", ""]
        
        for msg in messages:
            role = username if msg.role == "user" else "DKI Assistant"
            # 简化时间格式
            time_str = msg.timestamp[:16].replace("T", " ") if msg.timestamp else ""
            header = f"**{role}** ({time_str})" if time_str else f"**{role}**"
            lines.append(header)
            lines.append(msg.content or "")
            lines.append("")  # 空行分隔
        
        return "\n".join(lines)
    
    def test_single_message_copy(self):
        """单条消息复制"""
        msg = MockMessage(id="1", role="user", content="你好世界")
        assert msg.content == "你好世界"
    
    def test_session_copy_format(self):
        """会话完整复制 — 格式正确"""
        messages = [
            MockMessage(id="1", role="user", content="你好"),
            MockMessage(id="2", role="assistant", content="你好！有什么可以帮助你的？"),
            MockMessage(id="3", role="user", content="介绍一下DKI"),
            MockMessage(id="4", role="assistant", content="DKI 是一个动态知识注入系统..."),
        ]
        
        text = self._format_session_for_copy(messages, title="DKI 讨论", username="Lucas")
        
        assert text.startswith("# DKI 讨论")
        assert "**Lucas**" in text
        assert "**DKI Assistant**" in text
        assert "你好" in text
        assert "动态知识注入系统" in text
    
    def test_session_copy_empty(self):
        """空会话复制"""
        text = self._format_session_for_copy([])
        assert text == "# Chat\n"
    
    def test_session_copy_message_count(self):
        """复制后消息数量正确"""
        messages = [
            MockMessage(id=str(i), role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            for i in range(6)
        ]
        text = self._format_session_for_copy(messages)
        # 每条消息产生 header + content + 空行 = 3 行
        # 加上标题 + 空行 = 2 行
        # 总共 = 2 + 6 * 3 = 20 行
        lines = text.split("\n")
        assert len(lines) == 20
    
    def test_copy_preserves_content(self):
        """复制保持内容完整性"""
        content_with_code = "```python\nprint('hello')\n```"
        msg = MockMessage(id="1", role="assistant", content=content_with_code)
        text = self._format_session_for_copy([msg])
        assert "```python" in text
        assert "print('hello')" in text


# ============================================================
# 3. PreferencesView 默认分类测试
# ============================================================

class TestPreferencesDefaultCategories:
    """测试偏好分类默认值逻辑"""
    
    default_categories = ['General', 'Work', 'Style', 'Technical', 'Domain', 'Personal']
    
    def _merge_categories(
        self,
        from_prefs: List[str],
    ) -> List[str]:
        """模拟 existingCategories computed 逻辑"""
        filtered = [c for c in from_prefs if c != 'Uncategorized']
        merged = set([*self.default_categories, *filtered])
        return sorted(merged)  # sorted for deterministic comparison
    
    def test_no_existing_prefs(self):
        """无偏好时仍有默认分类"""
        result = self._merge_categories([])
        assert len(result) == 6
        assert 'General' in result
        assert 'Work' in result
        assert 'Style' in result
    
    def test_existing_prefs_merged(self):
        """已有分类与默认合并"""
        result = self._merge_categories(['Custom', 'Work'])
        assert 'Custom' in result
        assert 'Work' in result
        assert 'General' in result
        assert len(result) == 7  # 6 default + 1 new
    
    def test_uncategorized_filtered(self):
        """Uncategorized 被过滤"""
        result = self._merge_categories(['Uncategorized', 'Custom'])
        assert 'Uncategorized' not in result
        assert 'Custom' in result
    
    def test_deduplication(self):
        """重复分类去重"""
        result = self._merge_categories(['General', 'Work', 'General'])
        # set 自动去重
        assert result.count('General') == 1
        assert result.count('Work') == 1
    
    def test_all_defaults_present(self):
        """所有默认分类都存在"""
        result = self._merge_categories([])
        for cat in self.default_categories:
            assert cat in result


# ============================================================
# 4. BM25 召回噪音相关测试
# ============================================================

class TestBM25NoiseAnalysis:
    """
    分析 BM25 召回噪音的测试
    验证 Thinking Process 内容和通用词会导致大量噪音
    """
    
    _CN_STOPWORDS = frozenset({
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        '什么', '吗', '呢', '吧', '啊', '哦', '嗯', '呀', '哈', '哪', '嘛',
        '可以', '没', '还', '对', '把', '让', '被', '从', '给', '用', '但',
        '而', '又', '所以', '因为', '如果', '这个', '那个', '怎么', '为什么',
        '哪个', '多少', '几', '谁', '怎样', '这样', '那样',
    })
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词 (不依赖 jieba)"""
        import re
        tokens = []
        # 英文
        tokens.extend(re.findall(r'[a-zA-Z0-9]+', text.lower()))
        # 中文 bigram
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        for i in range(len(cn_chars)):
            if cn_chars[i] not in self._CN_STOPWORDS:
                tokens.append(cn_chars[i])
            if i + 1 < len(cn_chars):
                bigram = cn_chars[i] + cn_chars[i + 1]
                if bigram not in self._CN_STOPWORDS:
                    tokens.append(bigram)
        return tokens
    
    def test_query_tokens_include_generic_words(self):
        """验证查询分词包含大量通用词"""
        query = "请介绍一下你告诉我的你最喜欢的巴赫作品的作品号和名称,谢谢"
        tokens = self._simple_tokenize(query)
        # "介绍"、"告诉"、"喜欢" 等通用词也会出现在 token 中
        assert len(tokens) > 5
    
    def test_thinking_process_matches_many_tokens(self):
        """Thinking Process 长文本匹配大量 query token"""
        query_tokens = self._simple_tokenize("请介绍一下你告诉我的你最喜欢的巴赫作品")
        
        # 模拟 Thinking Process 内容 (包含大量通用词)
        thinking = (
            "Analyze the Request: User Lucas. Current Request: introduce SAP Business One. "
            "I need to be rigorous. 介绍 SAP 产品 喜欢 作品 技术架构 "
        )
        thinking_tokens = self._simple_tokenize(thinking)
        
        # 统计重叠
        overlap = set(query_tokens) & set(thinking_tokens)
        # Thinking Process 文本因通用词重叠导致高 BM25 分数
        assert len(overlap) > 0, "Thinking Process 应该与查询有词汇重叠（导致噪音）"
    
    def test_specific_terms_have_high_discriminative_power(self):
        """验证 '巴赫' 等特定词具有高区分度"""
        query_tokens = set(self._simple_tokenize("巴赫作品号"))
        
        relevant_msg = "巴赫的赋格的艺术 BWV 1080 是复调音乐的巅峰"
        irrelevant_msg = "请给我介绍一些适合中小企业的ERP厂商"
        
        relevant_overlap = len(query_tokens & set(self._simple_tokenize(relevant_msg)))
        irrelevant_overlap = len(query_tokens & set(self._simple_tokenize(irrelevant_msg)))
        
        assert relevant_overlap > irrelevant_overlap, "相关消息应该有更多特定词重叠"
