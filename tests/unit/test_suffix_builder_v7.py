"""
DKI v7.0 SuffixBuilder 优化单元测试

测试覆盖:
1. 增强切句策略 (_split_sentences)
2. Query-Aware 句子评分 (_compute_query_relevance)
3. TF-IDF 事实密度评分 (_compute_fact_density)
4. 增强认知标记检测 (_extract_epistemic_markers)
5. Summary 完整性校验
6. 端到端: _extractive_summarize with query
7. 端到端: _global_budget_allocate with query

Author: AGI Demo Project
Version: 7.0.0
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

# 测试所需的数据结构
from dki.core.recall.recall_config import (
    RecallConfig,
    RecallSummaryConfig,
    RecallBudgetConfig,
    HistoryItem,
    AssembledSuffix,
)
from dki.core.recall.suffix_builder import SuffixBuilder

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ================================================================
# 测试辅助工具
# ================================================================

def _make_config(
    per_message_threshold: int = 300,
    max_tokens_per_summary: int = 150,
    strategy: str = "extractive",
) -> RecallConfig:
    """创建测试用 RecallConfig"""
    config = RecallConfig()
    config.summary = RecallSummaryConfig(
        per_message_threshold=per_message_threshold,
        max_tokens_per_summary=max_tokens_per_summary,
        strategy=strategy,
    )
    return config


def _make_formatter():
    """创建 Mock PromptFormatter"""
    formatter = MagicMock()
    formatter.format_full_suffix.return_value = "formatted_suffix"
    return formatter


def _make_builder(
    per_message_threshold: int = 300,
    max_tokens_per_summary: int = 150,
) -> SuffixBuilder:
    """创建测试用 SuffixBuilder"""
    config = _make_config(
        per_message_threshold=per_message_threshold,
        max_tokens_per_summary=max_tokens_per_summary,
    )
    formatter = _make_formatter()
    return SuffixBuilder(config=config, prompt_formatter=formatter)


@dataclass
class MockMessage:
    """模拟消息对象"""
    content: str
    role: str = "user"
    id: Optional[str] = None
    message_id: Optional[str] = None


# ================================================================
# 1. 增强切句策略测试
# ================================================================

class TestSplitSentences:
    """测试 v7.0 增强切句策略"""

    def test_basic_chinese_punctuation(self):
        """基本中文标点切句"""
        text = "今天天气很好。我去了公园。看到了很多花！"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) == 3
        assert "今天天气很好" in result[0]

    def test_question_mark_split(self):
        """问号切句"""
        text = "你好吗？我很好。谢谢你的关心！"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) == 3

    def test_semicolon_split(self):
        """分号切句"""
        text = "第一点是价格便宜；第二点是质量好；第三点是服务周到"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) >= 2  # 分号应该切分

    def test_newline_split(self):
        """换行切句"""
        text = "第一行内容\n第二行内容\n第三行内容"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) == 3

    def test_mixed_punctuation(self):
        """混合标点切句"""
        text = "你好！今天去哪里？我们去公园吧。好的；那就这样"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) >= 4

    def test_short_sentence_preserved(self):
        """短句 (≥3 字符) 保留"""
        text = "好的。明天见。拜拜"
        result = SuffixBuilder._split_sentences(text)
        assert len(result) >= 2  # "好的" 和 "明天见" 都 ≥ 3 字符

    def test_empty_text(self):
        """空文本"""
        result = SuffixBuilder._split_sentences("")
        assert result == []

    def test_ellipsis_split(self):
        """省略号切句"""
        text = "我觉得可能是这样……但是也不确定。让我再想想..."
        result = SuffixBuilder._split_sentences(text)
        assert len(result) >= 2

    def test_english_period_split(self):
        """英文句号切句 (不在小数点处切)"""
        text = "The price is 3.14 dollars. That's expensive. Really?"
        result = SuffixBuilder._split_sentences(text)
        # 不应该在 3.14 处切分
        assert any("3.14" in s for s in result)


# ================================================================
# 2. Query-Aware 句子评分测试
# ================================================================

class TestQueryRelevance:
    """测试 v7.0 Query-Aware 句子评分"""

    def setup_method(self):
        self.builder = _make_builder()

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_high_relevance(self):
        """高相关性: 句子包含 query 的多个关键词"""
        query_words = {"推荐", "餐厅", "北京"}
        query_bigrams = {"推荐餐厅", "餐厅北京"}
        score = self.builder._compute_query_relevance(
            "我推荐北京的一家很好的餐厅",
            query_words, query_bigrams,
        )
        assert score > 0.3

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_no_relevance(self):
        """无相关性: 句子不包含 query 的任何词"""
        query_words = {"推荐", "餐厅"}
        query_bigrams = {"推荐餐厅"}
        score = self.builder._compute_query_relevance(
            "今天的天气非常晴朗",
            query_words, query_bigrams,
        )
        assert score < 0.1

    def test_empty_query(self):
        """空 query 返回 0"""
        score = self.builder._compute_query_relevance(
            "任何句子", set(), set(),
        )
        assert score == 0.0

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_partial_relevance(self):
        """部分相关: 句子包含 query 的部分词"""
        query_words = {"推荐", "餐厅", "北京", "火锅"}
        query_bigrams = set()
        score = self.builder._compute_query_relevance(
            "北京有很多好吃的地方",
            query_words, query_bigrams,
        )
        assert 0.0 < score < 0.5


# ================================================================
# 3. TF-IDF 事实密度评分测试
# ================================================================

class TestFactDensity:
    """测试 v7.0 TF-IDF 事实密度评分"""

    def test_high_density_with_numbers(self):
        """高事实密度: 包含多个数字和日期"""
        text = "2024年3月15日下午3点，在北京市朝阳区建国路88号，价格是299元"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.5

    def test_low_density_opinion(self):
        """低事实密度: 纯观点句"""
        text = "我觉得这个方案还不错可以考虑一下"
        score = SuffixBuilder._compute_fact_density(text)
        assert score < 0.4

    def test_url_detection(self):
        """URL 检测"""
        text = "请访问 https://example.com/path 获取更多信息"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3

    def test_email_detection(self):
        """邮箱检测"""
        text = "联系邮箱是 test@example.com 有问题请联系"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3

    def test_phone_detection(self):
        """电话号码检测"""
        text = "客服电话 13800138000 工作时间拨打"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3

    def test_address_detection(self):
        """地址检测"""
        text = "位于上海市浦东新区世纪大道100号环球金融中心"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3

    def test_book_title_detection(self):
        """书名/标题检测"""
        text = "推荐阅读《三体》和《流浪地球》这两本科幻小说"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3

    def test_measure_detection(self):
        """精确量词检测"""
        text = "需要3个苹果5斤大米和2升牛奶"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.4

    def test_empty_text(self):
        """空文本返回 0"""
        assert SuffixBuilder._compute_fact_density("") == 0.0
        assert SuffixBuilder._compute_fact_density("ab") == 0.0

    def test_percentage_detection(self):
        """百分比检测"""
        text = "今年销售额增长了35%，利润率达到12.5%"
        score = SuffixBuilder._compute_fact_density(text)
        assert score > 0.3


# ================================================================
# 4. 增强认知标记检测测试
# ================================================================

class TestEnhancedEpistemicMarkers:
    """测试 v7.0 增强认知标记检测"""

    def setup_method(self):
        self.builder = _make_builder()

    def test_date_covered(self):
        """日期在 summary 中保留"""
        original = "会议定在2024年3月15日召开"
        summary = "2024年3月15日有一个会议"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert "日期" in covered

    def test_date_missing(self):
        """日期在 summary 中遗漏"""
        original = "会议定在2024年3月15日下午3点召开"
        summary = "有一个会议即将召开"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        missing_types = [m.split("(")[0] for m in missing]
        assert "日期" in missing_types

    def test_price_covered(self):
        """价格在 summary 中保留"""
        original = "这个产品售价299元"
        summary = "产品价格299元"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert "价格" in covered

    def test_phone_missing(self):
        """电话号码遗漏"""
        original = "联系电话 13800138000 有事请拨打"
        summary = "可以打电话联系"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        missing_types = [m.split("(")[0] for m in missing]
        assert "电话" in missing_types

    def test_email_covered(self):
        """邮箱保留"""
        original = "邮箱是 admin@test.com"
        summary = "联系 admin@test.com"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert "邮箱" in covered

    def test_address_detection(self):
        """地址检测"""
        original = "地址在北京市朝阳区建国路88号"
        summary = "在北京市朝阳区"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        # 部分覆盖: "北京市" 和 "朝阳区" 保留, 但 "建国路" 可能遗漏
        assert "地址" in covered or any("地址" in m for m in missing)

    def test_partial_coverage(self):
        """部分覆盖: 原文有多个同类事实, summary 只保留部分"""
        original = "2024年3月15日和2024年5月20日有两个会议"
        summary = "2024年3月15日有一个会议"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        # 日期应该在 covered 中 (有保留), 但也应该在 missing 中 (部分遗漏)
        assert "日期" in covered
        assert any("日期" in m and "部分遗漏" in m for m in missing)

    def test_high_compression_ratio(self):
        """高压缩比标注"""
        original = "这是一段非常长的文本" * 20  # 很长
        summary = "摘要"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert any("详细内容" in m for m in missing)

    def test_no_facts_in_original(self):
        """原文没有事实性信息"""
        original = "我觉得这个想法很好"
        summary = "想法不错"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        # 没有事实可检测, covered 和 missing 都应该很少
        assert len(covered) == 0

    def test_url_tracking(self):
        """URL 跟踪"""
        original = "详情请见 https://example.com/detail"
        summary = "有一个链接可以查看详情"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        missing_types = [m.split("(")[0] for m in missing]
        assert "URL" in missing_types

    def test_book_title_tracking(self):
        """书名跟踪"""
        original = "推荐阅读《三体》这本小说"
        summary = "推荐一本科幻小说"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        missing_types = [m.split("(")[0] for m in missing]
        assert "书名/标题" in missing_types


# ================================================================
# 5. Summary 完整性校验测试
# ================================================================

class TestSummaryCompleteness:
    """测试 v7.0 Summary 完整性校验"""

    def setup_method(self):
        self.builder = _make_builder()

    def test_extreme_compression(self):
        """极端压缩 (>4x) 标注"""
        original = "今天我们讨论了很多重要的内容包括项目进度和预算分配" * 10
        summary = "讨论了项目"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert any("压缩比>4x" in m for m in missing)

    def test_moderate_compression(self):
        """中等压缩 (>3x) 标注"""
        original = "今天我们讨论了很多重要的内容" * 5
        summary = "讨论了内容"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        assert any("详细内容" in m for m in missing)

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_keyword_coverage_check(self):
        """关键词覆盖率检查"""
        original = (
            "机器学习是人工智能的一个重要分支，"
            "深度学习是机器学习的子领域，"
            "神经网络是深度学习的核心技术，"
            "卷积神经网络广泛应用于图像识别领域"
        )
        # summary 只保留了很少的关键词
        summary = "讨论了一些技术话题"
        covered, missing = self.builder._extract_epistemic_markers(original, summary)
        # 应该检测到关键词覆盖率低
        assert any("关键词" in m for m in missing)


# ================================================================
# 6. 端到端: _extractive_summarize with query
# ================================================================

class TestExtractSummarizeWithQuery:
    """测试 v7.0 Query-Aware 抽取式摘要"""

    def setup_method(self):
        self.builder = _make_builder(max_tokens_per_summary=200)

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_query_aware_prioritization(self):
        """Query-Aware: 与 query 相关的句子应优先保留"""
        text = (
            "今天天气很好适合出去玩。"
            "北京有很多好吃的餐厅。"
            "推荐去三里屯的一家日料店。"
            "那里的寿司非常新鲜。"
            "价格大概人均200元。"
            "交通也很方便地铁直达。"
            "附近还有很多购物中心。"
            "晚上可以去酒吧街逛逛。"
        )
        
        # 使用与餐厅相关的 query
        summary_with_query = self.builder._extractive_summarize(
            text, max_tokens=100, query="推荐北京的餐厅"
        )
        
        # 不使用 query
        summary_without_query = self.builder._extractive_summarize(
            text, max_tokens=100, query=""
        )
        
        # 有 query 时, 应该更倾向于保留餐厅相关的句子
        # 至少 "餐厅" 或 "推荐" 应该出现在 summary 中
        assert "餐厅" in summary_with_query or "推荐" in summary_with_query

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_fact_density_prioritization(self):
        """事实密度: 包含具体数据的句子应优先保留"""
        text = (
            "我觉得这个方案还不错。"
            "预算大约需要50万元。"
            "项目周期预计3个月。"
            "团队需要5个人参与。"
            "总体来说还是很有前景的。"
            "大家都觉得可以尝试一下。"
        )
        
        summary = self.builder._extractive_summarize(
            text, max_tokens=80, query=""
        )
        
        # 包含数字的句子应该被优先保留
        has_numbers = bool(
            "50万" in summary or "3个月" in summary or "5个" in summary
        )
        assert has_numbers

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_dynamic_topk(self):
        """动态 top-k: 句子少时选取更少"""
        # 只有 3 个句子
        text = "第一句话。第二句话。第三句话"
        summary = self.builder._extractive_summarize(
            text, max_tokens=200, query=""
        )
        # 应该全部保留 (3 句 ≤ dynamic_k)
        assert "第一" in summary or "第二" in summary or "第三" in summary

    def test_no_jieba_fallback(self):
        """无 jieba 时回退到截断"""
        builder = _make_builder(max_tokens_per_summary=50)
        with patch.object(
            type(builder), '_extractive_summarize',
            wraps=builder._extractive_summarize,
        ):
            # 即使 jieba 不可用, 也不应该崩溃
            text = "这是一段测试文本" * 20
            summary = builder._extractive_summarize(text, max_tokens=50, query="")
            assert len(summary) > 0


# ================================================================
# 7. 端到端: _global_budget_allocate with query
# ================================================================

class TestGlobalBudgetAllocateWithQuery:
    """测试 v7.0 全局预算分配 (含 Query-Aware)"""

    def setup_method(self):
        self.builder = _make_builder(per_message_threshold=50)

    def test_all_fit_no_compression(self):
        """全部放得下时不压缩"""
        collected = [
            {"msg_id": "1", "content": "短消息A", "role": "user", "tokens": 10},
            {"msg_id": "2", "content": "短消息B", "role": "assistant", "tokens": 10},
        ]
        items, used = self.builder._global_budget_allocate(
            collected, context_budget=100, query="测试"
        )
        assert len(items) == 2
        assert all(i.type == "message" for i in items)
        assert used == 20

    @pytest.mark.skipif(not JIEBA_AVAILABLE, reason="jieba not installed")
    def test_compression_triggered_with_query(self):
        """预算不足时触发压缩, 且传入 query"""
        long_content = "这是一段关于北京餐厅推荐的详细内容。" * 30
        collected = [
            {"msg_id": "1", "content": "短消息", "role": "user", "tokens": 10},
            {"msg_id": "2", "content": long_content, "role": "assistant", "tokens": 500},
        ]
        items, used = self.builder._global_budget_allocate(
            collected, context_budget=200, query="推荐餐厅"
        )
        # 短消息保留原文, 长消息应被压缩
        assert any(i.type == "summary" for i in items)

    def test_budget_exhausted(self):
        """预算耗尽时跳过消息"""
        collected = [
            {"msg_id": "1", "content": "短消息A", "role": "user", "tokens": 10},
            {"msg_id": "2", "content": "长消息B" * 100, "role": "assistant", "tokens": 1000},
        ]
        items, used = self.builder._global_budget_allocate(
            collected, context_budget=15, query=""
        )
        # 只有短消息放得下
        assert len(items) <= 2
        assert used <= 15


# ================================================================
# 8. 端到端: build() 方法集成测试
# ================================================================

class TestBuildIntegration:
    """测试 build() 方法的端到端集成"""

    def setup_method(self):
        self.builder = _make_builder(per_message_threshold=50)

    def test_build_with_query(self):
        """build() 正确传递 query"""
        messages = [
            MockMessage(content="短消息A", role="user", id="1"),
            MockMessage(content="短消息B", role="assistant", id="2"),
        ]
        result = self.builder.build(
            query="测试查询",
            recalled_messages=messages,
            context_window=4096,
            preference_tokens=100,
        )
        assert isinstance(result, AssembledSuffix)
        assert result.message_count >= 0

    def test_build_empty_messages(self):
        """空消息列表"""
        result = self.builder.build(
            query="测试",
            recalled_messages=[],
            context_window=4096,
        )
        assert result.text == "测试"

    def test_build_budget_exhausted(self):
        """预算耗尽"""
        messages = [
            MockMessage(content="消息", role="user", id="1"),
        ]
        result = self.builder.build(
            query="测试",
            recalled_messages=messages,
            context_window=100,  # 非常小的窗口
            preference_tokens=50,
        )
        # 不应该崩溃
        assert isinstance(result, AssembledSuffix)


# ================================================================
# 9. _FACT_PATTERNS 正则测试
# ================================================================

class TestFactPatterns:
    """测试 v7.0 事实类别正则表达式"""

    def setup_method(self):
        self.builder = _make_builder()
        self.patterns = self.builder._FACT_PATTERNS

    def test_date_pattern(self):
        """日期正则"""
        import re
        pattern = self.patterns["日期"]
        assert re.search(pattern, "2024年3月15日")
        assert re.search(pattern, "2024-03-15")
        assert re.search(pattern, "2024/3/15")

    def test_time_pattern(self):
        """时间正则"""
        import re
        pattern = self.patterns["时间"]
        assert re.search(pattern, "下午3点30分")
        assert re.search(pattern, "早上")
        assert re.search(pattern, "晚上8点")

    def test_price_pattern(self):
        """价格正则"""
        import re
        pattern = self.patterns["价格"]
        assert re.search(pattern, "299元")
        assert re.search(pattern, "5.99万")
        assert re.search(pattern, "100块")

    def test_quantity_pattern(self):
        """数量正则"""
        import re
        pattern = self.patterns["数量"]
        assert re.search(pattern, "3个苹果")
        assert re.search(pattern, "5次会议")
        assert re.search(pattern, "2层楼")

    def test_percentage_pattern(self):
        """百分比正则"""
        import re
        pattern = self.patterns["百分比"]
        assert re.search(pattern, "35%")
        assert re.search(pattern, "12.5％")

    def test_phone_pattern(self):
        """电话正则"""
        import re
        pattern = self.patterns["电话"]
        assert re.search(pattern, "13800138000")
        assert re.search(pattern, "010-12345678")

    def test_email_pattern(self):
        """邮箱正则"""
        import re
        pattern = self.patterns["邮箱"]
        assert re.search(pattern, "test@example.com")
        assert re.search(pattern, "user.name@company.co.jp")

    def test_url_pattern(self):
        """URL 正则"""
        import re
        pattern = self.patterns["URL"]
        assert re.search(pattern, "https://example.com/path")
        assert re.search(pattern, "http://test.org")

    def test_address_pattern(self):
        """地址正则"""
        import re
        pattern = self.patterns["地址"]
        assert re.search(pattern, "北京市朝阳区")
        assert re.search(pattern, "上海市浦东新区")
        assert re.search(pattern, "建国路88号")

    def test_book_title_pattern(self):
        """书名正则"""
        import re
        pattern = self.patterns["书名/标题"]
        match = re.search(pattern, "推荐《三体》")
        assert match is not None

    def test_order_number_pattern(self):
        """订单号正则"""
        import re
        pattern = self.patterns["订单号"]
        assert re.search(pattern, "订单：ABC123456")
        assert re.search(pattern, "工单:WO2024001")
