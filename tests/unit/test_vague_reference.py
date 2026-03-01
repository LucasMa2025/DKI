"""
Unit tests for vague reference detection and clarification instruction (v6.5)

Tests:
1. detect_vague_reference: 中文/英文模糊指代检测
2. build_clarification_instruction: 澄清指令生成
3. 边界情况: 空输入、具体主题、非模糊指代

Author: AGI Demo Project
"""

import pytest

from dki.core.text_utils import (
    detect_vague_reference,
    build_clarification_instruction,
    VagueReferenceResult,
)


# ============================================================
# detect_vague_reference - 中文模糊指代
# ============================================================

class TestDetectVagueReferenceChinese:
    """中文模糊指代检测"""
    
    def test_classic_vague_reference(self):
        """经典模糊指代: 前段时间说的那件事"""
        result = detect_vague_reference("前段时间和你说的那件事你现在怎么想")
        assert result.is_vague is True
        assert result.confidence >= 0.8
        assert result.language == "cn"
        assert result.matched_pattern  # 非空
    
    def test_last_time_discussed(self):
        """上次聊的那个"""
        result = detect_vague_reference("上次聊的那个怎么样了")
        assert result.is_vague is True
        assert result.language == "cn"
    
    def test_remember_that_thing(self):
        """你还记得那件事吗"""
        result = detect_vague_reference("你还记得那件事吗")
        assert result.is_vague is True
        assert result.language == "cn"
    
    def test_that_thing_progress(self):
        """那件事有什么进展"""
        result = detect_vague_reference("那件事有什么进展")
        assert result.is_vague is True
        assert result.language == "cn"
    
    def test_previously_mentioned(self):
        """之前提到的那个"""
        result = detect_vague_reference("之前提到的那个你怎么看")
        assert result.is_vague is True
        assert result.language == "cn"
    
    def test_specific_topic_not_vague(self):
        """有具体主题的不算模糊: 关于Python的问题"""
        result = detect_vague_reference("关于Python的问题你怎么看")
        assert result.is_vague is False
        assert result.language == "cn"
    
    def test_specific_topic_with_about(self):
        """有具体主题: 有关机器学习"""
        result = detect_vague_reference("有关机器学习的那个方案怎么样了")
        assert result.is_vague is False
        assert result.language == "cn"
    
    def test_normal_question_not_vague(self):
        """普通问题不算模糊"""
        result = detect_vague_reference("今天天气怎么样")
        assert result.is_vague is False
    
    def test_direct_question_not_vague(self):
        """直接提问不算模糊"""
        result = detect_vague_reference("帮我写一个Python排序算法")
        assert result.is_vague is False


# ============================================================
# detect_vague_reference - 英文模糊指代
# ============================================================

class TestDetectVagueReferenceEnglish:
    """英文模糊指代检测"""
    
    def test_last_time_talked(self):
        """Last time we talked about that thing"""
        result = detect_vague_reference("Last time we talked about that thing, what do you think?")
        assert result.is_vague is True
        assert result.language == "en"
    
    def test_do_you_remember(self):
        """Do you remember that thing we discussed?"""
        result = detect_vague_reference("Do you remember that thing we discussed?")
        assert result.is_vague is True
        assert result.language == "en"
    
    def test_that_matter_progress(self):
        """That matter we discussed, any progress?"""
        result = detect_vague_reference("That matter we discussed, any progress?")
        assert result.is_vague is True
        assert result.language == "en"
    
    def test_specific_topic_not_vague_en(self):
        """Specific topic: about machine learning"""
        result = detect_vague_reference("about machine learning, what do you think?")
        assert result.is_vague is False
        assert result.language == "en"
    
    def test_normal_question_en(self):
        """Normal English question"""
        result = detect_vague_reference("How do I sort a list in Python?")
        assert result.is_vague is False


# ============================================================
# detect_vague_reference - 边界情况
# ============================================================

class TestDetectVagueReferenceEdgeCases:
    """边界情况"""
    
    def test_empty_string(self):
        result = detect_vague_reference("")
        assert result.is_vague is False
    
    def test_none_input(self):
        """None 输入不应崩溃"""
        # detect_vague_reference 接受 str, 但 None 应安全处理
        result = detect_vague_reference(None)
        assert result.is_vague is False
    
    def test_whitespace_only(self):
        result = detect_vague_reference("   \n\t  ")
        assert result.is_vague is False
    
    def test_bool_conversion(self):
        """VagueReferenceResult 的 bool 转换"""
        vague = VagueReferenceResult(is_vague=True, confidence=0.85)
        not_vague = VagueReferenceResult(is_vague=False)
        
        assert bool(vague) is True
        assert bool(not_vague) is False
    
    def test_short_input(self):
        """很短的输入"""
        result = detect_vague_reference("那个")
        # 太短可能不匹配完整模式
        assert isinstance(result, VagueReferenceResult)


# ============================================================
# build_clarification_instruction
# ============================================================

class TestBuildClarificationInstruction:
    """澄清指令生成"""
    
    def test_chinese_instruction(self):
        instruction = build_clarification_instruction("cn")
        assert "重要指令" in instruction
        assert "模糊" in instruction
        assert "澄清" in instruction or "线索" in instruction
        assert len(instruction) > 50
    
    def test_english_instruction(self):
        instruction = build_clarification_instruction("en")
        assert "IMPORTANT" in instruction
        assert "vague" in instruction.lower()
        assert len(instruction) > 50
    
    def test_default_is_chinese(self):
        """默认语言是中文"""
        instruction = build_clarification_instruction()
        assert "重要指令" in instruction
    
    def test_unknown_language_falls_back_to_english(self):
        """未知语言回退到英文"""
        instruction = build_clarification_instruction("fr")
        assert "IMPORTANT" in instruction
    
    def test_instruction_not_empty(self):
        """指令不为空"""
        for lang in ["cn", "en"]:
            instruction = build_clarification_instruction(lang)
            assert instruction.strip()


# ============================================================
# 集成: detect + build 联合使用
# ============================================================

class TestVagueReferenceIntegration:
    """检测 + 澄清指令联合使用"""
    
    def test_detect_then_build_cn(self):
        """中文: 检测到模糊指代后生成澄清指令"""
        result = detect_vague_reference("前段时间和你说的那件事你现在怎么想")
        assert result.is_vague
        
        instruction = build_clarification_instruction(result.language)
        assert "重要指令" in instruction
    
    def test_detect_then_build_en(self):
        """英文: 检测到模糊指代后生成澄清指令"""
        result = detect_vague_reference("Do you remember that thing we discussed?")
        assert result.is_vague
        
        instruction = build_clarification_instruction(result.language)
        assert "IMPORTANT" in instruction
    
    def test_no_vague_no_instruction(self):
        """无模糊指代时不应生成澄清指令"""
        result = detect_vague_reference("帮我写一个Python排序算法")
        assert not result.is_vague
        # 即使调用 build_clarification_instruction 也不会出错
        # 但在实际代码中, 只有 is_vague=True 时才会调用
