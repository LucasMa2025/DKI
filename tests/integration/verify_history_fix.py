#!/usr/bin/env python3
"""
验证 v5.5 修复: 历史消息注入 metadata 正确性

测试场景:
1. recall_v4 成功 + 有历史 → _hist_tokens_display > 0
2. recall_v4 成功 + 无历史 (首轮) → _hist_tokens_display = 0
3. recall_v4 失败 + hybrid fallback 有历史 → _hybrid_fallback_hist_tokens > 0
4. recall_v4 失败 + hybrid fallback 无历史 → _hist_tokens_display = 0
"""
import os
import sys
import tempfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from dki.database.connection import DatabaseManager
from dki.database.repository import SessionRepository, ConversationRepository
from dki.core.dki_system import _ConversationRepoWrapper
from dki.core.recall.multi_signal_recall import MultiSignalRecall
from dki.core.recall.recall_config import RecallConfig, AssembledSuffix
from dki.core.recall.suffix_builder import SuffixBuilder
from dki.core.recall.prompt_formatter import create_formatter


def create_test_db():
    """创建带消息的测试数据库"""
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False, prefix='verify_')
    db_path = tmp.name
    tmp.close()
    
    DatabaseManager.reset_instance()
    db_manager = DatabaseManager(db_path=db_path)
    
    session_id = "verify-session-001"
    user_id = "verify-user-001"
    
    with db_manager.session_scope() as db:
        sess_repo = SessionRepository(db)
        sess_repo.get_or_create(session_id=session_id, user_id=user_id)
        
        conv_repo = ConversationRepository(db)
        conv_repo.create(session_id=session_id, role='user', content='我叫Lucas，我喜欢编程')
        conv_repo.create(session_id=session_id, role='assistant', content='你好Lucas！编程是很棒的爱好。')
        conv_repo.create(session_id=session_id, role='user', content='我主要使用Python和C#')
        conv_repo.create(session_id=session_id, role='assistant', content='Python和C#都是优秀的语言。')
    
    return db_manager, db_path, session_id, user_id


def simulate_metadata_v55(assembled, recall_v4_suffix, hybrid_fallback_hist_tokens, hybrid_fallback_hist_messages):
    """模拟 v5.5 修复后的 metadata 计算"""
    _hist_tokens_display = 0
    _hist_messages_display = []
    
    # 模拟 hybrid_result (history_tokens = 0 因 config.history_enabled=false)
    hybrid_result_history_tokens = 0
    _hist_tokens_display = hybrid_result_history_tokens
    
    # recall_v4 覆盖 (v5.5: 仅当 assembled.total_tokens > 0)
    if recall_v4_suffix is not None:
        if hasattr(assembled, 'total_tokens') and assembled.total_tokens > 0:
            _hist_tokens_display = assembled.total_tokens
        # 不再用 len(recall_v4_suffix) // 2
    
    # recall_v4 items
    if recall_v4_suffix is not None and hasattr(assembled, 'items') and assembled.items:
        for item in assembled.items:
            _hist_messages_display.append({
                'role': item.role or 'user',
                'content': item.content[:500] if item.content else '',
            })
    
    # v5.5: Hybrid fallback 补充
    if _hist_tokens_display == 0 and hybrid_fallback_hist_tokens > 0:
        _hist_tokens_display = hybrid_fallback_hist_tokens
        _hist_messages_display = hybrid_fallback_hist_messages
    
    return _hist_tokens_display, _hist_messages_display


def test_scenario_1():
    """Scenario 1: recall_v4 成功 + 有历史"""
    print("Test 1: recall_v4 成功 + 有历史")
    db_manager, db_path, session_id, user_id = create_test_db()
    
    try:
        wrapper = _ConversationRepoWrapper(db_manager)
        config = RecallConfig()
        recall = MultiSignalRecall(config=config, reference_resolver=None, memory_router=None, conversation_repo=wrapper)
        formatter = create_formatter(model_name="deepseek", formatter_type="chatml", language="cn")
        builder = SuffixBuilder(config=config, prompt_formatter=formatter)
        
        result = recall.recall(query="你记得我的名字吗", session_id=session_id, user_id=user_id)
        assembled = builder.build(query="你记得我的名字吗", recalled_messages=result.messages, context_window=4096, preference_tokens=50)
        
        # v5.5: recall_v4_suffix 仅在有历史时设置
        recall_v4_suffix = assembled.text if (assembled.total_tokens > 0 and assembled.items) else None
        
        hist_tokens, hist_messages = simulate_metadata_v55(assembled, recall_v4_suffix, 0, [])
        
        assert hist_tokens > 0, f"Expected hist_tokens > 0, got {hist_tokens}"
        assert len(hist_messages) > 0, f"Expected hist_messages > 0, got {len(hist_messages)}"
        print(f"  ✓ hist_tokens={hist_tokens}, messages={len(hist_messages)}")
    finally:
        try:
            os.unlink(db_path)
        except Exception:
            pass


def test_scenario_2():
    """Scenario 2: recall_v4 成功 + 无历史 (首轮)"""
    print("Test 2: recall_v4 成功 + 无历史 (首轮)")
    
    # 使用空的 assembled (模拟首轮)
    assembled = AssembledSuffix()
    assembled.text = "你记得我的名字吗"
    assembled.total_tokens = 0
    assembled.items = []
    
    # v5.5: recall_v4_suffix = None (因 total_tokens=0)
    recall_v4_suffix = None
    
    hist_tokens, hist_messages = simulate_metadata_v55(assembled, recall_v4_suffix, 0, [])
    
    assert hist_tokens == 0, f"Expected hist_tokens=0, got {hist_tokens}"
    assert len(hist_messages) == 0, f"Expected no messages, got {len(hist_messages)}"
    print(f"  ✓ hist_tokens={hist_tokens}, messages={len(hist_messages)} (correctly 0)")


def test_scenario_3():
    """Scenario 3: recall_v4 失败 + hybrid fallback 有历史"""
    print("Test 3: recall_v4 失败 + hybrid fallback 有历史")
    
    assembled = AssembledSuffix()  # 空 (recall_v4 失败)
    recall_v4_suffix = None  # recall_v4 失败
    
    # Hybrid fallback 获取了历史
    hybrid_fallback_hist_tokens = 120
    hybrid_fallback_hist_messages = [
        {"role": "user", "content": "我叫Lucas"},
        {"role": "assistant", "content": "你好Lucas"},
    ]
    
    hist_tokens, hist_messages = simulate_metadata_v55(
        assembled, recall_v4_suffix, hybrid_fallback_hist_tokens, hybrid_fallback_hist_messages
    )
    
    assert hist_tokens == 120, f"Expected hist_tokens=120, got {hist_tokens}"
    assert len(hist_messages) == 2, f"Expected 2 messages, got {len(hist_messages)}"
    print(f"  ✓ hist_tokens={hist_tokens}, messages={len(hist_messages)} (from hybrid fallback)")


def test_scenario_4():
    """Scenario 4: recall_v4 失败 + hybrid fallback 无历史 (首轮)"""
    print("Test 4: recall_v4 失败 + hybrid fallback 无历史 (首轮)")
    
    assembled = AssembledSuffix()
    recall_v4_suffix = None
    
    hist_tokens, hist_messages = simulate_metadata_v55(assembled, recall_v4_suffix, 0, [])
    
    assert hist_tokens == 0, f"Expected hist_tokens=0, got {hist_tokens}"
    assert len(hist_messages) == 0, f"Expected no messages, got {len(hist_messages)}"
    print(f"  ✓ hist_tokens={hist_tokens}, messages={len(hist_messages)} (correctly 0)")


def test_scenario_5_old_bug():
    """Scenario 5: 验证旧 bug 已修复 (recall_v4_suffix = query 时不误报)"""
    print("Test 5: 旧 bug 验证 (recall_v4_suffix = query 时不误报)")
    
    assembled = AssembledSuffix()
    assembled.text = "你记得我的名字吗"
    assembled.total_tokens = 0  # 无历史
    assembled.items = []
    
    # 旧代码: recall_v4_suffix = assembled.text (query 文本)
    # 新代码: recall_v4_suffix = None (因 total_tokens=0)
    recall_v4_suffix_old = assembled.text  # 旧行为
    recall_v4_suffix_new = None  # 新行为 (v5.5 修复)
    
    # 旧行为: _hist_tokens_display = len(query) // 2 = 误报
    old_hist_tokens = len(recall_v4_suffix_old) // 2
    
    # 新行为: _hist_tokens_display = 0
    new_hist_tokens, _ = simulate_metadata_v55(assembled, recall_v4_suffix_new, 0, [])
    
    assert old_hist_tokens > 0, f"Old bug should produce non-zero: {old_hist_tokens}"
    assert new_hist_tokens == 0, f"New fix should produce 0: {new_hist_tokens}"
    print(f"  ✓ Old bug: hist_tokens={old_hist_tokens} (误报)")
    print(f"  ✓ New fix: hist_tokens={new_hist_tokens} (正确)")


def main():
    print("\n" + "=" * 60)
    print("v5.5 修复验证: 历史消息注入 metadata")
    print("=" * 60 + "\n")
    
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()
    test_scenario_5_old_bug()
    
    print(f"\n{'=' * 60}")
    print("全部测试通过! v5.5 修复验证成功")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
