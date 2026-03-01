#!/usr/bin/env python3
"""
端到端诊断: 追踪历史消息从DB写入到metadata输出的完整链路

不需要GPU, 使用SQLite + mock model 测试:
1. 写入消息到DB
2. _ConversationRepoWrapper 读取
3. MultiSignalRecall 召回
4. SuffixBuilder 组装
5. metadata 中 _hist_tokens_display 计算

Author: Diagnostic Script
"""
import os
import sys
import tempfile
import json

# 确保项目根在 path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository,
    ConversationRepository,
)


def step1_write_messages():
    """步骤1: 写入测试消息到数据库"""
    print("=" * 60)
    print("STEP 1: 写入消息到数据库")
    print("=" * 60)
    
    # 创建临时数据库
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False, prefix='diag_')
    db_path = tmp.name
    tmp.close()
    
    print(f"  数据库: {db_path}")
    
    # DatabaseManager 是 Singleton, 需要重置
    DatabaseManager.reset_instance()
    db_manager = DatabaseManager(db_path=db_path)
    # Tables are created automatically in _init_database
    
    session_id = "test-session-001"
    user_id = "test-user-001"
    
    # 写入会话和消息
    with db_manager.session_scope() as db:
        sess_repo = SessionRepository(db)
        sess_repo.get_or_create(session_id=session_id, user_id=user_id)
        
        conv_repo = ConversationRepository(db)
        
        # 第1轮
        conv_repo.create(session_id=session_id, role='user', content='你好，我叫Lucas，我喜欢编程')
        conv_repo.create(session_id=session_id, role='assistant', content='你好Lucas！很高兴认识你。编程是一个很棒的爱好。')
        
        # 第2轮
        conv_repo.create(session_id=session_id, role='user', content='我主要使用Python和C#进行开发')
        conv_repo.create(session_id=session_id, role='assistant', content='Python和C#都是非常优秀的编程语言。Python适合数据科学和AI，C#适合企业应用开发。')
        
        # 第3轮
        conv_repo.create(session_id=session_id, role='user', content='我们聊聊ERP系统吧')
        conv_repo.create(session_id=session_id, role='assistant', content='好的！ERP系统是企业资源规划系统，用于整合企业的各种业务流程。')
    
    # 验证写入
    with db_manager.session_scope() as db:
        conv_repo = ConversationRepository(db)
        msgs = conv_repo.get_by_session(session_id)
        print(f"  写入 {len(msgs)} 条消息")
        for m in msgs:
            print(f"    [{m.role}] {m.content[:40]}... (id={m.id})")
    
    print(f"  ✓ 步骤1完成\n")
    return db_manager, db_path, session_id, user_id


def step2_wrapper_read(db_manager, session_id):
    """步骤2: _ConversationRepoWrapper 读取"""
    print("=" * 60)
    print("STEP 2: _ConversationRepoWrapper 读取")
    print("=" * 60)
    
    # 导入 wrapper
    from dki.core.dki_system import _ConversationRepoWrapper, _DetachedMessage
    
    wrapper = _ConversationRepoWrapper(db_manager)
    
    # get_by_session
    msgs = wrapper.get_by_session(session_id=session_id)
    print(f"  get_by_session 返回 {len(msgs)} 条")
    
    for m in msgs:
        is_detached = isinstance(m, _DetachedMessage)
        role = getattr(m, 'role', '?')
        content = getattr(m, 'content', '')
        msg_id = getattr(m, 'id', '?')
        print(f"    [detached={is_detached}] [{role}] {content[:40]}... (id={msg_id})")
        
        # 验证属性可访问性
        assert role in ('user', 'assistant'), f"角色异常: {role}"
        assert len(content) > 0, f"内容为空! id={msg_id}"
    
    # get_recent
    recent = wrapper.get_recent(session_id=session_id, limit=4)
    print(f"\n  get_recent(limit=4) 返回 {len(recent)} 条")
    for m in recent:
        print(f"    [{m.role}] {m.content[:40]}...")
    
    print(f"  ✓ 步骤2完成\n")
    return msgs


def step3_multi_signal_recall(db_manager, session_id, user_id):
    """步骤3: MultiSignalRecall 召回"""
    print("=" * 60)
    print("STEP 3: MultiSignalRecall 召回")
    print("=" * 60)
    
    from dki.core.dki_system import _ConversationRepoWrapper
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig
    
    wrapper = _ConversationRepoWrapper(db_manager)
    config = RecallConfig()  # 使用默认配置
    
    recall = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )
    
    query = "你记得我的名字吗？我用什么编程语言？"
    print(f"  查询: {query}")
    
    result = recall.recall(
        query=query,
        session_id=session_id,
        user_id=user_id,
    )
    
    print(f"  召回结果:")
    print(f"    总消息数: {len(result.messages)}")
    print(f"    keyword_hits: {result.keyword_hits}")
    print(f"    bm25_hits: {result.bm25_hits}")
    print(f"    vector_hits: {result.vector_hits}")
    print(f"    recent_turns_added: {result.recent_turns_added}")
    
    for m in result.messages:
        role = getattr(m, 'role', '?')
        content = getattr(m, 'content', '')
        msg_id = getattr(m, 'id', '?')
        print(f"    [{role}] {content[:50]}... (id={msg_id})")
    
    if not result.messages:
        print("  ⚠ 警告: 召回消息为空!")
        print("  分析: jieba/bm25 可能未安装, 检查近期轮次...")
        print(f"    min_recent_turns 配置: {config.budget.min_recent_turns}")
    
    print(f"  ✓ 步骤3完成\n")
    return result


def step4_suffix_builder(db_manager, recall_result, query):
    """步骤4: SuffixBuilder 组装"""
    print("=" * 60)
    print("STEP 4: SuffixBuilder 组装")
    print("=" * 60)
    
    from dki.core.recall.recall_config import RecallConfig
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter
    
    config = RecallConfig()
    formatter = create_formatter(model_name="deepseek", formatter_type="chatml", language="cn")
    
    builder = SuffixBuilder(
        config=config,
        prompt_formatter=formatter,
        token_counter=None,  # 使用粗估
        model_adapter=None,
    )
    
    assembled = builder.build(
        query=query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=50,
    )
    
    print(f"  组装结果:")
    print(f"    text 长度: {len(assembled.text)} 字符")
    print(f"    total_tokens: {assembled.total_tokens}")
    print(f"    message_count: {assembled.message_count}")
    print(f"    summary_count: {assembled.summary_count}")
    print(f"    items 数量: {len(assembled.items)}")
    print(f"    trace_ids: {assembled.trace_ids}")
    print(f"    has_fact_call_instruction: {assembled.has_fact_call_instruction}")
    
    if assembled.items:
        print(f"\n  Items 详情:")
        for i, item in enumerate(assembled.items):
            print(f"    [{i}] type={item.type}, role={item.role}, "
                  f"tokens={item.token_count}, content={item.content[:50]}...")
    
    if assembled.text:
        print(f"\n  组装文本 (前 300 字符):")
        print(f"    {assembled.text[:300]}")
    
    print(f"  ✓ 步骤4完成\n")
    return assembled


def step5_metadata_simulation(assembled, recall_v4_suffix, preference_content):
    """步骤5: 模拟 metadata 计算"""
    print("=" * 60)
    print("STEP 5: 模拟 metadata 中 _hist_tokens_display 计算")
    print("=" * 60)
    
    # 模拟 dki_system.py 中的 metadata 计算逻辑
    _hist_tokens_display = 0
    _pref_tokens_display = 0
    _use_recall_v4 = True
    
    # 偏好 token 计算
    if preference_content:
        import re
        _cn = len(re.findall(r'[\u4e00-\u9fff]', preference_content))
        _en = len(re.findall(r'[a-zA-Z]+', preference_content))
        _pref_tokens_display = int(_cn * 1.5 + _en * 1.3) or len(preference_content)
    
    print(f"  偏好 tokens: {_pref_tokens_display}")
    
    # 关键检查: recall_v4_suffix 的值
    print(f"\n  recall_v4_suffix 值:")
    print(f"    是否为 None: {recall_v4_suffix is None}")
    print(f"    是否为空字符串: {recall_v4_suffix == ''}")
    print(f"    bool(recall_v4_suffix): {bool(recall_v4_suffix)}")
    if recall_v4_suffix:
        print(f"    长度: {len(recall_v4_suffix)}")
    
    # 模拟 dki_system.py 第 1139 行的条件
    print(f"\n  条件检查 (line 1139): if recall_v4_suffix and self._use_recall_v4:")
    print(f"    recall_v4_suffix = {bool(recall_v4_suffix)}")
    print(f"    _use_recall_v4 = {_use_recall_v4}")
    print(f"    条件结果 = {bool(recall_v4_suffix) and _use_recall_v4}")
    
    if recall_v4_suffix and _use_recall_v4:
        # 粗估 (无 tokenizer 时)
        _hist_tokens_display = len(recall_v4_suffix) // 2
        print(f"\n  _hist_tokens_display (粗估) = {_hist_tokens_display}")
    else:
        print(f"\n  ⚠ 条件不满足! _hist_tokens_display 保持为 0")
    
    # 检查 has_history 条件
    has_history = assembled.total_tokens > 0 and assembled.items
    has_preference = preference_content is not None and len(preference_content) > 0
    
    print(f"\n  has_history 条件 (line 908):")
    print(f"    assembled.total_tokens = {assembled.total_tokens}")
    print(f"    bool(assembled.items) = {bool(assembled.items)}")
    print(f"    has_history = {has_history}")
    print(f"    has_preference = {has_preference}")
    print(f"    会构建 chat prompt: {has_history or has_preference}")
    
    # 关键发现
    print(f"\n  === 关键发现 ===")
    if assembled.total_tokens == 0 and not assembled.items:
        print(f"  ⚠ assembled.total_tokens=0 且 items 为空")
        print(f"    说明 recall_result.messages 为空 (步骤3没有召回到消息)")
        print(f"    recall_v4_suffix = assembled.text = query (不是历史)")
        print(f"    但 line 1139 的条件仍为 True (因为 query 不为空)")
        print(f"    导致 _hist_tokens_display = len(query) // 2 = 误报!")
    elif assembled.total_tokens > 0:
        print(f"  ✓ assembled.total_tokens={assembled.total_tokens}, 有历史")
        print(f"    recall_v4_suffix 包含真实历史文本")
        print(f"    _hist_tokens_display 应该正确")
    
    print(f"\n  ✓ 步骤5完成\n")
    return _hist_tokens_display


def step6_check_recall_v4_suffix_bug():
    """步骤6: 验证 recall_v4_suffix 的 bug"""
    print("=" * 60)
    print("STEP 6: 验证 recall_v4_suffix 赋值逻辑")
    print("=" * 60)
    
    from dki.core.recall.recall_config import AssembledSuffix
    
    # Case 1: recalled_messages 为空
    print("  Case 1: recalled_messages 为空")
    result = AssembledSuffix()
    query = "你记得我的名字吗"
    # 模拟 SuffixBuilder.build() 当 recalled_messages 为空时
    result.text = query  # SuffixBuilder line 96
    recall_v4_suffix = result.text  # dki_system line 901
    print(f"    recall_v4_suffix = '{recall_v4_suffix}'")
    print(f"    bool(recall_v4_suffix) = {bool(recall_v4_suffix)}")
    print(f"    assembled.total_tokens = {result.total_tokens}")
    print(f"    assembled.items = {result.items}")
    print(f"    ⚠ BUG: recall_v4_suffix 是 query 文本, 不是历史!")
    print(f"    ⚠ BUG: 但 line 1139 条件为 True, _hist_tokens_display 被设为 query 的 token 数")
    
    # Case 2: recalled_messages 非空
    print(f"\n  Case 2: recalled_messages 非空")
    result2 = AssembledSuffix()
    result2.text = "[历史消息]\nUser: 你好\nAssistant: 你好\n\n你记得我的名字吗"
    result2.total_tokens = 50
    result2.items = ["placeholder"]
    recall_v4_suffix2 = result2.text
    print(f"    recall_v4_suffix 长度 = {len(recall_v4_suffix2)}")
    print(f"    assembled.total_tokens = {result2.total_tokens}")
    print(f"    ✓ 正确: recall_v4_suffix 包含历史 + query")
    
    print(f"\n  === 根因分析 ===")
    print(f"  当 recall_result.messages 为空时 (第一轮或召回失败):")
    print(f"    SuffixBuilder.build() 返回 text=query, total_tokens=0")
    print(f"    dki_system.py line 901: recall_v4_suffix = query (非空字符串)")
    print(f"    dki_system.py line 1139: if recall_v4_suffix → True (因为 query 非空)")
    print(f"    dki_system.py line 1143: _hist_tokens_display = len(query) // 2")
    print(f"    这会误报历史 token 数! 但实际上没有历史.")
    print(f"")
    print(f"  但如果 UI 显示 history_tokens = 0, 说明:")
    print(f"    1. recall_v4_suffix 确实为 None (recall_v4 块抛异常)")
    print(f"    2. 或 self._use_recall_v4 为 False")
    print(f"    3. 或 hybrid_result 覆盖了 _hist_tokens_display (line 1130-1134)")
    print(f"")
    print(f"  关键: line 1130-1134 在 line 1139 之前!")
    print(f"    if hybrid_result:")
    print(f"        _hist_tokens_display = hybrid_result.history_tokens")
    print(f"    这里 hybrid_result.history_tokens 可能为 0!")
    print(f"    然后 line 1139 才用 recall_v4 覆盖.")
    print(f"    如果 recall_v4 成功, 会覆盖为正确值.")
    print(f"    如果 recall_v4 失败 (异常), recall_v4_suffix=None, 不覆盖.")
    
    print(f"\n  ✓ 步骤6完成\n")


def main():
    print("\n" + "=" * 60)
    print("DKI 历史消息注入 端到端诊断")
    print("=" * 60 + "\n")
    
    # Step 1: 写入消息
    db_manager, db_path, session_id, user_id = step1_write_messages()
    
    try:
        # Step 2: Wrapper 读取
        step2_wrapper_read(db_manager, session_id)
        
        # Step 3: MultiSignalRecall 召回
        query = "你记得我的名字吗？我用什么编程语言？"
        recall_result = step3_multi_signal_recall(db_manager, session_id, user_id)
        
        # Step 4: SuffixBuilder 组装
        assembled = step4_suffix_builder(db_manager, recall_result, query)
        
        # Step 5: Metadata 模拟
        preference_content = "语言风格: 细腻温暖\n推理过程: 严谨\n个人爱好: 编程"
        recall_v4_suffix = assembled.text
        hist_tokens = step5_metadata_simulation(assembled, recall_v4_suffix, preference_content)
        
        # Step 6: Bug 分析
        step6_check_recall_v4_suffix_bug()
        
        # 最终总结
        print("=" * 60)
        print("诊断总结")
        print("=" * 60)
        print(f"  消息写入: ✓")
        print(f"  Wrapper读取: ✓")
        print(f"  MultiSignalRecall: {len(recall_result.messages)} 条消息")
        print(f"  SuffixBuilder: total_tokens={assembled.total_tokens}, items={len(assembled.items)}")
        print(f"  _hist_tokens_display: {hist_tokens}")
        
        if assembled.total_tokens > 0:
            print(f"\n  ✓ 历史消息注入链路正常!")
            print(f"    如果 UI 仍显示 0, 问题可能在:")
            print(f"    1. _use_recall_v4 配置")
            print(f"    2. recall_v4 块中抛异常被 catch")
            print(f"    3. hybrid_result 覆盖")
        else:
            print(f"\n  ⚠ 历史消息注入链路有问题!")
            print(f"    recall_result.messages 为空 ({len(recall_result.messages)})")
            print(f"    原因: min_recent_turns={recall_result.recent_turns_added}")
        
    finally:
        # 清理
        try:
            os.unlink(db_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
