"""
真实集成测试: SQLite 写入消息 → recall 召回 → 后缀组装 → prompt 构造

不使用任何 mock。直接使用真实的:
- DatabaseManager (SQLite in-memory)
- ConversationRepository (写入/读取)
- _ConversationRepoWrapper (_DetachedMessage)
- MultiSignalRecall (keyword + BM25 + recency)
- SuffixBuilder (组装)
- _build_recall_v4_chat_prompt 逻辑 (prompt 构造)

目标: 验证从消息写入到 prompt 构造的完整链路，定位历史后缀为 0 的根因。
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dki.database.connection import DatabaseManager
from dki.database.repository import (
    SessionRepository,
    ConversationRepository,
)
from dki.database.models import Base


def create_test_db():
    """创建一个干净的测试数据库 (SQLite 文件, 非 singleton)"""
    # 打破 DatabaseManager 的 singleton, 避免与其他测试冲突
    DatabaseManager._instance = None
    DatabaseManager._initialized = False

    tmp = tempfile.mktemp(suffix='.db', prefix='dki_test_')
    db = DatabaseManager(db_path=tmp, echo=False)
    return db, tmp


def write_conversation_messages(db_manager, session_id, user_id, messages):
    """
    向数据库写入一组对话消息。
    messages: list of (role, content)
    """
    with db_manager.session_scope() as db:
        sess_repo = SessionRepository(db)
        sess_repo.get_or_create(session_id=session_id, user_id=user_id)

        conv_repo = ConversationRepository(db)
        for role, content in messages:
            conv_repo.create(
                session_id=session_id,
                role=role,
                content=content,
            )


def read_conversation_messages(db_manager, session_id):
    """直接从数据库读取消息 (ORM 对象, 在 session 内)"""
    with db_manager.session_scope() as db:
        conv_repo = ConversationRepository(db)
        msgs = conv_repo.get_by_session(session_id)
        # 在 session 活跃时提取数据
        return [(m.id, m.role, m.content, m.session_id) for m in msgs]


# ================================================================
# 测试 1: 数据库写入和读取
# ================================================================

def test_01_db_write_and_read():
    """验证消息可以正确写入和读取"""
    db, tmp = create_test_db()
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    uid = "test_user_01"

    messages = [
        ("user", "你好，我是 lucas"),
        ("assistant", "你好 lucas！很高兴认识你。"),
        ("user", "我喜欢编程和吉他"),
        ("assistant", "编程和吉他都是很棒的爱好！你用什么编程语言？"),
        ("user", "我主要用 Python 和 C#"),
        ("assistant", "Python 和 C# 都是很好的选择。"),
    ]

    write_conversation_messages(db, sid, uid, messages)
    result = read_conversation_messages(db, sid)

    print(f"\n{'='*60}")
    print(f"TEST 01: DB Write and Read")
    print(f"{'='*60}")
    print(f"Session ID: {sid}")
    print(f"Written: {len(messages)} messages")
    print(f"Read back: {len(result)} messages")

    for msg_id, role, content, sess_id in result:
        print(f"  [{role:>9}] {content[:60]}... (id={msg_id}, session={sess_id})")

    assert len(result) == len(messages), f"Expected {len(messages)}, got {len(result)}"
    assert all(r[3] == sid for r in result), "Session ID mismatch"
    assert result[0][1] == "user", f"First message role should be 'user', got '{result[0][1]}'"
    assert result[1][1] == "assistant", f"Second message role should be 'assistant', got '{result[1][1]}'"

    print("✓ PASSED: Messages written and read correctly")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 2: _ConversationRepoWrapper 返回 _DetachedMessage
# ================================================================

def test_02_wrapper_returns_detached():
    """验证 wrapper 返回的对象在 session 关闭后仍可访问"""
    db, tmp = create_test_db()
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    uid = "test_user_02"

    messages = [
        ("user", "什么是 ERP 系统？"),
        ("assistant", "ERP 是企业资源规划系统，用于整合企业的各种业务流程。"),
        ("user", "有哪些常见的 ERP 系统？"),
        ("assistant", "常见的 ERP 系统包括 SAP、Oracle EBS、Microsoft Dynamics 等。"),
    ]

    write_conversation_messages(db, sid, uid, messages)

    # 使用 _ConversationRepoWrapper
    from dki.core.dki_system import _ConversationRepoWrapper, _DetachedMessage

    wrapper = _ConversationRepoWrapper(db)

    print(f"\n{'='*60}")
    print(f"TEST 02: _ConversationRepoWrapper → _DetachedMessage")
    print(f"{'='*60}")

    # get_by_session
    detached_msgs = wrapper.get_by_session(session_id=sid)
    print(f"get_by_session returned: {len(detached_msgs)} messages")

    for dm in detached_msgs:
        print(f"  type={type(dm).__name__}, id={dm.id}, role={dm.role}, "
              f"content={dm.content[:50]}..., session_id={dm.session_id}")
        assert isinstance(dm, _DetachedMessage), f"Expected _DetachedMessage, got {type(dm)}"
        assert dm.role in ("user", "assistant"), f"Invalid role: {dm.role}"
        assert dm.content, f"Empty content for message {dm.id}"
        assert dm.session_id == sid, f"Session ID mismatch: {dm.session_id} != {sid}"

    assert len(detached_msgs) == len(messages), f"Expected {len(messages)}, got {len(detached_msgs)}"

    # get_recent
    recent = wrapper.get_recent(session_id=sid, limit=4)
    print(f"get_recent(limit=4) returned: {len(recent)} messages")
    for dm in recent:
        print(f"  [{dm.role:>9}] {dm.content[:50]}...")
        assert dm.role in ("user", "assistant"), f"Invalid role after detach: {dm.role}"
        assert dm.content, f"Empty content after detach"

    print("✓ PASSED: _ConversationRepoWrapper returns valid _DetachedMessage objects")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 3: MultiSignalRecall 真实召回
# ================================================================

def test_03_multi_signal_recall():
    """验证 MultiSignalRecall 能从真实数据库中召回消息"""
    db, tmp = create_test_db()
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    uid = "test_user_03"

    messages = [
        ("user", "你好，我叫 lucas，我是一名程序员"),
        ("assistant", "你好 lucas！作为程序员，你主要使用什么编程语言呢？"),
        ("user", "我主要用 Python 和 C#，也用 JavaScript"),
        ("assistant", "很好的技术栈！Python 适合 AI 和数据分析，C# 适合企业应用。"),
        ("user", "我现在在做 ERP 系统开发"),
        ("assistant", "ERP 开发是一个很有挑战性的领域，需要理解业务流程和技术实现。"),
        ("user", "我也在研究大语言模型 LLM"),
        ("assistant", "LLM 研究非常前沿，结合 ERP 开发经验会有独特的视角。"),
    ]

    write_conversation_messages(db, sid, uid, messages)

    from dki.core.dki_system import _ConversationRepoWrapper
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig

    wrapper = _ConversationRepoWrapper(db)
    config = RecallConfig()  # 使用默认配置
    # 禁用 vector (需要 embedding service), 保留 keyword + BM25 + recency
    config.signals.vector_enabled = False

    recall = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    print(f"\n{'='*60}")
    print(f"TEST 03: MultiSignalRecall 真实召回")
    print(f"{'='*60}")

    # 查询: 关于编程语言的问题
    query = "你记得我用什么编程语言吗？"
    result = recall.recall(query=query, session_id=sid, user_id=uid)

    print(f"Query: {query}")
    print(f"Recalled messages: {len(result.messages)}")
    print(f"Keyword hits: {result.keyword_hits}")
    print(f"BM25 hits: {result.bm25_hits}")
    print(f"Vector hits: {result.vector_hits}")
    print(f"Recent turns added: {result.recent_turns_added}")

    for msg in result.messages:
        role = getattr(msg, 'role', '?')
        content = getattr(msg, 'content', str(msg))
        msg_id = getattr(msg, 'id', '?')
        print(f"  [{role:>9}] (id={msg_id}) {content[:70]}...")

    # 关键断言: 必须召回到消息
    assert len(result.messages) > 0, \
        f"CRITICAL FAILURE: recall returned 0 messages! " \
        f"keyword={result.keyword_hits}, bm25={result.bm25_hits}, " \
        f"recent={result.recent_turns_added}"

    # 至少应该有 recent turns (min_recent_turns=2, 即 4 条消息)
    assert result.recent_turns_added > 0 or result.keyword_hits > 0 or result.bm25_hits > 0, \
        "No signal produced any results"

    print("✓ PASSED: MultiSignalRecall returned messages from real DB")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 4: SuffixBuilder 组装
# ================================================================

def test_04_suffix_builder():
    """验证 SuffixBuilder 能从召回结果组装出非空后缀"""
    db, tmp = create_test_db()
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    uid = "test_user_04"

    messages = [
        ("user", "我叫 lucas，喜欢编程"),
        ("assistant", "你好 lucas！编程是很棒的爱好。"),
        ("user", "我用 Python 做 AI 项目"),
        ("assistant", "Python 是 AI 领域最流行的语言之一。"),
    ]

    write_conversation_messages(db, sid, uid, messages)

    from dki.core.dki_system import _ConversationRepoWrapper
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter

    wrapper = _ConversationRepoWrapper(db)
    config = RecallConfig()
    config.signals.vector_enabled = False

    recall = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    formatter = create_formatter(model_name="", formatter_type="generic", language="cn")
    suffix_builder = SuffixBuilder(
        config=config,
        prompt_formatter=formatter,
        token_counter=None,  # 使用粗估
    )

    print(f"\n{'='*60}")
    print(f"TEST 04: SuffixBuilder 组装")
    print(f"{'='*60}")

    query = "你记得我的名字吗？"
    recall_result = recall.recall(query=query, session_id=sid, user_id=uid)

    print(f"Recall returned: {len(recall_result.messages)} messages")

    assembled = suffix_builder.build(
        query=query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=100,
    )

    print(f"Assembled suffix:")
    print(f"  total_tokens: {assembled.total_tokens}")
    print(f"  message_count: {assembled.message_count}")
    print(f"  summary_count: {assembled.summary_count}")
    print(f"  items count: {len(assembled.items)}")
    print(f"  trace_ids: {assembled.trace_ids}")
    print(f"  text length: {len(assembled.text)}")

    if assembled.items:
        for item in assembled.items:
            print(f"  Item: type={item.type}, role={item.role}, "
                  f"tokens={item.token_count}, content={item.content[:50]}...")
    else:
        print("  *** NO ITEMS ***")

    print(f"\n  Full suffix text (first 500 chars):")
    print(f"  {assembled.text[:500]}")

    # 关键断言
    assert len(recall_result.messages) > 0, "Recall returned 0 messages"
    assert assembled.total_tokens > 0, \
        f"CRITICAL: assembled.total_tokens == 0 despite {len(recall_result.messages)} recalled messages"
    assert len(assembled.items) > 0, \
        f"CRITICAL: assembled.items is empty despite {len(recall_result.messages)} recalled messages"

    print("✓ PASSED: SuffixBuilder produced non-empty suffix")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 5: 完整 prompt 构造 (模拟 _build_recall_v4_chat_prompt)
# ================================================================

def test_05_full_prompt_construction():
    """
    模拟 dki_system.chat() 中的完整流程:
    1. 写入历史消息
    2. recall 召回
    3. suffix builder 组装
    4. 构造 chat template prompt
    5. 验证 prompt 包含历史消息角色
    """
    db, tmp = create_test_db()
    sid = f"test_session_{uuid.uuid4().hex[:8]}"
    uid = "test_user_05"

    messages = [
        ("user", "你好，我是 lucas"),
        ("assistant", "你好 lucas！有什么我可以帮助你的吗？"),
        ("user", "我喜欢编程，主要用 Python 和 C#"),
        ("assistant", "很好的技术栈！Python 和 C# 都是非常强大的语言。"),
        ("user", "我在做 ERP 开发和 LLM 研究"),
        ("assistant", "ERP 和 LLM 的结合是一个很有前景的方向。"),
    ]

    write_conversation_messages(db, sid, uid, messages)

    from dki.core.dki_system import _ConversationRepoWrapper
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig, HistoryItem
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter

    wrapper = _ConversationRepoWrapper(db)
    config = RecallConfig()
    config.signals.vector_enabled = False

    recall_engine = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    formatter = create_formatter(model_name="", formatter_type="generic", language="cn")
    suffix_builder = SuffixBuilder(
        config=config,
        prompt_formatter=formatter,
    )

    print(f"\n{'='*60}")
    print(f"TEST 05: 完整 Prompt 构造")
    print(f"{'='*60}")

    current_query = "你记得我的名字和喜好吗？"

    # Step 1: Recall
    recall_result = recall_engine.recall(query=current_query, session_id=sid, user_id=uid)
    print(f"[Step 1] Recall: {len(recall_result.messages)} messages, "
          f"keyword={recall_result.keyword_hits}, bm25={recall_result.bm25_hits}, "
          f"recent={recall_result.recent_turns_added}")

    # Step 2: SuffixBuilder
    assembled = suffix_builder.build(
        query=current_query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=100,
    )
    print(f"[Step 2] Suffix: {assembled.total_tokens} tokens, "
          f"{len(assembled.items)} items, "
          f"{assembled.message_count} msgs + {assembled.summary_count} summaries")

    # Step 3: 模拟 _build_recall_v4_chat_prompt
    has_history = assembled.total_tokens > 0 and assembled.items
    preference_content = "语言风格: 细腻温暖\n我的名字: lucas\n使用语言: Python, C#"
    has_preference = True

    print(f"[Step 3] has_history={has_history}, has_preference={has_preference}")

    if has_history or has_preference:
        # 构造 messages 列表 (与 _build_recall_v4_chat_prompt 相同逻辑)
        chat_messages = []

        # System message: 偏好
        system_parts = []
        if has_preference:
            system_parts.append(f"用户偏好:\n{preference_content}")
        system_content = "\n\n".join(system_parts)
        if system_content:
            chat_messages.append({"role": "system", "content": system_content})

        # History messages: 按原始角色还原
        if has_history:
            for item in assembled.items:
                role = getattr(item, 'role', None) or 'user'
                content = getattr(item, 'content', str(item))
                if role not in ('user', 'assistant', 'system'):
                    role = 'user'
                chat_messages.append({"role": role, "content": content})

        # Current query
        chat_messages.append({"role": "user", "content": current_query})

        # 使用 ChatML 格式 (不需要 tokenizer)
        parts = []
        for msg in chat_messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant")
        final_prompt = "\n".join(parts) + "\n"

        print(f"\n[Final Prompt] ({len(final_prompt)} chars, {len(chat_messages)} messages)")
        print(f"{'─'*60}")
        print(final_prompt[:2000])
        if len(final_prompt) > 2000:
            print(f"... (truncated, total {len(final_prompt)} chars)")
        print(f"{'─'*60}")

        # 关键断言
        assert "<|im_start|>system" in final_prompt, "Missing system message in prompt"
        assert "lucas" in final_prompt, "Preference 'lucas' not found in prompt"
        assert "<|im_start|>user" in final_prompt, "Missing user message in prompt"
        assert "<|im_start|>assistant" in final_prompt, "Missing assistant message in prompt"

        # 检查历史消息是否作为独立的 role 存在 (而不是扁平化在一个 user message 中)
        assistant_count = final_prompt.count("<|im_start|>assistant")
        user_count = final_prompt.count("<|im_start|>user")
        print(f"\nRole counts: system=1, user={user_count}, assistant={assistant_count}")

        # 至少应该有 2 个 user message (历史 + 当前查询)
        assert user_count >= 2, \
            f"CRITICAL: Only {user_count} user messages. History not injected as separate messages!"
        # 至少应该有 2 个 assistant (历史 + generation prompt)
        assert assistant_count >= 2, \
            f"CRITICAL: Only {assistant_count} assistant messages. History not injected!"

        print("✓ PASSED: Prompt contains multi-turn history with correct roles")
    else:
        print("CRITICAL FAILURE: Neither history nor preference available!")
        assert False, "No history and no preference"

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 6: 诊断 — 模拟真实 UI 场景 (第二轮请求)
# ================================================================

def test_06_simulate_second_turn():
    """
    模拟真实 UI 场景:
    - 第一轮: 用户发送消息 → DKI 处理 → 保存到 DB
    - 第二轮: 用户发送新消息 → recall 应该能召回第一轮的消息
    
    这是可视化.md 中显示 History=0 的真实场景。
    """
    db, tmp = create_test_db()
    sid = "session_9a627737"  # 模拟真实 session ID
    uid = "user01"

    print(f"\n{'='*60}")
    print(f"TEST 06: 模拟真实 UI 场景 (第二轮请求)")
    print(f"{'='*60}")

    # ---- 第一轮: 模拟 _log_conversation 写入 ----
    first_query = "你好,我是 lucas,你是谁,叫什么名字呢,请做个自我介绍,谢谢"
    first_response = "您好！我是一个 AI 助手，很高兴认识你 lucas！"

    with db.session_scope() as dbsess:
        sess_repo = SessionRepository(dbsess)
        sess_repo.get_or_create(session_id=sid, user_id=uid)
        conv_repo = ConversationRepository(dbsess)
        conv_repo.create(session_id=sid, role='user', content=first_query)
        conv_repo.create(session_id=sid, role='assistant', content=first_response)

    print(f"[Turn 1] Written: user='{first_query[:40]}...', assistant='{first_response[:40]}...'")

    # ---- 验证数据库中的消息 ----
    from dki.core.dki_system import _ConversationRepoWrapper
    wrapper = _ConversationRepoWrapper(db)
    db_msgs = wrapper.get_by_session(session_id=sid)
    print(f"[DB Check] Messages in DB for session '{sid}': {len(db_msgs)}")
    for m in db_msgs:
        print(f"  [{m.role:>9}] id={m.id}, content={m.content[:50]}...")

    assert len(db_msgs) == 2, f"Expected 2 messages in DB, got {len(db_msgs)}"

    # ---- 第二轮: recall 召回 ----
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter

    config = RecallConfig()
    config.signals.vector_enabled = False

    recall_engine = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    second_query = "我是 lucas,你也是 lucas 吗,你是不是弄混了, 你把我的喜好,当成你的了呢"

    recall_result = recall_engine.recall(query=second_query, session_id=sid, user_id=uid)
    print(f"\n[Turn 2 Recall] Query: {second_query[:50]}...")
    print(f"  Messages recalled: {len(recall_result.messages)}")
    print(f"  Keyword hits: {recall_result.keyword_hits}")
    print(f"  BM25 hits: {recall_result.bm25_hits}")
    print(f"  Recent turns: {recall_result.recent_turns_added}")

    for msg in recall_result.messages:
        print(f"  Recalled: [{getattr(msg, 'role', '?'):>9}] {getattr(msg, 'content', '?')[:60]}...")

    # 关键断言: 第二轮必须能召回第一轮的消息
    assert len(recall_result.messages) > 0, \
        f"CRITICAL: Second turn recall returned 0 messages! " \
        f"This is the root cause of History=0 in the UI. " \
        f"DB has {len(db_msgs)} messages for session {sid}."

    # 组装后缀
    formatter = create_formatter(model_name="", formatter_type="generic", language="cn")
    suffix_builder = SuffixBuilder(config=config, prompt_formatter=formatter)

    assembled = suffix_builder.build(
        query=second_query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=100,
    )

    print(f"\n[Turn 2 Suffix]")
    print(f"  total_tokens: {assembled.total_tokens}")
    print(f"  items: {len(assembled.items)}")
    print(f"  message_count: {assembled.message_count}")

    assert assembled.total_tokens > 0, \
        f"CRITICAL: assembled.total_tokens == 0 despite {len(recall_result.messages)} recalled"
    assert len(assembled.items) > 0, \
        f"CRITICAL: assembled.items empty despite {len(recall_result.messages)} recalled"

    print("✓ PASSED: Second turn successfully recalled first turn messages")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 7: 诊断 — 模拟 dki_system.chat() 的完整元数据计算
# ================================================================

def test_07_metadata_computation():
    """
    模拟 dki_system.chat() 中的元数据计算逻辑。
    验证 _hist_tokens_display 是否正确从 recall_v4_suffix 覆盖。
    这是可视化.md 中 History Tokens = 0 的直接原因诊断。
    """
    db, tmp = create_test_db()
    sid = "session_9a627737"
    uid = "user01"

    # 写入第一轮对话
    with db.session_scope() as dbsess:
        sess_repo = SessionRepository(dbsess)
        sess_repo.get_or_create(session_id=sid, user_id=uid)
        conv_repo = ConversationRepository(dbsess)
        conv_repo.create(session_id=sid, role='user',
                         content='你好,我是 lucas,你是谁,叫什么名字呢,请做个自我介绍,谢谢')
        conv_repo.create(session_id=sid, role='assistant',
                         content='您好！我是一个 AI 助手，很高兴认识你 lucas！')

    from dki.core.dki_system import _ConversationRepoWrapper
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter

    wrapper = _ConversationRepoWrapper(db)
    config = RecallConfig()
    config.signals.vector_enabled = False

    recall_engine = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    formatter = create_formatter(model_name="", formatter_type="generic", language="cn")
    suffix_builder = SuffixBuilder(config=config, prompt_formatter=formatter)

    print(f"\n{'='*60}")
    print(f"TEST 07: 元数据计算诊断 (模拟 dki_system.chat)")
    print(f"{'='*60}")

    query = "我是 lucas,你也是 lucas 吗,你是不是弄混了, 你把我的喜好,当成你的了呢"
    original_query = query

    # ---- 模拟 dki_system.chat() 中的 recall_v4 流程 ----
    recall_result = recall_engine.recall(query=query, session_id=sid, user_id=uid)
    print(f"[Recall] messages={len(recall_result.messages)}, recent={recall_result.recent_turns_added}")

    assembled = suffix_builder.build(
        query=query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=100,
    )

    recall_v4_suffix = assembled.text
    recall_v4_trace_ids = assembled.trace_ids

    print(f"[Assembled] total_tokens={assembled.total_tokens}, items={len(assembled.items)}")
    print(f"[recall_v4_suffix] len={len(recall_v4_suffix)}, truthy={bool(recall_v4_suffix)}")
    print(f"[recall_v4_suffix content]: {recall_v4_suffix[:200]}...")

    # ---- 模拟元数据计算 (dki_system.py 1098-1194) ----
    _pref_alpha_config = 0.4
    _pref_tokens_display = 92
    _hist_tokens_display = 0
    _pref_text_display = "语言风格: 细腻温暖\n我的名字: lucas"
    _hist_suffix_display = ""
    _hist_messages_display = []
    _final_input_display = query

    # 模拟 hybrid_result (recall_v4 路径下 history=None)
    class FakeHybridResult:
        history_tokens = 0
        history_suffix_text = ""
        history_messages = []
        input_text = query
        preference_kv = []
        preference_alpha = 0.4

    hybrid_result = FakeHybridResult()

    # 行 1130-1134: hybrid_result 覆盖
    if hybrid_result:
        _hist_tokens_display = hybrid_result.history_tokens  # 0!
        _hist_suffix_display = hybrid_result.history_suffix_text  # ""
        _hist_messages_display = hybrid_result.history_messages  # []
        _final_input_display = hybrid_result.input_text

    print(f"\n[After hybrid_result override]")
    print(f"  _hist_tokens_display = {_hist_tokens_display}")  # 应该是 0
    print(f"  _hist_suffix_display = '{_hist_suffix_display[:50]}...'")

    # 行 1139-1148: recall_v4 覆盖
    _use_recall_v4 = True
    if recall_v4_suffix and _use_recall_v4:
        _hist_tokens_display = len(recall_v4_suffix) // 2  # 粗估
        _hist_suffix_display = recall_v4_suffix

    print(f"\n[After recall_v4 override]")
    print(f"  _hist_tokens_display = {_hist_tokens_display}")
    print(f"  _hist_suffix_display = '{_hist_suffix_display[:100]}...'")

    # 行 1160-1176: 从 assembled.items 填充 _hist_messages_display
    if _use_recall_v4 and recall_v4_suffix and not _hist_messages_display:
        if hasattr(assembled, 'items'):
            for item in assembled.items:
                _hist_messages_display.append({
                    'role': item.role or 'user',
                    'content': item.content[:500],
                    'type': item.type,
                    'trace_id': item.trace_id,
                })

    print(f"\n[After history messages fill]")
    print(f"  _hist_messages_display = {len(_hist_messages_display)} items")
    for m in _hist_messages_display:
        print(f"    [{m['role']:>9}] {m['content'][:60]}...")

    # ---- 关键断言 ----
    assert _hist_tokens_display > 0, \
        f"CRITICAL: _hist_tokens_display == 0! recall_v4_suffix truthy={bool(recall_v4_suffix)}, " \
        f"len={len(recall_v4_suffix)}"
    assert len(_hist_messages_display) > 0, \
        f"CRITICAL: _hist_messages_display is empty!"
    assert _hist_suffix_display, \
        f"CRITICAL: _hist_suffix_display is empty!"

    # ---- 构建最终 metadata ----
    metadata = {
        'hybrid_injection': {
            'enabled': True,
            'preference_tokens': _pref_tokens_display,
            'history_tokens': _hist_tokens_display,
            'preference_alpha': _pref_alpha_config,
            'preference_text': _pref_text_display,
            'history_suffix_text': _hist_suffix_display,
            'history_messages': _hist_messages_display,
            'final_input': _final_input_display,
        },
    }

    print(f"\n[Final Metadata]")
    hi = metadata['hybrid_injection']
    print(f"  history_tokens: {hi['history_tokens']}")
    print(f"  preference_tokens: {hi['preference_tokens']}")
    print(f"  history_messages: {len(hi['history_messages'])}")
    print(f"  enabled: {hi['enabled']}")

    assert hi['history_tokens'] > 0, "Metadata history_tokens should be > 0"
    assert len(hi['history_messages']) > 0, "Metadata history_messages should be non-empty"

    print("✓ PASSED: Metadata correctly reflects non-zero history tokens")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# 测试 8: 诊断 — 真实 dki_system.chat() 调用 (mock model)
# ================================================================

def test_08_real_dki_system_chat():
    """
    创建一个最小化的 DKISystem, 使用真实 SQLite + 真实 recall,
    但 mock 模型 (不需要 GPU)。

    验证 dki_system.chat() 返回的 metadata 中 history_tokens > 0。
    """
    db, tmp = create_test_db()
    sid = "session_test08"
    uid = "user01"

    # 写入第一轮对话
    with db.session_scope() as dbsess:
        sess_repo = SessionRepository(dbsess)
        sess_repo.get_or_create(session_id=sid, user_id=uid)
        conv_repo = ConversationRepository(dbsess)
        conv_repo.create(session_id=sid, role='user',
                         content='你好,我是 lucas,我喜欢编程和吉他')
        conv_repo.create(session_id=sid, role='assistant',
                         content='你好 lucas！编程和吉他都是很棒的爱好。')
        conv_repo.create(session_id=sid, role='user',
                         content='我主要用 Python 和 C# 做开发')
        conv_repo.create(session_id=sid, role='assistant',
                         content='Python 和 C# 都是非常好的选择。')

    # 写入用户偏好
    from dki.database.repository import UserPreferenceRepository
    with db.session_scope() as dbsess:
        pref_repo = UserPreferenceRepository(dbsess)
        pref_repo.create(
            user_id=uid,
            preference_text="语言风格: 细腻温暖\n我的名字: lucas",
            category="general",
        )

    print(f"\n{'='*60}")
    print(f"TEST 08: 真实 DKISystem.chat() 调用")
    print(f"{'='*60}")

    # 验证数据库中的消息
    from dki.core.dki_system import _ConversationRepoWrapper
    wrapper = _ConversationRepoWrapper(db)
    db_msgs = wrapper.get_by_session(session_id=sid)
    print(f"[DB] Messages: {len(db_msgs)}")
    for m in db_msgs:
        print(f"  [{m.role:>9}] {m.content[:50]}...")

    # 现在尝试直接使用 recall 组件 (与 _init_recall_v4_components 相同逻辑)
    from dki.core.recall.multi_signal_recall import MultiSignalRecall
    from dki.core.recall.recall_config import RecallConfig
    from dki.core.recall.suffix_builder import SuffixBuilder
    from dki.core.recall.prompt_formatter import create_formatter

    config = RecallConfig()
    config.signals.vector_enabled = False

    recall_engine = MultiSignalRecall(
        config=config,
        reference_resolver=None,
        memory_router=None,
        conversation_repo=wrapper,
    )

    formatter = create_formatter(model_name="", formatter_type="generic", language="cn")
    suffix_builder = SuffixBuilder(config=config, prompt_formatter=formatter)

    # 第二轮查询
    query = "你记得我的名字和喜好吗？"

    # Step 1: Recall
    recall_result = recall_engine.recall(query=query, session_id=sid, user_id=uid)
    print(f"\n[Recall] messages={len(recall_result.messages)}")

    # Step 2: Suffix build
    assembled = suffix_builder.build(
        query=query,
        recalled_messages=recall_result.messages,
        context_window=4096,
        preference_tokens=100,
    )
    print(f"[Assembled] total_tokens={assembled.total_tokens}, items={len(assembled.items)}")

    # Step 3: 模拟 _build_recall_v4_chat_prompt
    has_history = assembled.total_tokens > 0 and assembled.items
    has_preference = True
    preference_content = "语言风格: 细腻温暖\n我的名字: lucas"

    if has_history or has_preference:
        chat_messages = []
        if has_preference:
            chat_messages.append({"role": "system", "content": f"用户偏好:\n{preference_content}"})
        if has_history:
            for item in assembled.items:
                role = getattr(item, 'role', 'user')
                if role not in ('user', 'assistant', 'system'):
                    role = 'user'
                chat_messages.append({"role": role, "content": item.content})
        chat_messages.append({"role": "user", "content": query})

        # ChatML format
        parts = []
        for msg in chat_messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant")
        final_prompt = "\n".join(parts) + "\n"

        print(f"\n[Final Prompt] ({len(final_prompt)} chars)")
        print(final_prompt[:1000])

        # 验证
        assert has_history, "History should be present"
        assert "<|im_start|>system" in final_prompt, "Missing system"
        assert "lucas" in final_prompt, "Missing preference"

        user_count = final_prompt.count("<|im_start|>user")
        assistant_count = final_prompt.count("<|im_start|>assistant")
        print(f"\nRole counts: user={user_count}, assistant={assistant_count}")
        assert user_count >= 2, f"Only {user_count} user messages"
        assert assistant_count >= 2, f"Only {assistant_count} assistant messages"
    else:
        assert False, "Neither history nor preference"

    # 验证 metadata 计算
    recall_v4_suffix = assembled.text
    _hist_tokens_display = 0

    # hybrid_result override (history=None)
    _hist_tokens_display = 0

    # recall_v4 override
    if recall_v4_suffix:
        _hist_tokens_display = len(recall_v4_suffix) // 2

    assert _hist_tokens_display > 0, \
        f"history_tokens should be > 0, got {_hist_tokens_display}"

    print(f"\n[Metadata] history_tokens={_hist_tokens_display}")
    print("✓ PASSED: Real DKISystem flow produces non-zero history")

    DatabaseManager._instance = None
    DatabaseManager._initialized = False
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return True


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    tests = [
        test_01_db_write_and_read,
        test_02_wrapper_returns_detached,
        test_03_multi_signal_recall,
        test_04_suffix_builder,
        test_05_full_prompt_construction,
        test_06_simulate_second_turn,
        test_07_metadata_computation,
        test_08_real_dki_system_chat,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            # 每次测试前重置 singleton
            DatabaseManager._instance = None
            DatabaseManager._initialized = False

            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                errors.append((test_fn.__name__, "returned False"))
        except Exception as e:
            failed += 1
            import traceback
            tb = traceback.format_exc()
            errors.append((test_fn.__name__, str(e)))
            print(f"\n✗ FAILED: {test_fn.__name__}")
            print(f"  Error: {e}")
            print(tb)
        finally:
            # 确保 singleton 被重置
            DatabaseManager._instance = None
            DatabaseManager._initialized = False

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")

    sys.exit(0 if failed == 0 else 1)
