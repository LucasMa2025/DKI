"""
实验系统 Store 层单元测试

测试范围:
1. ExperimentDBConfig 配置
2. ExperimentDBManager 数据库初始化与 schema 迁移
3. SQLiteChatStore CRUD 操作 (用户/会话/消息/偏好)
4. create_experiment_store 工厂函数
5. Schema 迁移 (旧表缺少新列的场景)

Author: AGI Demo Project
Version: 1.0.0
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DKI_ENV", "test")

from dki.experiment.store.connection import ExperimentDBConfig, ExperimentDBManager
from dki.experiment.store.models import DemoBase, DemoUser, DemoSession, DemoMessage, DemoPreference
from dki.experiment.store.factory import create_experiment_store
from dki.experiment.store.base import IChatStore, StoreError


class TestExperimentDBConfig(unittest.TestCase):
    """ExperimentDBConfig 配置测试"""

    def test_default_config(self):
        """默认配置应为 SQLite, 路径 ./data/dki.db"""
        cfg = ExperimentDBConfig()
        self.assertEqual(cfg.backend, "sqlite")
        self.assertEqual(cfg.sqlite_path, "./data/dki.db")
        self.assertTrue(cfg.enable_wal)
        self.assertEqual(cfg.busy_timeout_ms, 5000)

    def test_custom_config(self):
        """自定义配置"""
        cfg = ExperimentDBConfig(
            sqlite_path="/tmp/test.db",
            enable_wal=False,
            busy_timeout_ms=3000,
            echo=True,
        )
        self.assertEqual(cfg.sqlite_path, "/tmp/test.db")
        self.assertFalse(cfg.enable_wal)
        self.assertEqual(cfg.busy_timeout_ms, 3000)
        self.assertTrue(cfg.echo)

    def test_connection_url(self):
        """连接 URL 生成"""
        cfg = ExperimentDBConfig(sqlite_path="./data/test.db")
        self.assertEqual(cfg.get_connection_url(), "sqlite:///./data/test.db")

    def test_from_dict(self):
        """从字典创建配置"""
        data = {
            "backend": "sqlite",
            "sqlite_path": "/custom/path.db",
            "enable_wal": False,
            "echo": True,
        }
        cfg = ExperimentDBConfig.from_dict(data)
        self.assertEqual(cfg.sqlite_path, "/custom/path.db")
        self.assertFalse(cfg.enable_wal)
        self.assertTrue(cfg.echo)

    def test_from_empty_dict(self):
        """空字典使用默认值"""
        cfg = ExperimentDBConfig.from_dict({})
        self.assertEqual(cfg.backend, "sqlite")
        self.assertEqual(cfg.sqlite_path, "./data/dki.db")


class TestExperimentDBManager(unittest.TestCase):
    """ExperimentDBManager 测试"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_mgr.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_init_database_creates_tables(self):
        """init_database 应创建所有 demo_* 表"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        self.assertTrue(mgr.is_connected)

        # 检查表是否存在
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor}
        conn.close()

        self.assertIn("demo_users", tables)
        self.assertIn("demo_sessions", tables)
        self.assertIn("demo_messages", tables)
        self.assertIn("demo_preferences", tables)

    def test_init_database_creates_directory(self):
        """init_database 应自动创建数据库目录"""
        nested_path = os.path.join(self.tmp_dir, "sub", "dir", "test.db")
        cfg = ExperimentDBConfig(sqlite_path=nested_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        self.assertTrue(os.path.exists(nested_path))

    def test_session_scope_commit(self):
        """session_scope 应自动提交"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        with mgr.session_scope() as session:
            user = DemoUser(
                id="test_001",
                username="test_user",
                display_name="Test User",
            )
            session.add(user)

        # 验证数据已持久化
        with mgr.session_scope() as session:
            found = session.query(DemoUser).filter_by(id="test_001").first()
            self.assertIsNotNone(found)
            self.assertEqual(found.username, "test_user")

    def test_session_scope_rollback_on_error(self):
        """session_scope 在异常时应回滚"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        # 先添加一个用户
        with mgr.session_scope() as session:
            user = DemoUser(id="u1", username="existing_user")
            session.add(user)

        # 尝试添加重复用户 (应触发 IntegrityError)
        with self.assertRaises(Exception):
            with mgr.session_scope() as session:
                dup = DemoUser(id="u2", username="existing_user")  # duplicate username
                session.add(dup)

        self.assertGreater(mgr.stats.error_count, 0)

    def test_dispose(self):
        """dispose 应断开连接"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()
        self.assertTrue(mgr.is_connected)

        mgr.dispose()
        self.assertFalse(mgr.is_connected)

    def test_pool_stats(self):
        """get_pool_stats 应返回统计信息"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        stats = mgr.get_pool_stats()
        self.assertEqual(stats["backend"], "sqlite")
        self.assertTrue(stats["connected"])
        self.assertEqual(stats["total_sessions"], 0)


class TestSchemaMigration(unittest.TestCase):
    """Schema 迁移测试 — 修复旧表缺少新列的问题"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_migration.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_migrate_adds_missing_password_hash(self):
        """当 demo_users 表缺少 password_hash 列时, init_database 应自动添加"""
        # 1. 手动创建旧版 demo_users 表 (无 password_hash)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE demo_users (
                id VARCHAR(64) PRIMARY KEY,
                username VARCHAR(64) NOT NULL UNIQUE,
                display_name VARCHAR(128),
                email VARCHAR(128),
                avatar VARCHAR(256),
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME,
                last_login_at DATETIME,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute(
            "INSERT INTO demo_users (id, username) VALUES ('u1', 'old_user')"
        )
        conn.commit()

        # 验证旧表没有 password_hash
        cursor = conn.execute("PRAGMA table_info(demo_users)")
        old_columns = {row[1] for row in cursor}
        self.assertNotIn("password_hash", old_columns)
        conn.close()

        # 2. 运行 init_database (应触发迁移)
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        # 3. 验证 password_hash 列已添加
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(demo_users)")
        new_columns = {row[1] for row in cursor}
        self.assertIn("password_hash", new_columns)

        # 4. 验证旧数据未丢失
        cursor = conn.execute("SELECT username FROM demo_users WHERE id='u1'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "old_user")

        # 5. 验证 ORM 查询正常
        with mgr.session_scope() as session:
            user = session.query(DemoUser).filter_by(id="u1").first()
            self.assertIsNotNone(user)
            self.assertEqual(user.username, "old_user")
            self.assertIsNone(user.password_hash)  # 新列默认 NULL

        conn.close()
        mgr.dispose()

    def test_migrate_no_op_when_column_exists(self):
        """当列已存在时, 迁移应为 no-op"""
        # 先正常创建完整表
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        # 再次调用 init_database (不应报错)
        mgr2 = ExperimentDBManager(cfg)
        mgr2.init_database()
        self.assertTrue(mgr2.is_connected)

        mgr.dispose()
        mgr2.dispose()

    def test_migrate_no_op_when_table_missing(self):
        """当表不存在时, create_all 会创建完整表, 迁移应为 no-op"""
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        mgr = ExperimentDBManager(cfg)
        mgr.init_database()

        # 验证所有表都有完整 schema
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(demo_users)")
        columns = {row[1] for row in cursor}
        self.assertIn("password_hash", columns)
        conn.close()

        mgr.dispose()


class TestSQLiteChatStoreCRUD(unittest.TestCase):
    """SQLiteChatStore CRUD 操作测试"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_crud.db")
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        self.store = create_experiment_store(cfg)

    def tearDown(self):
        self.store.disconnect()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---- User CRUD ----

    def test_create_user(self):
        """创建用户"""
        user, created = self.store.get_or_create_user("alice", "Alice")
        self.assertTrue(created)
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.display_name, "Alice")
        self.assertIsNotNone(user.id)

    def test_get_existing_user(self):
        """获取已存在的用户"""
        user1, created1 = self.store.get_or_create_user("bob", "Bob")
        user2, created2 = self.store.get_or_create_user("bob", "Bob Updated")
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(user1.id, user2.id)

    def test_get_user_by_id(self):
        """通过 ID 获取用户"""
        user, _ = self.store.get_or_create_user("charlie", "Charlie")
        found = self.store.get_user(user.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.username, "charlie")

    # ---- Session CRUD ----

    def test_create_session(self):
        """创建会话"""
        user, _ = self.store.get_or_create_user("user1", "User 1")
        session = self.store.create_session(user.id, "Test Session", "sess_001")
        self.assertEqual(session.id, "sess_001")
        self.assertEqual(session.user_id, user.id)

    def test_get_session(self):
        """获取会话"""
        user, _ = self.store.get_or_create_user("user2", "User 2")
        self.store.create_session(user.id, "S1", "sess_002")
        found = self.store.get_session("sess_002")
        self.assertIsNotNone(found)
        self.assertEqual(found.title, "S1")

    def test_get_nonexistent_session(self):
        """获取不存在的会话应返回 None"""
        found = self.store.get_session("nonexistent_session")
        self.assertIsNone(found)

    # ---- Message CRUD ----

    def test_add_message(self):
        """添加消息"""
        user, _ = self.store.get_or_create_user("msg_user", "Msg User")
        self.store.create_session(user.id, "Chat", "sess_msg")
        msg = self.store.add_message("sess_msg", user.id, "user", "Hello!")
        self.assertEqual(msg.content, "Hello!")
        self.assertEqual(msg.role, "user")

    def test_get_messages(self):
        """获取会话消息列表"""
        user, _ = self.store.get_or_create_user("msg_user2", "Msg User 2")
        self.store.create_session(user.id, "Chat2", "sess_msgs")

        self.store.add_message("sess_msgs", user.id, "user", "Hi")
        self.store.add_message("sess_msgs", user.id, "assistant", "Hello!")
        self.store.add_message("sess_msgs", user.id, "user", "How are you?")

        msgs = self.store.get_messages("sess_msgs")
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[1].role, "assistant")

    def test_get_messages_with_limit(self):
        """获取消息应支持 limit"""
        user, _ = self.store.get_or_create_user("lim_user", "Limit User")
        self.store.create_session(user.id, "Chat", "sess_lim")
        for i in range(10):
            self.store.add_message("sess_lim", user.id, "user", f"Message {i}")

        msgs = self.store.get_messages("sess_lim", limit=5)
        self.assertEqual(len(msgs), 5)

    # ---- Preference CRUD ----

    def test_add_preference(self):
        """添加偏好"""
        user, _ = self.store.get_or_create_user("pref_user", "Pref User")
        pref = self.store.add_preference(
            user.id, "I am vegetarian", "dietary", priority=10
        )
        self.assertEqual(pref.preference_text, "I am vegetarian")
        self.assertEqual(pref.preference_type, "dietary")
        self.assertEqual(pref.priority, 10)

    def test_get_preferences(self):
        """获取用户偏好列表"""
        user, _ = self.store.get_or_create_user("pref_user2", "Pref User 2")
        self.store.add_preference(user.id, "Pref A", "general", priority=5)
        self.store.add_preference(user.id, "Pref B", "dietary", priority=10)

        prefs = self.store.get_preferences(user.id)
        self.assertEqual(len(prefs), 2)
        texts = {p.preference_text for p in prefs}
        self.assertIn("Pref A", texts)
        self.assertIn("Pref B", texts)

    def test_delete_preference(self):
        """删除偏好"""
        user, _ = self.store.get_or_create_user("del_pref_user", "Del Pref User")
        pref = self.store.add_preference(user.id, "To delete", "general")
        self.store.delete_preference(pref.id)

        prefs = self.store.get_preferences(user.id)
        # 软删除: is_active=False, 或者 get_preferences 只返回 active 的
        active_texts = {p.preference_text for p in prefs if p.is_active}
        self.assertNotIn("To delete", active_texts)


class TestCreateExperimentStore(unittest.TestCase):
    """create_experiment_store 工厂函数测试"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_creates_sqlite_store(self):
        """应创建 SQLiteChatStore 实例"""
        db_path = os.path.join(self.tmp_dir, "factory_test.db")
        cfg = ExperimentDBConfig(sqlite_path=db_path)
        store = create_experiment_store(cfg)

        self.assertIsNotNone(store)
        self.assertIsInstance(store, IChatStore)
        store.disconnect()

    def test_rejects_unsupported_backend(self):
        """不支持的后端应抛出 StoreError"""
        cfg = ExperimentDBConfig(backend="postgresql", sqlite_path="irrelevant")
        with self.assertRaises(StoreError):
            create_experiment_store(cfg)

    def test_store_is_connected(self):
        """创建后 store 应已连接"""
        db_path = os.path.join(self.tmp_dir, "connected_test.db")
        cfg = ExperimentDBConfig(sqlite_path=db_path)
        store = create_experiment_store(cfg)

        # 应能执行基本操作
        user, created = store.get_or_create_user("factory_user", "Factory User")
        self.assertTrue(created)
        store.disconnect()


class TestDkiBridge(unittest.TestCase):
    """dki_bridge.py 配置生成测试"""

    def test_build_adapter_config_structure(self):
        """生成的配置应包含正确的表映射"""
        from dki.experiment.dki_bridge import build_experiment_adapter_config

        cfg = ExperimentDBConfig(sqlite_path="./data/test.db")
        adapter_config = build_experiment_adapter_config(cfg)

        # 检查顶层键
        self.assertIn("database", adapter_config)
        self.assertIn("preferences", adapter_config)
        self.assertIn("messages", adapter_config)
        self.assertIn("users", adapter_config)
        self.assertIn("sessions", adapter_config)

        # 检查数据库配置
        self.assertEqual(adapter_config["database"]["type"], "sqlite")
        self.assertEqual(adapter_config["database"]["database"], "./data/test.db")

        # 检查表名映射
        self.assertEqual(adapter_config["preferences"]["table"], "demo_preferences")
        self.assertEqual(adapter_config["messages"]["table"], "demo_messages")
        self.assertEqual(adapter_config["users"]["table"], "demo_users")
        self.assertEqual(adapter_config["sessions"]["table"], "demo_sessions")

    def test_adapter_config_fields_mapping(self):
        """字段映射应正确"""
        from dki.experiment.dki_bridge import build_experiment_adapter_config

        cfg = ExperimentDBConfig(sqlite_path="./data/test.db")
        adapter_config = build_experiment_adapter_config(cfg)

        # 偏好字段
        pref_fields = adapter_config["preferences"]["fields"]
        self.assertEqual(pref_fields["user_id"], "user_id")
        self.assertEqual(pref_fields["preference_text"], "preference_text")
        self.assertEqual(pref_fields["priority"], "priority")

        # 消息字段
        msg_fields = adapter_config["messages"]["fields"]
        self.assertEqual(msg_fields["content"], "content")
        self.assertEqual(msg_fields["role"], "role")
        self.assertEqual(msg_fields["session_id"], "session_id")


class TestExperimentModels(unittest.TestCase):
    """ORM 模型测试"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_models.db")
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        self.mgr = ExperimentDBManager(cfg)
        self.mgr.init_database()

    def tearDown(self):
        self.mgr.dispose()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_demo_user_to_dict(self):
        """DemoUser.to_dict() 应返回正确格式"""
        with self.mgr.session_scope() as session:
            user = DemoUser(id="u1", username="test", display_name="Test")
            session.add(user)

        with self.mgr.session_scope() as session:
            user = session.query(DemoUser).filter_by(id="u1").first()
            d = user.to_dict()
            self.assertEqual(d["id"], "u1")
            self.assertEqual(d["username"], "test")
            # to_dict 可能使用 camelCase 或 snake_case, 检查任一
            self.assertTrue(
                "created_at" in d or "createdAt" in d,
                f"Neither 'created_at' nor 'createdAt' found in {list(d.keys())}"
            )

    def test_demo_preference_to_dict(self):
        """DemoPreference.to_dict() 应返回正确格式"""
        with self.mgr.session_scope() as session:
            user = DemoUser(id="u2", username="pref_test")
            session.add(user)
            pref = DemoPreference(
                id="p1",
                user_id="u2",
                preference_text="I like Python",
                preference_type="technical",
                priority=8,
                is_active=True,
            )
            session.add(pref)

        with self.mgr.session_scope() as session:
            pref = session.query(DemoPreference).filter_by(id="p1").first()
            d = pref.to_dict()
            self.assertEqual(d["preference_text"], "I like Python")
            self.assertEqual(d["priority"], 8)
            self.assertTrue(d["is_active"])

    def test_demo_message_to_dict(self):
        """DemoMessage.to_dict() 应返回正确格式"""
        with self.mgr.session_scope() as session:
            user = DemoUser(id="u3", username="msg_test")
            session.add(user)
            sess = DemoSession(id="s1", user_id="u3", title="Test")
            session.add(sess)
            msg = DemoMessage(
                id="m1",
                session_id="s1",
                user_id="u3",
                role="user",
                content="Hello world",
            )
            session.add(msg)

        with self.mgr.session_scope() as session:
            msg = session.query(DemoMessage).filter_by(id="m1").first()
            d = msg.to_dict()
            self.assertEqual(d["content"], "Hello world")
            self.assertEqual(d["role"], "user")
            self.assertEqual(d["session_id"], "s1")

    def test_user_password_hash_nullable(self):
        """password_hash 列应可为 NULL"""
        with self.mgr.session_scope() as session:
            user = DemoUser(id="u4", username="no_password")
            session.add(user)

        with self.mgr.session_scope() as session:
            user = session.query(DemoUser).filter_by(id="u4").first()
            self.assertIsNone(user.password_hash)

    def test_user_password_hash_set(self):
        """password_hash 列可以设置值"""
        with self.mgr.session_scope() as session:
            user = DemoUser(
                id="u5",
                username="with_password",
                password_hash="hashed_value_123",
            )
            session.add(user)

        with self.mgr.session_scope() as session:
            user = session.query(DemoUser).filter_by(id="u5").first()
            self.assertEqual(user.password_hash, "hashed_value_123")


class TestExperimentRunnerDataclasses(unittest.TestCase):
    """ExperimentRunner 数据类测试"""

    def test_injection_info_to_dict(self):
        """InjectionInfo.to_dict() 应包含所有字段"""
        from dki.experiment.runner import InjectionInfo

        info = InjectionInfo(
            mode="dki",
            original_query="推荐素食",
            preference_text="我是素食主义者",
            preference_tokens=50,
            history_suffix="历史记录",
            history_tokens=30,
            alpha=0.4,
        )
        d = info.to_dict()
        self.assertEqual(d["mode"], "dki")
        self.assertEqual(d["preference_text"], "我是素食主义者")
        self.assertEqual(d["preference_tokens"], 50)
        self.assertEqual(d["alpha"], 0.4)

    def test_injection_info_display_text(self):
        """InjectionInfo.get_display_text() 应返回格式化文本"""
        from dki.experiment.runner import InjectionInfo

        info = InjectionInfo(
            mode="dki",
            original_query="推荐素食",
            preference_text="我是素食主义者",
            preference_tokens=50,
            alpha=0.4,
        )
        text = info.get_display_text()
        self.assertIn("DKI", text)
        self.assertIn("推荐素食", text)
        self.assertIn("素食主义者", text)

    def test_experiment_config_to_dict(self):
        """ExperimentConfig.to_dict() 应包含所有配置"""
        from dki.experiment.runner import ExperimentConfig

        cfg = ExperimentConfig(
            name="test_exp",
            description="A test experiment",
            modes=["dki", "rag"],
            max_samples=10,
        )
        d = cfg.to_dict()
        self.assertEqual(d["name"], "test_exp")
        self.assertEqual(d["modes"], ["dki", "rag"])
        self.assertEqual(d["max_samples"], 10)

    def test_experiment_result_to_dict(self):
        """ExperimentResult.to_dict() 应包含注入信息"""
        from dki.experiment.runner import ExperimentResult, InjectionInfo

        info = InjectionInfo(mode="dki", original_query="test")
        result = ExperimentResult(
            mode="dki",
            dataset="persona_chat",
            sample_id="s001",
            query="test query",
            response="test response",
            latency_ms=100.0,
            memories_used=["pref:3"],
            alpha=0.4,
            injection_info=info,
        )
        d = result.to_dict()
        self.assertEqual(d["mode"], "dki")
        self.assertIsNotNone(d["injection_info"])
        self.assertEqual(d["injection_info"]["mode"], "dki")


class TestExperimentRunnerImports(unittest.TestCase):
    """验证 runner.py 的导入和结构"""

    def test_no_dki_system_import(self):
        """runner.py 不应导入 DKISystem"""
        import dki.experiment.runner as runner_module
        self.assertFalse(hasattr(runner_module, 'DKISystem'))

    def test_runner_init_signature(self):
        """ExperimentRunner.__init__ 应接受正确的参数"""
        import inspect
        from dki.experiment.runner import ExperimentRunner

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())

        self.assertNotIn('dki_system', params)
        self.assertIn('dki_plugin', params)
        self.assertIn('model_adapter', params)
        self.assertIn('db_path', params)

    def test_runner_has_store_methods(self):
        """ExperimentRunner 应有 store 相关方法"""
        from dki.experiment.runner import ExperimentRunner

        self.assertTrue(hasattr(ExperimentRunner, '_store_add_message'))
        self.assertTrue(hasattr(ExperimentRunner, '_store_ensure_session'))
        self.assertTrue(hasattr(ExperimentRunner, '_extract_injection_info_from_meta'))

    def test_runner_has_experiment_methods(self):
        """ExperimentRunner 应有各实验方法"""
        from dki.experiment.runner import ExperimentRunner

        self.assertTrue(hasattr(ExperimentRunner, 'run_experiment'))
        self.assertTrue(hasattr(ExperimentRunner, 'run_persona_chat_experiment'))
        self.assertTrue(hasattr(ExperimentRunner, 'run_longmemeval'))
        self.assertTrue(hasattr(ExperimentRunner, 'run_alpha_sensitivity'))
        self.assertTrue(hasattr(ExperimentRunner, 'run_ablation_study'))
        self.assertTrue(hasattr(ExperimentRunner, 'run_context_constrained'))

    def test_runner_model_property(self):
        """ExperimentRunner 应有 model 属性"""
        from dki.experiment.runner import ExperimentRunner
        self.assertTrue(hasattr(ExperimentRunner, 'model'))


class TestExperimentPackageInit(unittest.TestCase):
    """验证 __init__.py 导出正确"""

    def test_runner_exported(self):
        """ExperimentRunner 应从 experiment 包导出"""
        from dki.experiment import ExperimentRunner
        self.assertTrue(hasattr(ExperimentRunner, 'run_experiment'))

    def test_bridge_exported(self):
        """build_experiment_adapter_config 应从 experiment 包导出"""
        from dki.experiment import build_experiment_adapter_config
        self.assertTrue(callable(build_experiment_adapter_config))

    def test_injection_info_exported(self):
        """InjectionInfo 应从 experiment 包导出"""
        from dki.experiment import InjectionInfo
        self.assertTrue(hasattr(InjectionInfo, 'to_dict'))

    def test_no_sqlite_adapter_exported(self):
        """旧的 SQLiteDataAdapter 不应再导出"""
        import dki.experiment as exp_module
        self.assertFalse(hasattr(exp_module, 'SQLiteDataAdapter'))


class TestEndToEndStoreWorkflow(unittest.TestCase):
    """端到端 Store 工作流测试 — 模拟实验系统的典型操作序列"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "e2e_test.db")
        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        self.store = create_experiment_store(cfg)

    def tearDown(self):
        self.store.disconnect()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_experiment_workflow(self):
        """模拟完整的实验工作流:
        1. 创建用户
        2. 写入偏好
        3. 创建会话
        4. 添加多轮消息
        5. 查询偏好和消息
        """
        # Step 1: 创建用户
        user, created = self.store.get_or_create_user(
            "exp_user_vegetarian", "素食实验用户"
        )
        self.assertTrue(created)

        # Step 2: 写入偏好
        pref1 = self.store.add_preference(
            user.id, "我是素食主义者，不吃任何肉类和海鲜", "general", priority=10
        )
        pref2 = self.store.add_preference(
            user.id, "我住在北京海淀区", "general", priority=7
        )

        # Step 3: 创建会话
        session = self.store.create_session(user.id, "PersonaChat Exp", "exp_sess_001")

        # Step 4: 添加多轮消息
        self.store.add_message("exp_sess_001", user.id, "user", "推荐一些适合我的午餐")
        self.store.add_message(
            "exp_sess_001", user.id, "assistant",
            "根据您的素食偏好，推荐蔬菜沙拉和豆腐汤"
        )
        self.store.add_message("exp_sess_001", user.id, "user", "附近有什么好的素食餐厅？")
        self.store.add_message(
            "exp_sess_001", user.id, "assistant",
            "在北京海淀区，推荐净心莲素食餐厅"
        )

        # Step 5: 查询验证
        prefs = self.store.get_preferences(user.id)
        self.assertEqual(len(prefs), 2)
        pref_texts = {p.preference_text for p in prefs}
        self.assertIn("我是素食主义者，不吃任何肉类和海鲜", pref_texts)

        msgs = self.store.get_messages("exp_sess_001")
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[1].role, "assistant")

    def test_multi_user_isolation(self):
        """多用户数据隔离"""
        user1, _ = self.store.get_or_create_user("user_a", "User A")
        user2, _ = self.store.get_or_create_user("user_b", "User B")

        self.store.add_preference(user1.id, "Pref for A", "general")
        self.store.add_preference(user2.id, "Pref for B", "general")

        prefs_a = self.store.get_preferences(user1.id)
        prefs_b = self.store.get_preferences(user2.id)

        self.assertEqual(len(prefs_a), 1)
        self.assertEqual(len(prefs_b), 1)
        self.assertEqual(prefs_a[0].preference_text, "Pref for A")
        self.assertEqual(prefs_b[0].preference_text, "Pref for B")

    def test_preference_soft_delete(self):
        """偏好软删除后不影响其他偏好"""
        user, _ = self.store.get_or_create_user("del_user", "Del User")
        p1 = self.store.add_preference(user.id, "Keep this", "general")
        p2 = self.store.add_preference(user.id, "Delete this", "general")

        self.store.delete_preference(p2.id)

        prefs = self.store.get_preferences(user.id)
        active_prefs = [p for p in prefs if p.is_active]
        self.assertEqual(len(active_prefs), 1)
        self.assertEqual(active_prefs[0].preference_text, "Keep this")

    def test_dki_bridge_config_with_real_db(self):
        """验证 dki_bridge 生成的配置可指向真实数据库"""
        from dki.experiment.dki_bridge import build_experiment_adapter_config

        cfg = ExperimentDBConfig(sqlite_path=self.db_path)
        adapter_config = build_experiment_adapter_config(cfg)

        # 配置应指向正确的数据库
        self.assertEqual(adapter_config["database"]["database"], self.db_path)
        self.assertEqual(adapter_config["database"]["type"], "sqlite")

        # 表映射应匹配 store 使用的表
        self.assertEqual(adapter_config["preferences"]["table"], "demo_preferences")
        self.assertEqual(adapter_config["messages"]["table"], "demo_messages")


if __name__ == '__main__':
    unittest.main()
