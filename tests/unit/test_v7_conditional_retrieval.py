"""
v7.0 条件语义检索 单元测试

测试覆盖:
1. VectorIndexConfig 数据类及 from_dict 解析
2. VectorIndexCoreConfig / VectorIndexEmbeddingConfig / VectorIndexRetrievalConfig / VectorIndexMetadataConfig
3. VectorSearchConfig.has_vector_capability 属性 (核心判断逻辑)
4. ConfigDrivenAdapterConfig.from_dict 解析 vector_index_config
5. ConfigDrivenAdapter._init_vector_handler BM25-only 降级
6. ConfigDrivenAdapter._bm25_score BM25 评分算法
7. ConfigDrivenAdapter._search_with_bm25_only BM25 检索
8. ConfigDrivenAdapter.search_relevant_history 路由逻辑
9. ConfigDrivenAdapter._create_embedding_service 路由
10. DKIPlugin._detect_retrieval_mode 检测逻辑
11. InjectionMetadata.retrieval_mode 字段

Author: AGI Demo Project
"""

import pytest
import math
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Test 1: VectorIndexCoreConfig 数据类
# ============================================================================

class TestVectorIndexCoreConfig:
    """测试 VectorIndexCoreConfig 数据类"""

    def test_default_values(self):
        """测试默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexCoreConfig
        config = VectorIndexCoreConfig()
        assert config.index_type == "HNSW"
        assert config.dimension == 768
        assert config.vector_data_type == "float32"
        assert config.similarity_metric == "cosine"

    def test_custom_values(self):
        """测试自定义值"""
        from dki.adapters.config_driven_adapter import VectorIndexCoreConfig
        config = VectorIndexCoreConfig(
            index_type="FLAT",
            dimension=1536,
            vector_data_type="float16",
            similarity_metric="l2",
        )
        assert config.index_type == "FLAT"
        assert config.dimension == 1536
        assert config.vector_data_type == "float16"
        assert config.similarity_metric == "l2"


# ============================================================================
# Test 2: VectorIndexEmbeddingConfig 数据类
# ============================================================================

class TestVectorIndexEmbeddingConfig:
    """测试 VectorIndexEmbeddingConfig 数据类"""

    def test_default_values(self):
        """测试默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexEmbeddingConfig
        config = VectorIndexEmbeddingConfig()
        assert config.api_type == "local"
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.api_endpoint is None
        assert config.api_key is None
        assert config.normalization is True

    def test_openai_config(self):
        """测试 OpenAI 配置"""
        from dki.adapters.config_driven_adapter import VectorIndexEmbeddingConfig
        config = VectorIndexEmbeddingConfig(
            api_type="openai",
            model_name="text-embedding-ada-002",
            api_endpoint="https://api.openai.com/v1/embeddings",
            api_key="sk-test-key",
            normalization=True,
        )
        assert config.api_type == "openai"
        assert config.model_name == "text-embedding-ada-002"
        assert config.api_endpoint == "https://api.openai.com/v1/embeddings"
        assert config.api_key == "sk-test-key"

    def test_local_config(self):
        """测试本地模型配置"""
        from dki.adapters.config_driven_adapter import VectorIndexEmbeddingConfig
        config = VectorIndexEmbeddingConfig(
            api_type="local",
            model_name="bge-large-zh",
            api_key="local",
        )
        assert config.api_type == "local"
        assert config.api_key == "local"


# ============================================================================
# Test 3: VectorIndexRetrievalConfig 数据类
# ============================================================================

class TestVectorIndexRetrievalConfig:
    """测试 VectorIndexRetrievalConfig 数据类"""

    def test_default_values(self):
        """测试默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexRetrievalConfig
        config = VectorIndexRetrievalConfig()
        assert config.top_k == 10
        assert config.index_file_path is None

    def test_custom_values(self):
        """测试自定义值"""
        from dki.adapters.config_driven_adapter import VectorIndexRetrievalConfig
        config = VectorIndexRetrievalConfig(
            top_k=20,
            index_file_path="./data/vector.index",
        )
        assert config.top_k == 20
        assert config.index_file_path == "./data/vector.index"


# ============================================================================
# Test 4: VectorIndexMetadataConfig 数据类
# ============================================================================

class TestVectorIndexMetadataConfig:
    """测试 VectorIndexMetadataConfig 数据类"""

    def test_default_values(self):
        """测试默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexMetadataConfig
        config = VectorIndexMetadataConfig()
        assert config.id_mapping_table == "vector_id_mapping"
        assert config.primary_key == "vector_id"

    def test_custom_values(self):
        """测试自定义值"""
        from dki.adapters.config_driven_adapter import VectorIndexMetadataConfig
        config = VectorIndexMetadataConfig(
            id_mapping_table="custom_mapping",
            primary_key="custom_id",
        )
        assert config.id_mapping_table == "custom_mapping"
        assert config.primary_key == "custom_id"


# ============================================================================
# Test 5: VectorIndexConfig 及 from_dict
# ============================================================================

class TestVectorIndexConfig:
    """测试 VectorIndexConfig 完整配置"""

    def test_default_values(self):
        """测试默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexConfig
        config = VectorIndexConfig()
        assert config.core.dimension == 768
        assert config.embedding.api_type == "local"
        assert config.retrieval.top_k == 10
        assert config.metadata.primary_key == "vector_id"

    def test_from_dict_full(self):
        """测试从完整字典创建"""
        from dki.adapters.config_driven_adapter import VectorIndexConfig
        data = {
            "core": {
                "index_type": "HNSW",
                "dimension": 1536,
                "vector_data_type": "float32",
                "similarity_metric": "cosine",
            },
            "embedding": {
                "api_type": "openai",
                "model_name": "text-embedding-ada-002",
                "api_endpoint": "https://api.openai.com/v1/embeddings",
                "api_key": "sk-test",
                "normalization": True,
            },
            "retrieval": {
                "top_k": 20,
                "index_file_path": "./index.bin",
            },
            "metadata": {
                "id_mapping_table": "my_mapping",
                "primary_key": "doc_id",
            },
        }
        config = VectorIndexConfig.from_dict(data)
        assert config.core.dimension == 1536
        assert config.core.index_type == "HNSW"
        assert config.embedding.api_type == "openai"
        assert config.embedding.model_name == "text-embedding-ada-002"
        assert config.embedding.api_key == "sk-test"
        assert config.retrieval.top_k == 20
        assert config.retrieval.index_file_path == "./index.bin"
        assert config.metadata.id_mapping_table == "my_mapping"
        assert config.metadata.primary_key == "doc_id"

    def test_from_dict_partial(self):
        """测试从部分字典创建 (仅 core + embedding)"""
        from dki.adapters.config_driven_adapter import VectorIndexConfig
        data = {
            "core": {"dimension": 384},
            "embedding": {"api_type": "local", "model_name": "bge-small-zh"},
        }
        config = VectorIndexConfig.from_dict(data)
        assert config.core.dimension == 384
        assert config.embedding.api_type == "local"
        assert config.embedding.model_name == "bge-small-zh"
        # 未指定的部分使用默认值
        assert config.retrieval.top_k == 10
        assert config.metadata.primary_key == "vector_id"

    def test_from_dict_empty(self):
        """测试空字典创建 (全部使用默认值)"""
        from dki.adapters.config_driven_adapter import VectorIndexConfig
        config = VectorIndexConfig.from_dict({})
        assert config.core.dimension == 768
        assert config.embedding.api_type == "local"

    def test_from_dict_with_defaults_override(self):
        """测试部分覆盖默认值"""
        from dki.adapters.config_driven_adapter import VectorIndexConfig
        data = {
            "core": {"dimension": 768, "similarity_metric": "ip"},
            "embedding": {"normalization": False},
        }
        config = VectorIndexConfig.from_dict(data)
        assert config.core.similarity_metric == "ip"
        assert config.embedding.normalization is False


# ============================================================================
# Test 6: VectorSearchConfig.has_vector_capability (核心判断)
# ============================================================================

class TestHasVectorCapability:
    """测试 VectorSearchConfig.has_vector_capability 属性"""

    def test_disabled_always_false(self):
        """disabled 时始终返回 False"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType, VectorIndexConfig,
        )
        config = VectorSearchConfig(
            enabled=False,
            vector_index_config=VectorIndexConfig(),
        )
        assert config.has_vector_capability is False

    def test_enabled_with_vector_index_config(self):
        """有 vector_index_config 时返回 True"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorIndexConfig,
        )
        config = VectorSearchConfig(
            enabled=True,
            vector_index_config=VectorIndexConfig(),
        )
        assert config.has_vector_capability is True

    def test_enabled_without_vector_index_config_dynamic(self):
        """无 vector_index_config 且 type=dynamic 时返回 False"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=None,
        )
        assert config.has_vector_capability is False

    def test_enabled_pgvector_with_embedding_field(self):
        """pgvector + embedding_field 时返回 True"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.PGVECTOR,
            embedding_field="embedding_vec",
            vector_index_config=None,
        )
        assert config.has_vector_capability is True

    def test_enabled_pgvector_without_embedding_field(self):
        """pgvector 但无 embedding_field 时返回 False"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.PGVECTOR,
            embedding_field=None,
            vector_index_config=None,
        )
        assert config.has_vector_capability is False

    def test_enabled_none_type(self):
        """type=NONE 时返回 False (无向量能力)"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.NONE,
            vector_index_config=None,
        )
        assert config.has_vector_capability is False

    def test_enabled_faiss_without_vic(self):
        """type=FAISS 但无 vector_index_config 时返回 False"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.FAISS,
            vector_index_config=None,
        )
        assert config.has_vector_capability is False

    def test_enabled_faiss_with_vic(self):
        """type=FAISS + vector_index_config 时返回 True"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType, VectorIndexConfig,
        )
        config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.FAISS,
            vector_index_config=VectorIndexConfig(),
        )
        assert config.has_vector_capability is True


# ============================================================================
# Test 7: ConfigDrivenAdapterConfig.from_dict 解析 vector_index_config
# ============================================================================

class TestConfigDrivenAdapterConfigVIC:
    """测试 ConfigDrivenAdapterConfig 解析 vector_index_config"""

    def test_from_dict_with_vector_index_config(self):
        """从字典解析含 vector_index_config 的完整配置"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapterConfig, VectorSearchType,
        )
        data = {
            "database": {"type": "sqlite", "database": ":memory:"},
            "vector_search": {
                "enabled": True,
                "type": "dynamic",
                "vector_index_config": {
                    "core": {"dimension": 768, "index_type": "HNSW"},
                    "embedding": {
                        "api_type": "openai",
                        "model_name": "text-embedding-ada-002",
                        "api_endpoint": "https://api.openai.com/v1/embeddings",
                        "api_key": "sk-test",
                    },
                    "retrieval": {"top_k": 15},
                },
            },
        }
        config = ConfigDrivenAdapterConfig.from_dict(data)
        assert config.vector_search.enabled is True
        assert config.vector_search.type == VectorSearchType.DYNAMIC
        assert config.vector_search.vector_index_config is not None
        assert config.vector_search.vector_index_config.core.dimension == 768
        assert config.vector_search.vector_index_config.embedding.api_type == "openai"
        assert config.vector_search.vector_index_config.retrieval.top_k == 15
        assert config.vector_search.has_vector_capability is True

    def test_from_dict_without_vector_index_config(self):
        """从字典解析不含 vector_index_config 的配置"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapterConfig, VectorSearchType,
        )
        data = {
            "database": {"type": "sqlite", "database": ":memory:"},
            "vector_search": {
                "enabled": True,
                "type": "dynamic",
                "embedding_dim": 768,
                "dynamic": {"strategy": "hybrid"},
            },
        }
        config = ConfigDrivenAdapterConfig.from_dict(data)
        assert config.vector_search.vector_index_config is None
        assert config.vector_search.has_vector_capability is False

    def test_from_dict_pgvector_with_embedding_field_no_vic(self):
        """pgvector + embedding_field 但无 vector_index_config"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapterConfig, VectorSearchType,
        )
        data = {
            "database": {"type": "postgresql", "host": "localhost"},
            "vector_search": {
                "type": "pgvector",
                "embedding_field": "embedding_vec",
                "embedding_dim": 1536,
            },
        }
        config = ConfigDrivenAdapterConfig.from_dict(data)
        assert config.vector_search.type == VectorSearchType.PGVECTOR
        assert config.vector_search.embedding_field == "embedding_vec"
        assert config.vector_search.has_vector_capability is True
        assert config.vector_search.vector_index_config is None


# ============================================================================
# Test 8: BM25 评分算法
# ============================================================================

class TestBM25Score:
    """测试 ConfigDrivenAdapter._bm25_score"""

    def _make_adapter(self):
        """创建一个最小化的 adapter 实例用于测试 _bm25_score"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        config = ConfigDrivenAdapterConfig()
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        return adapter

    def _make_message(self, content: str):
        """创建一个 mock ChatMessage"""
        msg = MagicMock()
        msg.content = content
        return msg

    def test_bm25_score_basic(self):
        """基本 BM25 评分"""
        adapter = self._make_adapter()
        messages = [
            self._make_message("今天去吃火锅"),
            self._make_message("明天去看电影"),
            self._make_message("火锅很好吃推荐一下"),
        ]
        results = adapter._bm25_score("火锅推荐", messages)
        assert len(results) == 3
        # 包含 "火锅" 的消息分数应更高
        scores = {msg.content: score for msg, score in results}
        assert scores["今天去吃火锅"] > scores["明天去看电影"]
        assert scores["火锅很好吃推荐一下"] > scores["明天去看电影"]

    def test_bm25_score_empty_query(self):
        """空查询"""
        adapter = self._make_adapter()
        messages = [self._make_message("测试消息")]
        results = adapter._bm25_score("", messages)
        assert len(results) == 1
        assert results[0][1] == 0.0

    def test_bm25_score_no_match(self):
        """无匹配"""
        adapter = self._make_adapter()
        messages = [
            self._make_message("今天天气真好"),
            self._make_message("明天下雨"),
        ]
        results = adapter._bm25_score("xyzabc", messages)
        assert all(score == 0.0 for _, score in results)

    def test_bm25_score_english(self):
        """英文 BM25 评分"""
        adapter = self._make_adapter()
        messages = [
            self._make_message("I love python programming"),
            self._make_message("The weather is nice today"),
            self._make_message("Python is a great language for programming"),
        ]
        results = adapter._bm25_score("python programming", messages)
        scores = {msg.content: score for msg, score in results}
        # 包含 python 和 programming 的消息分数应更高
        assert scores["I love python programming"] > scores["The weather is nice today"]
        assert scores["Python is a great language for programming"] > scores["The weather is nice today"]

    def test_bm25_score_single_message(self):
        """单条消息"""
        adapter = self._make_adapter()
        messages = [self._make_message("推荐一家好餐厅")]
        results = adapter._bm25_score("餐厅", messages)
        assert len(results) == 1
        assert results[0][1] > 0.0

    def test_bm25_score_custom_params(self):
        """自定义 k1 和 b 参数"""
        adapter = self._make_adapter()
        messages = [
            self._make_message("火锅火锅火锅"),
            self._make_message("火锅"),
        ]
        # k1=0 时，词频不影响分数
        results_k0 = adapter._bm25_score("火锅", messages, k1=0.0, b=0.0)
        # k1 较高时，词频影响更大
        results_k2 = adapter._bm25_score("火锅", messages, k1=2.0, b=0.75)
        # 两种参数下都应该有结果
        assert len(results_k0) == 2
        assert len(results_k2) == 2


# ============================================================================
# Test 9: _init_vector_handler BM25-only 降级
# ============================================================================

class TestInitVectorHandler:
    """测试 _init_vector_handler 的 BM25-only 降级逻辑"""

    @pytest.mark.asyncio
    async def test_disabled_vector_search(self):
        """vector_search.enabled=False 时不初始化"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig, VectorSearchConfig,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = VectorSearchConfig(enabled=False)

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._vector_handler = None
        adapter._embedding_service = None
        adapter._bm25_only_mode = False

        await adapter._init_vector_handler()
        assert adapter._bm25_only_mode is False
        assert adapter._vector_handler is None
        assert adapter._embedding_service is None

    @pytest.mark.asyncio
    async def test_no_vector_capability_dynamic(self):
        """无 vector_index_config + type=DYNAMIC → BM25-only"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=None,
        )

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._vector_handler = None
        adapter._embedding_service = None
        adapter._bm25_only_mode = False

        await adapter._init_vector_handler()
        assert adapter._bm25_only_mode is True
        assert adapter._embedding_service is None

    @pytest.mark.asyncio
    async def test_no_vector_capability_none_type(self):
        """无 vector_index_config + type=NONE → 不设置 BM25-only"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.NONE,
            vector_index_config=None,
        )

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._vector_handler = None
        adapter._embedding_service = None
        adapter._bm25_only_mode = False

        await adapter._init_vector_handler()
        # type=NONE 不会设置 bm25_only_mode (走 keyword 路径)
        assert adapter._bm25_only_mode is False

    @pytest.mark.asyncio
    async def test_embedding_service_failure_fallback(self):
        """Embedding 服务创建失败时回退到 BM25-only"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType, VectorIndexConfig,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=VectorIndexConfig(),
        )

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._vector_handler = None
        adapter._embedding_service = None
        adapter._bm25_only_mode = False

        # Mock _create_embedding_service 返回 None (创建失败)
        async def mock_create_embedding():
            return None

        adapter._create_embedding_service = mock_create_embedding

        await adapter._init_vector_handler()
        assert adapter._bm25_only_mode is True
        assert adapter._vector_handler is None


# ============================================================================
# Test 10: _create_embedding_service 路由逻辑
# ============================================================================

class TestCreateEmbeddingService:
    """测试 _create_embedding_service 的路由逻辑"""

    def _make_adapter(self, vs_config):
        """创建 adapter 实例"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = vs_config
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        return adapter

    @pytest.mark.asyncio
    async def test_route_to_local_with_vic(self):
        """vector_index_config + api_type=local → LocalEmbeddingService"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorIndexConfig,
            VectorIndexEmbeddingConfig, VectorIndexCoreConfig,
        )
        vic = VectorIndexConfig(
            core=VectorIndexCoreConfig(dimension=384),
            embedding=VectorIndexEmbeddingConfig(
                api_type="local",
                model_name="all-MiniLM-L6-v2",
            ),
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            vector_index_config=vic,
        )
        adapter = self._make_adapter(vs_config)

        # Mock _create_local_embedding_service
        mock_local = MagicMock(return_value="local_service")
        adapter._create_local_embedding_service = mock_local

        result = await adapter._create_embedding_service()
        mock_local.assert_called_once_with(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=True,
        )
        assert result == "local_service"

    @pytest.mark.asyncio
    async def test_route_to_remote_with_vic(self):
        """vector_index_config + api_type=openai → RemoteEmbeddingService"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorIndexConfig,
            VectorIndexEmbeddingConfig, VectorIndexCoreConfig,
        )
        vic = VectorIndexConfig(
            core=VectorIndexCoreConfig(dimension=1536),
            embedding=VectorIndexEmbeddingConfig(
                api_type="openai",
                model_name="text-embedding-ada-002",
                api_endpoint="https://api.openai.com/v1/embeddings",
                api_key="sk-test",
                normalization=True,
            ),
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            vector_index_config=vic,
        )
        adapter = self._make_adapter(vs_config)

        mock_remote = MagicMock(return_value="remote_service")
        adapter._create_remote_embedding_service = mock_remote

        result = await adapter._create_embedding_service()
        mock_remote.assert_called_once_with(
            api_type="openai",
            model_name="text-embedding-ada-002",
            api_endpoint="https://api.openai.com/v1/embeddings",
            api_key="sk-test",
            dimension=1536,
            normalize=True,
        )
        assert result == "remote_service"

    @pytest.mark.asyncio
    async def test_route_to_local_with_api_key_local(self):
        """api_key='local' 时路由到本地"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorIndexConfig,
            VectorIndexEmbeddingConfig, VectorIndexCoreConfig,
        )
        vic = VectorIndexConfig(
            core=VectorIndexCoreConfig(dimension=768),
            embedding=VectorIndexEmbeddingConfig(
                api_type="custom",
                model_name="my-model",
                api_key="local",
            ),
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            vector_index_config=vic,
        )
        adapter = self._make_adapter(vs_config)

        mock_local = MagicMock(return_value="local_service")
        adapter._create_local_embedding_service = mock_local

        result = await adapter._create_embedding_service()
        mock_local.assert_called_once()
        assert result == "local_service"

    @pytest.mark.asyncio
    async def test_fallback_to_legacy_config(self):
        """无 vector_index_config 时回退到旧配置"""
        from dki.adapters.config_driven_adapter import VectorSearchConfig
        vs_config = VectorSearchConfig(
            enabled=True,
            vector_index_config=None,
            embedding_api_url="https://custom.api/embed",
            embedding_api_key="key-123",
            embedding_model="custom-model",
            embedding_dim=512,
        )
        adapter = self._make_adapter(vs_config)

        mock_remote = MagicMock(return_value="legacy_remote")
        adapter._create_remote_embedding_service = mock_remote

        result = await adapter._create_embedding_service()
        mock_remote.assert_called_once_with(
            api_type="openai",
            model_name="custom-model",
            api_endpoint="https://custom.api/embed",
            api_key="key-123",
            dimension=512,
            normalize=True,
        )
        assert result == "legacy_remote"

    @pytest.mark.asyncio
    async def test_fallback_to_local_model(self):
        """无 vector_index_config 且无 API 配置时回退到本地模型"""
        from dki.adapters.config_driven_adapter import VectorSearchConfig
        vs_config = VectorSearchConfig(
            enabled=True,
            vector_index_config=None,
            embedding_api_url=None,
            embedding_api_key=None,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        adapter = self._make_adapter(vs_config)

        mock_local = MagicMock(return_value="default_local")
        adapter._create_local_embedding_service = mock_local

        result = await adapter._create_embedding_service()
        mock_local.assert_called_once_with(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            normalize=True,
        )
        assert result == "default_local"


# ============================================================================
# Test 11: search_relevant_history 路由逻辑
# ============================================================================

class TestSearchRelevantHistoryRouting:
    """测试 search_relevant_history 的 v7.0 路由逻辑"""

    def _make_adapter(self, vs_config, bm25_only=False):
        """创建 adapter 实例"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig, TableMapping,
        )
        config = ConfigDrivenAdapterConfig()
        config.vector_search = vs_config
        config.messages = TableMapping(
            table="messages",
            fields={"message_id": "id", "user_id": "user_id", "content": "content"},
        )

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._tables = {"messages": MagicMock()}
        adapter._bm25_only_mode = bm25_only
        adapter._vector_handler = None
        adapter._embedding_service = None
        return adapter

    @pytest.mark.asyncio
    async def test_no_messages_config(self):
        """无消息配置时返回空"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig, VectorSearchConfig,
        )
        config = ConfigDrivenAdapterConfig()
        config.messages = None

        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = config
        adapter._tables = {}
        adapter._bm25_only_mode = False

        result = await adapter.search_relevant_history("user1", "test")
        assert result == []

    @pytest.mark.asyncio
    async def test_disabled_routes_to_keywords(self):
        """disabled 时路由到 _search_with_keywords"""
        from dki.adapters.config_driven_adapter import VectorSearchConfig
        vs_config = VectorSearchConfig(enabled=False)
        adapter = self._make_adapter(vs_config)

        mock_keywords = AsyncMock(return_value=["kw_result"])
        adapter._search_with_keywords = mock_keywords

        result = await adapter.search_relevant_history("user1", "test", limit=5)
        mock_keywords.assert_called_once_with("user1", "test", 5, None)
        assert result == ["kw_result"]

    @pytest.mark.asyncio
    async def test_bm25_only_mode_routes_to_bm25(self):
        """BM25-only 模式路由到 _search_with_bm25_only"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=None,
        )
        adapter = self._make_adapter(vs_config, bm25_only=True)

        mock_bm25 = AsyncMock(return_value=["bm25_result"])
        adapter._search_with_bm25_only = mock_bm25

        result = await adapter.search_relevant_history("user1", "test", limit=5)
        mock_bm25.assert_called_once_with("user1", "test", 5, None)
        assert result == ["bm25_result"]

    @pytest.mark.asyncio
    async def test_has_capability_pgvector_routes(self):
        """有向量能力 + PGVECTOR → _search_with_pgvector"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType,
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.PGVECTOR,
            embedding_field="embedding",
            vector_index_config=None,
        )
        adapter = self._make_adapter(vs_config)

        mock_pgvector = AsyncMock(return_value=["pgvector_result"])
        adapter._search_with_pgvector = mock_pgvector

        result = await adapter.search_relevant_history("user1", "test", limit=5)
        mock_pgvector.assert_called_once_with("user1", "test", 5, None)
        assert result == ["pgvector_result"]

    @pytest.mark.asyncio
    async def test_has_capability_dynamic_with_handler(self):
        """有向量能力 + DYNAMIC + vector_handler → _search_with_dynamic_handler"""
        from dki.adapters.config_driven_adapter import (
            VectorSearchConfig, VectorSearchType, VectorIndexConfig,
        )
        vs_config = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=VectorIndexConfig(),
        )
        adapter = self._make_adapter(vs_config)
        adapter._vector_handler = MagicMock()  # 有 handler

        mock_dynamic = AsyncMock(return_value=["dynamic_result"])
        adapter._search_with_dynamic_handler = mock_dynamic

        result = await adapter.search_relevant_history("user1", "test", limit=5)
        mock_dynamic.assert_called_once_with("user1", "test", 5, None)
        assert result == ["dynamic_result"]


# ============================================================================
# Test 12: _search_with_bm25_only 方法
# ============================================================================

class TestSearchWithBM25Only:
    """测试 _search_with_bm25_only"""

    @pytest.mark.asyncio
    async def test_bm25_only_returns_sorted(self):
        """BM25-only 返回按分数排序的结果"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        msg1 = MagicMock(content="推荐一家火锅店")
        msg2 = MagicMock(content="今天天气不错")
        msg3 = MagicMock(content="火锅店的牛肉很好吃")

        adapter._get_user_messages = AsyncMock(return_value=[msg1, msg2, msg3])

        # Mock _bm25_score 返回带分数的结果
        adapter._bm25_score = MagicMock(return_value=[
            (msg1, 3.5),
            (msg2, 0.1),
            (msg3, 5.2),
        ])

        result = await adapter._search_with_bm25_only("user1", "火锅", 2, None)
        assert len(result) == 2
        assert result[0] == msg3  # 最高分
        assert result[1] == msg1  # 次高分

    @pytest.mark.asyncio
    async def test_bm25_only_empty_messages(self):
        """无消息时返回空"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter._get_user_messages = AsyncMock(return_value=[])

        result = await adapter._search_with_bm25_only("user1", "test", 5, None)
        assert result == []

    @pytest.mark.asyncio
    async def test_bm25_only_respects_limit(self):
        """BM25-only 尊重 limit 参数"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        messages = [MagicMock(content=f"msg_{i}") for i in range(10)]
        adapter._get_user_messages = AsyncMock(return_value=messages)
        adapter._bm25_score = MagicMock(
            return_value=[(msg, float(i)) for i, msg in enumerate(messages)]
        )

        result = await adapter._search_with_bm25_only("user1", "test", 3, None)
        assert len(result) == 3


# ============================================================================
# Test 13: DKIPlugin._detect_retrieval_mode
# ============================================================================

class TestDetectRetrievalMode:
    """测试 DKIPlugin._detect_retrieval_mode"""

    def _make_plugin(self, adapter):
        """创建一个最小化的 DKIPlugin 实例"""
        from dki.core.dki_plugin import DKIPlugin
        plugin = DKIPlugin.__new__(DKIPlugin)
        plugin.data_adapter = adapter
        return plugin

    def test_non_config_driven_adapter(self):
        """非 ConfigDrivenAdapter 返回 unknown"""
        adapter = MagicMock()
        adapter.__class__.__name__ = "ExampleAdapter"
        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "unknown"

    def test_config_driven_disabled(self):
        """vector_search.enabled=False → keyword"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter.adapter_config.vector_search = VectorSearchConfig(enabled=False)
        adapter._bm25_only_mode = False

        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "keyword"

    def test_config_driven_with_vic(self):
        """有 vector_index_config → bm25_embedding"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType, VectorIndexConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter.adapter_config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=VectorIndexConfig(),
        )
        adapter._bm25_only_mode = False

        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "bm25_embedding"

    def test_config_driven_pgvector(self):
        """pgvector + embedding_field → pgvector"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter.adapter_config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.PGVECTOR,
            embedding_field="embedding",
        )
        adapter._bm25_only_mode = False

        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "pgvector"

    def test_config_driven_bm25_only_mode(self):
        """_bm25_only_mode=True → bm25_only"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter.adapter_config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=None,
        )
        adapter._bm25_only_mode = True

        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "bm25_only"

    def test_config_driven_no_vic_no_flag(self):
        """无 vector_index_config 且 _bm25_only_mode=False → bm25_only"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
            VectorSearchConfig, VectorSearchType,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()
        adapter.adapter_config.vector_search = VectorSearchConfig(
            enabled=True,
            type=VectorSearchType.DYNAMIC,
            vector_index_config=None,
        )
        adapter._bm25_only_mode = False

        plugin = self._make_plugin(adapter)
        # has_vector_capability=False, _bm25_only_mode=False → 仍返回 bm25_only
        assert plugin._detect_retrieval_mode() == "bm25_only"

    def test_exception_returns_unknown(self):
        """异常时返回 unknown"""
        adapter = MagicMock()
        # 让 adapter_config 抛出异常
        type(adapter).adapter_config = PropertyMock(side_effect=AttributeError)

        plugin = self._make_plugin(adapter)
        assert plugin._detect_retrieval_mode() == "unknown"


# ============================================================================
# Test 14: InjectionMetadata.retrieval_mode 字段
# ============================================================================

class TestInjectionMetadataRetrievalMode:
    """测试 InjectionMetadata.retrieval_mode 字段"""

    def test_default_value(self):
        """默认值为 unknown"""
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata()
        assert metadata.retrieval_mode == "unknown"

    def test_set_bm25_only(self):
        """设置为 bm25_only"""
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata(retrieval_mode="bm25_only")
        assert metadata.retrieval_mode == "bm25_only"

    def test_set_bm25_embedding(self):
        """设置为 bm25_embedding"""
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata(retrieval_mode="bm25_embedding")
        assert metadata.retrieval_mode == "bm25_embedding"

    def test_to_dict_includes_retrieval_mode(self):
        """to_dict 包含 retrieval_mode"""
        from dki.core.dki_plugin import InjectionMetadata
        metadata = InjectionMetadata(retrieval_mode="pgvector")
        d = metadata.to_dict()
        assert "retrieval_mode" in d
        assert d["retrieval_mode"] == "pgvector"

    def test_to_dict_all_modes(self):
        """所有检索模式都能正确序列化"""
        from dki.core.dki_plugin import InjectionMetadata
        for mode in ["bm25_only", "bm25_embedding", "keyword", "pgvector", "unknown"]:
            metadata = InjectionMetadata(retrieval_mode=mode)
            d = metadata.to_dict()
            assert d["retrieval_mode"] == mode


# ============================================================================
# Test 15: RemoteEmbeddingService 创建
# ============================================================================

class TestRemoteEmbeddingService:
    """测试 _create_remote_embedding_service"""

    def test_create_openai_service(self):
        """创建 OpenAI Embedding 服务"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        service = adapter._create_remote_embedding_service(
            api_type="openai",
            model_name="text-embedding-ada-002",
            api_endpoint="https://api.openai.com/v1/embeddings",
            api_key="sk-test",
            dimension=1536,
            normalize=True,
        )
        assert service is not None
        assert service.model == "text-embedding-ada-002"
        assert service.api_url == "https://api.openai.com/v1/embeddings"
        assert service.api_key == "sk-test"
        assert service.dim == 1536
        assert service.normalize is True

    def test_create_with_default_endpoint(self):
        """使用默认 endpoint"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        service = adapter._create_remote_embedding_service(
            api_type="openai",
            model_name="text-embedding-ada-002",
            api_endpoint=None,  # 使用默认
            api_key="sk-test",
            dimension=1536,
            normalize=True,
        )
        assert "openai.com" in service.api_url

    def test_create_aliyun_service(self):
        """创建阿里云 Embedding 服务"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        service = adapter._create_remote_embedding_service(
            api_type="aliyun",
            model_name="text-embedding-v2",
            api_endpoint=None,
            api_key="aliyun-key",
            dimension=768,
            normalize=True,
        )
        assert "aliyun" in service.api_url.lower() or "dashscope" in service.api_url.lower()

    def test_normalize_vector(self):
        """测试向量归一化"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        service = adapter._create_remote_embedding_service(
            api_type="openai",
            model_name="test",
            api_endpoint="http://localhost",
            api_key="key",
            dimension=3,
            normalize=True,
        )
        vec = [3.0, 4.0, 0.0]
        normed = service._normalize_vector(vec)
        # 3/5, 4/5, 0
        assert abs(normed[0] - 0.6) < 1e-6
        assert abs(normed[1] - 0.8) < 1e-6
        assert abs(normed[2] - 0.0) < 1e-6

    def test_normalize_zero_vector(self):
        """零向量归一化不报错"""
        from dki.adapters.config_driven_adapter import (
            ConfigDrivenAdapter, ConfigDrivenAdapterConfig,
        )
        adapter = ConfigDrivenAdapter.__new__(ConfigDrivenAdapter)
        adapter.adapter_config = ConfigDrivenAdapterConfig()

        service = adapter._create_remote_embedding_service(
            api_type="openai",
            model_name="test",
            api_endpoint="http://localhost",
            api_key="key",
            dimension=3,
            normalize=True,
        )
        vec = [0.0, 0.0, 0.0]
        normed = service._normalize_vector(vec)
        assert normed == [0.0, 0.0, 0.0]


# ============================================================================
# Test 16: DKIPluginResponse 包含 retrieval_mode
# ============================================================================

class TestDKIPluginResponseRetrievalMode:
    """测试 DKIPluginResponse 序列化包含 retrieval_mode"""

    def test_response_to_dict(self):
        """响应序列化包含 retrieval_mode"""
        from dki.core.dki_plugin import DKIPluginResponse, InjectionMetadata
        metadata = InjectionMetadata(retrieval_mode="bm25_only")
        response = DKIPluginResponse(
            text="测试响应",
            input_tokens=10,
            output_tokens=20,
            metadata=metadata,
        )
        d = response.to_dict()
        assert d["metadata"]["retrieval_mode"] == "bm25_only"
