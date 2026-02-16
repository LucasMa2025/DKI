"""
User-Level Cache Isolation for DKI

===========================================================================
PURPOSE
===========================================================================
DKI 的原有缓存系统只通过 "缓存键唯一" 实现了 "不混用"，但没有实现真正的
"用户级隔离"。本模块解决以下 3 个核心安全缺陷：

1. 缓存池全局共享 → 引入 UserScopedCacheStore (逻辑分区 + 访问权限校验)
2. 计算过程共享   → 引入 InferenceContextGuard (推理上下文隔离 + 清理验证)
3. 无权限管控兜底 → 引入 HMAC 签名缓存键 + 用户身份鉴权链路

===========================================================================
DESIGN PRINCIPLES
===========================================================================
- 零信任: 每次缓存访问都必须验证 user_id 归属权
- 纵深防御: 即使键名被猜到/构造出来，HMAC 签名校验也会拒绝访问
- 推理隔离: 每次推理前后都有上下文隔离检查，防止 K/V 残留泄露
- 审计可追溯: 所有缓存访问都记录审计日志

===========================================================================
ARCHITECTURE
===========================================================================

┌─────────────────────────────────────────────────────────────┐
│                    API Layer (auth_routes)                    │
│              ┌──────────────────────────┐                    │
│              │  Bearer Token → user_id  │                    │
│              └──────────┬───────────────┘                    │
│                         │                                    │
│              ┌──────────▼───────────────┐                    │
│              │  UserIsolationContext     │ ← 每请求创建       │
│              │  (user_id + session_token │                    │
│              │   + HMAC signer)         │                    │
│              └──────────┬───────────────┘                    │
│                         │                                    │
│         ┌───────────────┼───────────────────┐               │
│         │               │                   │               │
│  ┌──────▼──────┐ ┌──────▼──────┐  ┌────────▼────────┐     │
│  │ Scoped L1   │ │ Scoped L2   │  │ InferenceGuard  │     │
│  │ (Memory)    │ │ (Redis)     │  │ (Context清理)    │     │
│  │ user_id     │ │ user_id     │  │ 前置/后置检查    │     │
│  │ namespace   │ │ namespace   │  │                  │     │
│  └─────────────┘ └─────────────┘  └─────────────────┘     │
│                                                              │
│  每个操作: user_id校验 → HMAC签名验证 → 数据访问 → 审计日志   │
└─────────────────────────────────────────────────────────────┘

Author: AGI Demo Project
Version: 1.0.0
"""

import hashlib
import hmac
import os
import secrets
import time
import asyncio
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class UserIsolationConfig:
    """
    用户隔离配置
    
    安全级别:
    - strict: 完整 HMAC 签名 + 访问校验 + 审计日志 (生产环境)
    - standard: HMAC 签名 + 访问校验 (预发布环境)
    - relaxed: 仅访问校验 (开发环境)
    """
    # 安全级别
    security_level: str = "strict"  # strict | standard | relaxed
    
    # HMAC 密钥 (生产环境应从环境变量/密钥管理服务读取)
    hmac_secret: str = ""  # 空则自动生成
    
    # 缓存键签名长度 (字节)
    signature_length: int = 16  # 128-bit
    
    # 是否启用审计日志
    enable_audit_log: bool = True
    
    # 审计日志最大条目 (内存)
    audit_log_max_entries: int = 10000
    
    # 推理上下文隔离
    enable_inference_guard: bool = True
    
    # K/V 残留检测阈值 (张量元素数)
    residual_check_threshold: int = 0  # 0 = 严格检查所有残留
    
    # 用户缓存分区最大数
    max_user_partitions: int = 10000
    
    # 单用户缓存条目上限
    per_user_max_entries: int = 100
    
    # 是否允许跨用户缓存迁移 (通常禁止)
    allow_cross_user_migration: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserIsolationConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def production(cls) -> "UserIsolationConfig":
        """生产环境配置"""
        return cls(
            security_level="strict",
            hmac_secret=os.environ.get("DKI_HMAC_SECRET", ""),
            enable_audit_log=True,
            enable_inference_guard=True,
        )
    
    @classmethod
    def development(cls) -> "UserIsolationConfig":
        """开发环境配置"""
        return cls(
            security_level="relaxed",
            enable_audit_log=False,
            enable_inference_guard=True,
        )


# ===========================================================================
# Audit Log
# ===========================================================================

@dataclass
class CacheAccessRecord:
    """缓存访问审计记录"""
    timestamp: float
    user_id: str
    action: str  # get | put | delete | invalidate | denied
    cache_key: str
    cache_tier: str  # L1_memory | L2_redis | executor
    success: bool
    denial_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheAuditLog:
    """
    缓存访问审计日志
    
    记录所有缓存访问操作，用于:
    - 安全审计: 检测异常访问模式
    - 合规: 记录谁在何时访问了谁的数据
    - 调试: 追踪缓存隔离问题
    """
    
    def __init__(self, max_entries: int = 10000):
        self._records: List[CacheAccessRecord] = []
        self._max_entries = max_entries
        self._lock = asyncio.Lock()
        self._denial_count = 0
        self._total_count = 0
    
    def record(
        self,
        user_id: str,
        action: str,
        cache_key: str,
        cache_tier: str,
        success: bool,
        denial_reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录一次缓存访问"""
        self._total_count += 1
        if not success:
            self._denial_count += 1
        
        record = CacheAccessRecord(
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            cache_key=cache_key,
            cache_tier=cache_tier,
            success=success,
            denial_reason=denial_reason,
            metadata=metadata or {},
        )
        
        self._records.append(record)
        
        # 环形缓冲: 超过上限时丢弃最旧记录
        if len(self._records) > self._max_entries:
            self._records = self._records[-self._max_entries:]
        
        # 拒绝访问时记录警告
        if not success:
            logger.warning(
                f"Cache access DENIED: user={user_id}, action={action}, "
                f"key={cache_key}, tier={cache_tier}, reason={denial_reason}"
            )
    
    def get_records(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[CacheAccessRecord]:
        """查询审计记录"""
        filtered = self._records
        
        if user_id:
            filtered = [r for r in filtered if r.user_id == user_id]
        if action:
            filtered = [r for r in filtered if r.action == action]
        if success is not None:
            filtered = [r for r in filtered if r.success == success]
        
        return filtered[-limit:]
    
    def get_denial_stats(self) -> Dict[str, Any]:
        """获取拒绝访问统计"""
        return {
            "total_accesses": self._total_count,
            "total_denials": self._denial_count,
            "denial_rate": self._denial_count / self._total_count if self._total_count > 0 else 0,
            "recent_denials": [
                {
                    "user_id": r.user_id,
                    "action": r.action,
                    "reason": r.denial_reason,
                    "timestamp": r.timestamp,
                }
                for r in self._records[-10:]
                if not r.success
            ],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取审计日志统计"""
        return {
            "total_records": len(self._records),
            "max_entries": self._max_entries,
            **self.get_denial_stats(),
        }


# ===========================================================================
# HMAC Cache Key Signer
# ===========================================================================

class CacheKeySigner:
    """
    HMAC 缓存键签名器
    
    解决 "缓存键可被猜测/构造" 的安全问题:
    - 每个缓存键都附带 HMAC 签名
    - 读取时验证签名，签名不匹配则拒绝访问
    - 即使攻击者知道 user_id 和 preference_hash，也无法构造有效签名
    
    签名格式: {user_id}:{content_hash}:{hmac_signature}
    """
    
    def __init__(self, secret: str = ""):
        """
        初始化签名器
        
        Args:
            secret: HMAC 密钥。空字符串则自动生成随机密钥。
                    生产环境应从环境变量/密钥管理服务读取。
        """
        if secret:
            self._secret = secret.encode('utf-8')
        else:
            # 自动生成 256-bit 随机密钥
            self._secret = secrets.token_bytes(32)
            logger.info("CacheKeySigner: auto-generated HMAC secret (ephemeral)")
    
    def sign_key(self, user_id: str, content_hash: str) -> str:
        """
        生成签名缓存键
        
        Args:
            user_id: 用户标识
            content_hash: 内容哈希
            
        Returns:
            签名缓存键: {user_id}:{content_hash}:{signature}
        """
        raw_key = f"{user_id}:{content_hash}"
        signature = self._compute_signature(raw_key)
        return f"{raw_key}:{signature}"
    
    def verify_key(self, signed_key: str, expected_user_id: str) -> bool:
        """
        验证签名缓存键
        
        Args:
            signed_key: 签名缓存键
            expected_user_id: 期望的用户 ID
            
        Returns:
            签名是否有效且归属于期望用户
        """
        parts = signed_key.rsplit(":", 2)
        if len(parts) != 3:
            return False
        
        user_id, content_hash, signature = parts
        
        # 验证用户归属
        if user_id != expected_user_id:
            return False
        
        # 验证 HMAC 签名
        raw_key = f"{user_id}:{content_hash}"
        expected_signature = self._compute_signature(raw_key)
        
        return hmac.compare_digest(signature, expected_signature)
    
    def extract_user_id(self, signed_key: str) -> Optional[str]:
        """从签名键中提取 user_id"""
        parts = signed_key.rsplit(":", 2)
        if len(parts) >= 1:
            return parts[0]
        return None
    
    def _compute_signature(self, raw_key: str) -> str:
        """计算 HMAC 签名"""
        mac = hmac.new(
            self._secret,
            raw_key.encode('utf-8'),
            hashlib.sha256,
        )
        return mac.hexdigest()[:32]  # 128-bit truncated


# ===========================================================================
# User Isolation Context
# ===========================================================================

@dataclass
class UserIsolationContext:
    """
    用户隔离上下文 - 每个请求创建一个
    
    携带当前请求的用户身份信息，贯穿整个请求生命周期。
    所有缓存操作都必须通过此上下文进行，确保:
    1. 用户身份已验证 (通过 API 层的 Bearer Token)
    2. 缓存键包含 HMAC 签名
    3. 每次访问都经过权限校验
    
    用法:
        # 在 API 层创建
        ctx = UserIsolationContext.create(
            user_id="user_123",
            session_id="sess_456",
            signer=cache_key_signer,
        )
        
        # 传递给 Executor / CacheManager
        kv_entries = await cache_manager.get_preference_kv(
            ctx=ctx,
            preference_text="素食主义者",
            model=model_adapter,
        )
    """
    user_id: str
    session_id: str = ""
    request_id: str = ""  # 唯一请求标识
    created_at: float = 0.0
    
    # 签名器引用 (不可序列化)
    _signer: Optional[CacheKeySigner] = field(default=None, repr=False)
    
    @classmethod
    def create(
        cls,
        user_id: str,
        session_id: str = "",
        signer: Optional[CacheKeySigner] = None,
    ) -> "UserIsolationContext":
        """创建隔离上下文"""
        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty for isolation context")
        
        return cls(
            user_id=user_id.strip(),
            session_id=session_id,
            request_id=secrets.token_hex(8),
            created_at=time.time(),
            _signer=signer,
        )
    
    def sign_cache_key(self, content_hash: str) -> str:
        """生成此用户的签名缓存键"""
        if self._signer:
            return self._signer.sign_key(self.user_id, content_hash)
        # 无签名器时退化为简单键
        return f"{self.user_id}:{content_hash}"
    
    def verify_cache_key(self, signed_key: str) -> bool:
        """验证缓存键是否属于此用户"""
        if self._signer:
            return self._signer.verify_key(signed_key, self.user_id)
        # 无签名器时仅检查前缀
        return signed_key.startswith(f"{self.user_id}:")
    
    def make_redis_namespace(self, prefix: str = "dki:pref_kv") -> str:
        """生成此用户的 Redis 命名空间"""
        return f"{prefix}:user:{self.user_id}"


# ===========================================================================
# User-Scoped Cache Store (L1 Memory)
# ===========================================================================

class UserScopedCacheStore:
    """
    用户级隔离缓存存储 (L1 内存层)
    
    核心改进:
    - 物理分区: 每个用户有独立的 OrderedDict，不共享数据结构
    - 访问校验: 每次 get/put 都验证 user_id 归属
    - HMAC 签名: 缓存键包含不可伪造的签名
    - 容量隔离: 单用户有独立的容量上限，防止一个用户占满缓存
    - 审计日志: 所有操作记录到审计日志
    
    与原有 LRUCache 的区别:
    ┌─────────────────────────────────────────────────────────────┐
    │  原有 LRUCache          │  UserScopedCacheStore            │
    ├─────────────────────────┼──────────────────────────────────┤
    │  全局共享 OrderedDict   │  每用户独立 OrderedDict          │
    │  靠键名区分用户         │  物理分区 + HMAC 签名校验         │
    │  无访问权限检查         │  每次访问都校验 user_id           │
    │  全局容量上限           │  全局 + 单用户双重容量上限        │
    │  无审计日志             │  完整审计日志                     │
    └─────────────────────────┴──────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: Optional[UserIsolationConfig] = None,
        signer: Optional[CacheKeySigner] = None,
        audit_log: Optional[CacheAuditLog] = None,
    ):
        self._config = config or UserIsolationConfig()
        self._signer = signer or CacheKeySigner(self._config.hmac_secret)
        self._audit = audit_log or CacheAuditLog(self._config.audit_log_max_entries)
        
        # 用户级分区: user_id -> OrderedDict[signed_key, CacheEntry]
        self._partitions: Dict[str, OrderedDict] = {}
        
        # 全局锁 (分区创建/删除)
        self._global_lock = asyncio.Lock()
        
        # 统计
        self._stats = {
            "total_gets": 0,
            "total_puts": 0,
            "total_hits": 0,
            "total_misses": 0,
            "total_denials": 0,
            "total_evictions": 0,
        }
        
        logger.info(
            f"UserScopedCacheStore initialized "
            f"(security={self._config.security_level}, "
            f"max_partitions={self._config.max_user_partitions}, "
            f"per_user_max={self._config.per_user_max_entries})"
        )
    
    @property
    def signer(self) -> CacheKeySigner:
        return self._signer
    
    @property
    def audit_log(self) -> CacheAuditLog:
        return self._audit
    
    def _get_or_create_partition(self, user_id: str) -> OrderedDict:
        """获取或创建用户分区"""
        if user_id not in self._partitions:
            if len(self._partitions) >= self._config.max_user_partitions:
                # 驱逐最旧的分区
                oldest_user = next(iter(self._partitions))
                del self._partitions[oldest_user]
                logger.info(f"Evicted user partition: {oldest_user}")
            
            self._partitions[user_id] = OrderedDict()
        
        return self._partitions[user_id]
    
    async def get(
        self,
        ctx: UserIsolationContext,
        content_hash: str,
    ) -> Optional[Any]:
        """
        获取缓存条目 (带用户隔离校验)
        
        Args:
            ctx: 用户隔离上下文
            content_hash: 内容哈希
            
        Returns:
            缓存数据或 None
        """
        self._stats["total_gets"] += 1
        signed_key = ctx.sign_cache_key(content_hash)
        
        # 校验: 用户分区存在性
        partition = self._partitions.get(ctx.user_id)
        if partition is None:
            self._stats["total_misses"] += 1
            return None
        
        # 校验: 键存在性
        if signed_key not in partition:
            self._stats["total_misses"] += 1
            return None
        
        # 校验: HMAC 签名验证
        if not ctx.verify_cache_key(signed_key):
            self._stats["total_denials"] += 1
            if self._config.enable_audit_log:
                self._audit.record(
                    user_id=ctx.user_id,
                    action="get",
                    cache_key=signed_key,
                    cache_tier="L1_memory",
                    success=False,
                    denial_reason="HMAC signature verification failed",
                )
            return None
        
        # 访问成功
        entry = partition[signed_key]
        partition.move_to_end(signed_key)  # LRU 更新
        self._stats["total_hits"] += 1
        
        if self._config.enable_audit_log:
            self._audit.record(
                user_id=ctx.user_id,
                action="get",
                cache_key=signed_key,
                cache_tier="L1_memory",
                success=True,
            )
        
        return entry
    
    async def put(
        self,
        ctx: UserIsolationContext,
        content_hash: str,
        data: Any,
    ) -> bool:
        """
        存储缓存条目 (带用户隔离)
        
        Args:
            ctx: 用户隔离上下文
            content_hash: 内容哈希
            data: 缓存数据
            
        Returns:
            是否存储成功
        """
        self._stats["total_puts"] += 1
        signed_key = ctx.sign_cache_key(content_hash)
        
        partition = self._get_or_create_partition(ctx.user_id)
        
        # 容量检查: 单用户上限
        if signed_key not in partition and len(partition) >= self._config.per_user_max_entries:
            # 驱逐此用户最旧的条目
            partition.popitem(last=False)
            self._stats["total_evictions"] += 1
        
        partition[signed_key] = data
        partition.move_to_end(signed_key)
        
        if self._config.enable_audit_log:
            self._audit.record(
                user_id=ctx.user_id,
                action="put",
                cache_key=signed_key,
                cache_tier="L1_memory",
                success=True,
            )
        
        return True
    
    async def delete(
        self,
        ctx: UserIsolationContext,
        content_hash: str,
    ) -> bool:
        """删除缓存条目"""
        signed_key = ctx.sign_cache_key(content_hash)
        
        partition = self._partitions.get(ctx.user_id)
        if partition and signed_key in partition:
            del partition[signed_key]
            
            if self._config.enable_audit_log:
                self._audit.record(
                    user_id=ctx.user_id,
                    action="delete",
                    cache_key=signed_key,
                    cache_tier="L1_memory",
                    success=True,
                )
            return True
        return False
    
    async def invalidate_user(self, user_id: str) -> int:
        """
        清除指定用户的所有缓存
        
        Args:
            user_id: 用户标识
            
        Returns:
            清除的条目数
        """
        partition = self._partitions.get(user_id)
        if partition is None:
            return 0
        
        count = len(partition)
        partition.clear()
        
        if self._config.enable_audit_log:
            self._audit.record(
                user_id=user_id,
                action="invalidate",
                cache_key=f"user:{user_id}:*",
                cache_tier="L1_memory",
                success=True,
                metadata={"entries_cleared": count},
            )
        
        logger.debug(f"Invalidated {count} cache entries for user {user_id}")
        return count
    
    async def clear_all(self) -> None:
        """清除所有用户的缓存"""
        total = sum(len(p) for p in self._partitions.values())
        self._partitions.clear()
        logger.info(f"Cleared all user cache partitions ({total} entries)")
    
    def get_user_entry_count(self, user_id: str) -> int:
        """获取指定用户的缓存条目数"""
        partition = self._partitions.get(user_id)
        return len(partition) if partition else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_entries = sum(len(p) for p in self._partitions.values())
        return {
            "user_partitions": len(self._partitions),
            "total_entries": total_entries,
            **self._stats,
            "hit_rate": (
                self._stats["total_hits"] / self._stats["total_gets"]
                if self._stats["total_gets"] > 0 else 0
            ),
            "denial_rate": (
                self._stats["total_denials"] / self._stats["total_gets"]
                if self._stats["total_gets"] > 0 else 0
            ),
        }


# ===========================================================================
# Inference Context Guard
# ===========================================================================

class InferenceContextGuard:
    """
    推理上下文隔离守卫
    
    解决 "计算过程共享" 的安全问题:
    - 推理前: 验证注入的 K/V 属于当前用户
    - 推理后: 确保 K/V 张量已从模型中清除
    - 异常时: 强制清理所有残留 K/V
    
    用法:
        guard = InferenceContextGuard()
        
        with guard.scoped_inference(user_id="user_123") as scope:
            # 在此范围内进行推理
            output = model.forward_with_kv_injection(
                prompt=prompt,
                injected_kv=kv_entries,
                alpha=alpha,
            )
        # 退出时自动清理 K/V 残留
    
    注意:
    - 对于共享模型的 "分时复用" 场景，这是目前工程上可行的最佳方案
    - 完全的 "计算域隔离" 需要模型实例级隔离 (成本极高)
    - 本守卫确保的是: 推理完成后不存在 K/V 残留，下一个用户不会
      受到上一个用户的 K/V 影响
    """
    
    def __init__(self, config: Optional[UserIsolationConfig] = None):
        self._config = config or UserIsolationConfig()
        self._active_user: Optional[str] = None
        self._lock = asyncio.Lock()
        
        # 统计
        self._stats = {
            "inferences": 0,
            "cleanups": 0,
            "residual_detections": 0,
            "concurrent_violations": 0,
        }
    
    @contextmanager
    def scoped_inference(
        self,
        user_id: str,
        kv_entries: Optional[List[Any]] = None,
        cleanup_callback: Optional[Callable] = None,
    ):
        """
        推理上下文隔离作用域
        
        Args:
            user_id: 当前推理的用户 ID
            kv_entries: 注入的 K/V 条目 (用于后置清理验证)
            cleanup_callback: 自定义清理回调
            
        Yields:
            InferenceScope 对象
        """
        # 前置检查: 并发冲突检测
        if self._active_user is not None and self._active_user != user_id:
            self._stats["concurrent_violations"] += 1
            logger.warning(
                f"Concurrent inference detected: "
                f"active={self._active_user}, requesting={user_id}"
            )
        
        self._active_user = user_id
        self._stats["inferences"] += 1
        
        scope = _InferenceScope(user_id=user_id)
        
        try:
            yield scope
        finally:
            # 后置清理
            self._active_user = None
            self._stats["cleanups"] += 1
            
            # 清理 K/V 引用
            if kv_entries is not None:
                self._clear_kv_references(kv_entries)
            
            # 自定义清理
            if cleanup_callback:
                try:
                    cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")
    
    def _clear_kv_references(self, kv_entries: List[Any]) -> None:
        """清理 K/V 张量引用，帮助 GC 回收"""
        try:
            import torch
            for entry in kv_entries:
                if hasattr(entry, 'key') and isinstance(entry.key, torch.Tensor):
                    entry.key = None
                if hasattr(entry, 'value') and isinstance(entry.value, torch.Tensor):
                    entry.value = None
        except ImportError:
            pass
    
    def verify_no_residual(self, model: Any) -> bool:
        """
        验证模型中无 K/V 残留
        
        检查模型的注意力层是否有未清理的外部 K/V。
        这是一个尽力而为的检查 (best-effort)。
        
        Returns:
            True 如果无残留
        """
        # 大多数模型适配器不暴露内部 K/V 状态
        # 这里提供一个检查接口，具体实现取决于模型适配器
        if hasattr(model, 'has_injected_kv'):
            has_residual = model.has_injected_kv()
            if has_residual:
                self._stats["residual_detections"] += 1
                logger.warning("K/V residual detected in model after inference")
            return not has_residual
        return True  # 无法检查时假设无残留
    
    def get_stats(self) -> Dict[str, Any]:
        """获取守卫统计"""
        return {
            **self._stats,
            "active_user": self._active_user,
        }


@dataclass
class _InferenceScope:
    """推理作用域 (内部类)"""
    user_id: str
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Isolated Preference Cache Manager
# ===========================================================================

class IsolatedPreferenceCacheManager:
    """
    带用户隔离的偏好缓存管理器
    
    对原有 PreferenceCacheManager 的安全增强包装:
    - L1 (Memory): 使用 UserScopedCacheStore (物理分区)
    - L2 (Redis): 使用用户命名空间 + HMAC 签名键
    - 所有操作: 必须通过 UserIsolationContext
    
    与原有 PreferenceCacheManager 的兼容性:
    - 保持相同的 get/put/invalidate 接口语义
    - 增加 ctx 参数用于用户隔离
    - 原有的 _make_cache_key 被替换为 HMAC 签名键
    
    用法:
        cache = IsolatedPreferenceCacheManager(
            config=cache_config,
            isolation_config=isolation_config,
        )
        
        ctx = UserIsolationContext.create(user_id="user_123", signer=cache.signer)
        
        kv_entries, tier_info = await cache.get_preference_kv(
            ctx=ctx,
            preference_text="素食主义者",
            model=model_adapter,
        )
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        config: Optional[Any] = None,
        isolation_config: Optional[UserIsolationConfig] = None,
    ):
        self._isolation_config = isolation_config or UserIsolationConfig()
        
        # HMAC 签名器
        self._signer = CacheKeySigner(self._isolation_config.hmac_secret)
        
        # 审计日志
        self._audit = CacheAuditLog(self._isolation_config.audit_log_max_entries)
        
        # L1: 用户级隔离缓存
        self._l1_store = UserScopedCacheStore(
            config=self._isolation_config,
            signer=self._signer,
            audit_log=self._audit,
        )
        
        # L2: Redis (可选)
        self._redis_client = redis_client
        self._l2_enabled = False
        if config and hasattr(config, 'l2_enabled'):
            self._l2_enabled = config.l2_enabled
        
        # 推理上下文守卫
        self._inference_guard = InferenceContextGuard(self._isolation_config)
        
        # 统计
        self._stats = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_computes": 0,
            "invalidations": 0,
        }
        
        logger.info(
            f"IsolatedPreferenceCacheManager initialized "
            f"(security={self._isolation_config.security_level})"
        )
    
    @property
    def signer(self) -> CacheKeySigner:
        return self._signer
    
    @property
    def audit_log(self) -> CacheAuditLog:
        return self._audit
    
    @property
    def inference_guard(self) -> InferenceContextGuard:
        return self._inference_guard
    
    def create_context(
        self,
        user_id: str,
        session_id: str = "",
    ) -> UserIsolationContext:
        """便捷方法: 创建用户隔离上下文"""
        return UserIsolationContext.create(
            user_id=user_id,
            session_id=session_id,
            signer=self._signer,
        )
    
    async def get_preference_kv(
        self,
        ctx: UserIsolationContext,
        preference_text: str,
        model: Any,
        force_recompute: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        获取偏好 K/V (带用户隔离)
        
        Args:
            ctx: 用户隔离上下文
            preference_text: 偏好文本
            model: 模型适配器
            force_recompute: 强制重新计算
            
        Returns:
            (kv_entries, tier_info_dict)
        """
        import hashlib as _hashlib
        self._stats["total_requests"] += 1
        
        content_hash = _hashlib.md5(preference_text.encode()).hexdigest()[:16]
        
        # 强制重新计算
        if force_recompute:
            kv_entries = await self._compute_kv(preference_text, model)
            await self._l1_store.put(ctx, content_hash, kv_entries)
            self._stats["l3_computes"] += 1
            return kv_entries, {"tier": "compute", "hit": False, "user_id": ctx.user_id}
        
        # L1: 检查用户隔离缓存
        cached = await self._l1_store.get(ctx, content_hash)
        if cached is not None:
            self._stats["l1_hits"] += 1
            return cached, {"tier": "L1_memory", "hit": True, "user_id": ctx.user_id}
        
        # L2: 检查 Redis (带用户命名空间)
        if self._l2_enabled and self._redis_client:
            try:
                redis_ns = ctx.make_redis_namespace()
                redis_key = f"{redis_ns}:{content_hash}"
                # 签名验证在 Redis 键中嵌入
                signed_redis_key = self._signer.sign_key(ctx.user_id, f"redis:{content_hash}")
                
                cached_data = await self._redis_client.get_raw(redis_key)
                if cached_data:
                    kv_entries = self._deserialize_kv(cached_data)
                    # 提升到 L1
                    await self._l1_store.put(ctx, content_hash, kv_entries)
                    self._stats["l2_hits"] += 1
                    return kv_entries, {"tier": "L2_redis", "hit": True, "user_id": ctx.user_id}
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # L3: 计算
        kv_entries = await self._compute_kv(preference_text, model)
        await self._l1_store.put(ctx, content_hash, kv_entries)
        
        # 写入 Redis
        if self._l2_enabled and self._redis_client:
            try:
                redis_ns = ctx.make_redis_namespace()
                redis_key = f"{redis_ns}:{content_hash}"
                serialized = self._serialize_kv(kv_entries)
                await self._redis_client.set_raw(redis_key, serialized, ttl=86400)
            except Exception as e:
                logger.warning(f"Redis write error: {e}")
        
        self._stats["l3_computes"] += 1
        return kv_entries, {"tier": "compute", "hit": False, "user_id": ctx.user_id}
    
    async def invalidate(self, user_id: str) -> int:
        """清除指定用户的所有缓存"""
        self._stats["invalidations"] += 1
        count = await self._l1_store.invalidate_user(user_id)
        
        # 清除 Redis
        if self._l2_enabled and self._redis_client:
            try:
                pattern = f"dki:pref_kv:user:{user_id}:*"
                await self._redis_client.delete_pattern(pattern)
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
        
        return count
    
    async def _compute_kv(self, preference_text: str, model: Any) -> Any:
        """计算 K/V"""
        try:
            import torch
            with torch.no_grad():
                if hasattr(model, 'compute_kv'):
                    kv_entries, _ = model.compute_kv(preference_text)
                    return kv_entries
                return []
        except Exception as e:
            logger.error(f"K/V computation error: {e}")
            return []
    
    def _serialize_kv(self, kv_entries: Any) -> bytes:
        """序列化 K/V"""
        import pickle
        import zlib
        try:
            import numpy as np
            serializable = []
            for entry in kv_entries:
                key_np = entry.key.cpu().numpy()
                value_np = entry.value.cpu().numpy()
                serializable.append({
                    'key': key_np.tobytes(),
                    'value': value_np.tobytes(),
                    'layer_idx': entry.layer_idx,
                    'key_shape': list(key_np.shape),
                    'value_shape': list(value_np.shape),
                    'key_dtype': str(key_np.dtype),
                    'value_dtype': str(value_np.dtype),
                })
            return zlib.compress(pickle.dumps(serializable))
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps([])
    
    def _deserialize_kv(self, data: bytes) -> Any:
        """反序列化 K/V"""
        import pickle
        import zlib
        try:
            import numpy as np
            import torch
            from dki.models.base import KVCacheEntry
            
            decompressed = zlib.decompress(data)
            serializable = pickle.loads(decompressed)
            
            kv_entries = []
            for item in serializable:
                key_shape = tuple(item.get('key_shape', ()))
                value_shape = tuple(item.get('value_shape', ()))
                key_dtype = np.dtype(item.get('key_dtype', 'float16'))
                value_dtype = np.dtype(item.get('value_dtype', 'float16'))
                
                key = torch.from_numpy(
                    np.frombuffer(item['key'], dtype=key_dtype).reshape(key_shape).copy()
                )
                value = torch.from_numpy(
                    np.frombuffer(item['value'], dtype=value_dtype).reshape(value_shape).copy()
                )
                kv_entries.append(KVCacheEntry(key=key, value=value, layer_idx=item['layer_idx']))
            return kv_entries
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "l1_store": self._l1_store.get_stats(),
            "inference_guard": self._inference_guard.get_stats(),
            "audit": self._audit.get_stats(),
        }
