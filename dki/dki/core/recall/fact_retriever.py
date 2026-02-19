"""
DKI Recall v4 — 事实检索器

通过 trace_id 检索原始消息内容。
支持 offset + limit 分块，用于处理长文本消息。

trace_id → 数据库查询 → 返回原始消息的分页结果

最大返回量由 config.recall.fact_call.max_fact_tokens 限制

Author: AGI Demo Project
Version: 4.0.0
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from dki.core.recall.recall_config import (
    RecallConfig,
    FactRequest,
    FactResponse,
)


class FactRetriever:
    """
    事实检索器 (function call 后端)
    
    通过 trace_id 检索原始消息内容。
    支持 offset + limit 分块，用于处理长文本消息。
    """

    def __init__(
        self,
        config: RecallConfig,
        conversation_repo: Optional[Any] = None,
    ):
        self.config = config
        self._conversation_repo = conversation_repo
        self._stats = {
            "retrievals": 0,
            "not_found": 0,
        }

    def retrieve(
        self,
        trace_id: str,
        session_id: str,
        offset: int = 0,
        limit: int = 5,
        db_session: Optional[Any] = None,
    ) -> FactResponse:
        """
        检索指定 trace_id 的原始消息
        
        如果 trace_id 对应的是一条长消息，
        则按 offset+limit 分块返回消息文本片段。
        
        如果 trace_id 对应的是一组会话轮次 (summary 覆盖多轮),
        则按 offset+limit 返回原始消息列表。
        
        Args:
            trace_id: 消息溯源 ID (message_id)
            session_id: 会话 ID
            offset: 分页偏移
            limit: 每次返回条数
            db_session: 数据库 session (可选)
        """
        self._stats["retrievals"] += 1

        # 配置限制
        max_batch = self.config.fact_call.batch_size
        limit = min(limit, max_batch)

        if not self._conversation_repo:
            logger.warning("FactRetriever: no conversation_repo configured")
            return FactResponse(
                trace_id=trace_id,
                total_count=0,
                offset=offset,
                has_more=False,
            )

        try:
            # 尝试通过 message_id 获取单条消息
            message = self._get_message_by_id(trace_id, session_id, db_session)

            if message is None:
                self._stats["not_found"] += 1
                logger.warning(f"FactRetriever: message not found for trace_id={trace_id}")
                return FactResponse(
                    trace_id=trace_id,
                    total_count=0,
                    offset=offset,
                    has_more=False,
                )

            content = getattr(message, "content", str(message))
            role = getattr(message, "role", "unknown")
            timestamp = str(getattr(message, "created_at", ""))

            # 如果内容很长, 按字符分块
            content_length = len(content)
            chunk_size = 500  # 每块约 500 字符

            if content_length <= chunk_size:
                # 短消息, 直接返回
                return FactResponse(
                    messages=[{
                        "role": role,
                        "content": content,
                        "timestamp": timestamp,
                    }],
                    trace_id=trace_id,
                    total_count=1,
                    offset=0,
                    has_more=False,
                )

            # 长消息, 分块
            chunks = self._chunk_content(content, chunk_size)
            total_chunks = len(chunks)

            # 分页
            start = offset
            end = min(offset + limit, total_chunks)

            chunk_messages = []
            for i in range(start, end):
                chunk_messages.append({
                    "role": role,
                    "content": chunks[i],
                    "timestamp": timestamp,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                })

            return FactResponse(
                messages=chunk_messages,
                trace_id=trace_id,
                total_count=total_chunks,
                offset=offset,
                has_more=(end < total_chunks),
            )

        except Exception as e:
            logger.error(f"FactRetriever error: {e}")
            return FactResponse(
                trace_id=trace_id,
                total_count=0,
                offset=offset,
                has_more=False,
            )

    # ================================================================
    # 消息检索
    # ================================================================

    def _get_message_by_id(
        self,
        message_id: str,
        session_id: str,
        db_session: Optional[Any],
    ) -> Optional[Any]:
        """根据 message_id 获取消息"""
        # 方式 1: 如果 repo 有 get_by_id 方法
        if hasattr(self._conversation_repo, "get_by_id"):
            try:
                return self._conversation_repo.get_by_id(message_id)
            except Exception:
                pass

        # 方式 2: 从会话消息中查找
        try:
            messages = self._conversation_repo.get_by_session(
                session_id=session_id,
            )
            for msg in messages:
                msg_id = str(
                    getattr(msg, "id", None)
                    or getattr(msg, "message_id", None)
                )
                if msg_id == message_id:
                    return msg
        except Exception as e:
            logger.warning(f"get_message_by_id fallback failed: {e}")

        return None

    # ================================================================
    # 内容分块
    # ================================================================

    @staticmethod
    def _chunk_content(content: str, chunk_size: int) -> List[str]:
        """
        按自然断点分块内容
        
        优先按段落/句子分割, 而非硬截断
        """
        # 先按段落分
        paragraphs = content.split('\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += ("\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单段超过 chunk_size, 按句子再分
                if len(para) > chunk_size:
                    sub_chunks = FactRetriever._chunk_by_sentence(
                        para, chunk_size
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [content]

    @staticmethod
    def _chunk_by_sentence(text: str, chunk_size: int) -> List[str]:
        """按句子分块"""
        import re
        sentences = re.split(r'([。！？.!?\n])', text)
        
        chunks = []
        current = ""
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            # 附加标点
            if i + 1 < len(sentences):
                sent += sentences[i + 1]
            
            if len(current) + len(sent) <= chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        
        if current:
            chunks.append(current)
        
        return chunks if chunks else [text]

    # ================================================================
    # 统计
    # ================================================================

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
