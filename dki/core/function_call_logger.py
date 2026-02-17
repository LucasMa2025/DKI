"""
DKI Function Call Logger (v3.2)

记录 DKI 系统中的 function call (如 retrieve_fact) 的完整信息。

支持两种持久化方式:
1. 数据库持久化 (默认): 写入 function_call_logs 表
2. 文本日志持久化 (降级): 当数据库不可用时, 写入文本日志文件

用途:
- 调试: 查看每次 fact call 的输入输出
- 可视化: 在 UI 中展示 function call 列表
- 审计: 追踪 function call 的调用链路
- 分析: 统计 function call 的使用频率和效果

Author: AGI Demo Project
Version: 3.2.0
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class FunctionCallLogger:
    """
    Function Call 日志记录器
    
    线程安全: 使用数据库事务或文件追加写入
    
    用法:
        fc_logger = FunctionCallLogger(db_manager=db_manager)
        
        # 记录一次 function call
        fc_logger.log(
            session_id="sess_123",
            function_name="retrieve_fact",
            arguments={"trace_id": "msg-005", "offset": 0, "limit": 5},
            response_text="[FACT_SEGMENT] ...",
            ...
        )
        
        # 查询某个 session 的所有 function call
        logs = fc_logger.get_by_session("sess_123")
    """
    
    def __init__(
        self,
        db_manager: Optional[Any] = None,
        text_log_dir: Optional[str] = None,
    ):
        """
        初始化 Function Call Logger
        
        Args:
            db_manager: 数据库管理器 (DatabaseManager)
            text_log_dir: 文本日志目录 (当数据库不可用时使用)
        """
        self._db_manager = db_manager
        self._text_log_dir = text_log_dir or "logs/function_calls"
        
        # 内存缓存 (用于快速查询和可视化)
        self._memory_logs: List[Dict[str, Any]] = []
        self._max_memory_logs = 500
        
        # 统计
        self._stats = {
            "total_logged": 0,
            "db_logged": 0,
            "text_logged": 0,
            "errors": 0,
        }
        
        logger.info(
            f"FunctionCallLogger initialized "
            f"(db={'yes' if db_manager else 'no'}, "
            f"text_log_dir={self._text_log_dir})"
        )
    
    def log(
        self,
        session_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        request_id: Optional[str] = None,
        round_index: int = 0,
        response_text: Optional[str] = None,
        response_tokens: int = 0,
        status: str = "success",
        error_message: Optional[str] = None,
        prompt_before: Optional[str] = None,
        prompt_after: Optional[str] = None,
        model_output_before: Optional[str] = None,
        latency_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """
        记录一次 function call
        
        Returns:
            日志 ID (数据库持久化时返回), 或 None
        """
        log_entry = {
            "session_id": session_id,
            "user_id": user_id,
            "turn_id": turn_id,
            "request_id": request_id,
            "round_index": round_index,
            "function_name": function_name,
            "arguments": arguments,
            "response_text": response_text,
            "response_tokens": response_tokens,
            "status": status,
            "error_message": error_message,
            "prompt_before": prompt_before,
            "prompt_after": prompt_after,
            "model_output_before": model_output_before,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # 1. 写入内存缓存
        self._memory_logs.append(log_entry)
        if len(self._memory_logs) > self._max_memory_logs:
            self._memory_logs = self._memory_logs[-self._max_memory_logs:]
        
        self._stats["total_logged"] += 1
        
        # 2. 尝试数据库持久化
        log_id = None
        if self._db_manager:
            try:
                log_id = self._persist_to_db(log_entry)
                self._stats["db_logged"] += 1
            except Exception as e:
                logger.warning(f"Failed to persist function call log to DB: {e}")
                # 降级到文本日志
                self._persist_to_text(log_entry)
                self._stats["text_logged"] += 1
        else:
            # 无数据库, 直接写入文本日志
            self._persist_to_text(log_entry)
            self._stats["text_logged"] += 1
        
        logger.debug(
            f"Function call logged: {function_name} "
            f"(session={session_id}, round={round_index}, "
            f"status={status}, latency={latency_ms:.1f}ms)"
        )
        
        return log_id
    
    def _persist_to_db(self, entry: Dict[str, Any]) -> Optional[int]:
        """持久化到数据库"""
        from dki.database.repository import FunctionCallLogRepository
        
        with self._db_manager.session_scope() as db:
            repo = FunctionCallLogRepository(db)
            log_obj = repo.create(
                session_id=entry["session_id"],
                function_name=entry["function_name"],
                arguments=entry["arguments"],
                user_id=entry.get("user_id"),
                turn_id=entry.get("turn_id"),
                request_id=entry.get("request_id"),
                round_index=entry.get("round_index", 0),
                response_text=entry.get("response_text"),
                response_tokens=entry.get("response_tokens", 0),
                status=entry.get("status", "success"),
                error_message=entry.get("error_message"),
                prompt_before=entry.get("prompt_before"),
                prompt_after=entry.get("prompt_after"),
                model_output_before=entry.get("model_output_before"),
                latency_ms=entry.get("latency_ms", 0),
                metadata=entry.get("metadata"),
            )
            return log_obj.id
    
    def _persist_to_text(self, entry: Dict[str, Any]) -> None:
        """持久化到文本日志文件"""
        try:
            log_dir = Path(self._text_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 按日期分文件
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = log_dir / f"function_calls_{date_str}.jsonl"
            
            # 追加写入 (JSONL 格式)
            with open(log_file, "a", encoding="utf-8") as f:
                # 清理大文本字段以避免日志过大
                compact_entry = {**entry}
                for field in ["prompt_before", "prompt_after", "response_text"]:
                    if compact_entry.get(field) and len(compact_entry[field]) > 2000:
                        compact_entry[field] = (
                            compact_entry[field][:2000] + 
                            f"... [truncated, total {len(entry[field])} chars]"
                        )
                
                f.write(json.dumps(compact_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to persist function call log to text: {e}")
            self._stats["errors"] += 1
    
    def get_by_session(
        self,
        session_id: str,
        limit: int = 100,
        include_prompts: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        获取某个 session 的所有 function call 日志
        
        Args:
            session_id: 会话 ID
            limit: 最大返回数量
            include_prompts: 是否包含完整 prompt (大文本)
        """
        # 优先从数据库查询
        if self._db_manager:
            try:
                return self._query_from_db(
                    session_id=session_id,
                    limit=limit,
                    include_prompts=include_prompts,
                )
            except Exception as e:
                logger.warning(f"Failed to query function call logs from DB: {e}")
        
        # 降级到内存缓存
        results = [
            log for log in self._memory_logs
            if log.get("session_id") == session_id
        ]
        
        if not include_prompts:
            results = [self._strip_prompts(log) for log in results]
        
        return results[-limit:]
    
    def get_by_request_id(
        self,
        request_id: str,
        include_prompts: bool = True,
    ) -> List[Dict[str, Any]]:
        """获取某个请求的所有 function call 日志"""
        if self._db_manager:
            try:
                from dki.database.repository import FunctionCallLogRepository
                with self._db_manager.session_scope() as db:
                    repo = FunctionCallLogRepository(db)
                    logs = repo.get_by_request_id(request_id)
                    results = [log.to_dict() for log in logs]
                    if not include_prompts:
                        results = [self._strip_prompts(r) for r in results]
                    return results
            except Exception as e:
                logger.warning(f"Failed to query by request_id: {e}")
        
        # 降级到内存缓存
        results = [
            log for log in self._memory_logs
            if log.get("request_id") == request_id
        ]
        if not include_prompts:
            results = [self._strip_prompts(log) for log in results]
        return results
    
    def _query_from_db(
        self,
        session_id: str,
        limit: int = 100,
        include_prompts: bool = False,
    ) -> List[Dict[str, Any]]:
        """从数据库查询"""
        from dki.database.repository import FunctionCallLogRepository
        
        with self._db_manager.session_scope() as db:
            repo = FunctionCallLogRepository(db)
            logs = repo.get_by_session(session_id, limit=limit)
            results = [log.to_dict() for log in logs]
            
            if not include_prompts:
                results = [self._strip_prompts(r) for r in results]
            
            return results
    
    @staticmethod
    def _strip_prompts(entry: Dict[str, Any]) -> Dict[str, Any]:
        """移除大文本字段 (prompt_before, prompt_after) 以减少传输量"""
        result = {**entry}
        for field in ["prompt_before", "prompt_after"]:
            if result.get(field):
                text = result[field]
                result[field] = (
                    text[:200] + f"... [{len(text)} chars total]"
                    if len(text) > 200 else text
                )
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "memory_logs_count": len(self._memory_logs),
        }
    
    def get_memory_logs(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """从内存缓存获取日志 (用于可视化)"""
        if session_id:
            logs = [
                log for log in self._memory_logs
                if log.get("session_id") == session_id
            ]
        else:
            logs = self._memory_logs
        
        return logs[-limit:]
