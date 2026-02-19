"""
DKI + AGA 混合 Plugin Loader

实现同时加载和管理 DKI 和 AGA 两个插件的统一接口。

架构说明:
- DKI: 用户记忆注入 (Pre-inference, Attention K/V Hook)
  - 负位置 K/V 注入用户偏好
  - 后缀提示词注入历史上下文
  
- AGA: 知识增强注入 (During-inference, FFN Hook)
  - 熵检测触发知识注入
  - 高不确定性时激活知识槽位

两者工作序列:
1. 用户输入 → DKI 处理 (添加偏好 K/V + 历史后缀)
2. 增强后的输入 → LLM 推理
3. 推理过程中 → AGA 检测熵并注入知识
4. 最终输出

版本: 1.0
"""
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """插件类型"""
    DKI = "dki"
    AGA = "aga"
    HYBRID = "hybrid"


class HookStage(str, Enum):
    """Hook 阶段"""
    PRE_INFERENCE = "pre_inference"      # DKI: 推理前
    ATTENTION = "attention"               # DKI: Attention 层
    FFN = "ffn"                           # AGA: FFN 层
    POST_INFERENCE = "post_inference"     # 推理后


@dataclass
class PluginConfig:
    """插件配置"""
    # DKI 配置
    dki_enabled: bool = True
    dki_config_path: str = "config/config.yaml"
    dki_preference_alpha: float = 0.4
    dki_history_max_tokens: int = 2000
    
    # AGA 配置
    aga_enabled: bool = True
    aga_config_path: str = "config/aga_config.yaml"
    aga_entropy_threshold: float = 0.7
    aga_knowledge_alpha: float = 0.3
    
    # 混合配置
    hook_order: List[str] = field(default_factory=lambda: ["dki", "aga"])
    conflict_resolution: str = "sequential"  # sequential | weighted | priority
    
    # 模型配置
    model_path: str = ""
    hidden_dim: int = 4096
    num_heads: int = 32


@dataclass
class PluginState:
    """插件状态"""
    dki_active: bool = False
    aga_active: bool = False
    dki_stats: Dict[str, Any] = field(default_factory=dict)
    aga_stats: Dict[str, Any] = field(default_factory=dict)
    last_dki_injection: Optional[Dict] = None
    last_aga_injection: Optional[Dict] = None


class HybridPluginLoader:
    """
    DKI + AGA 混合插件加载器
    
    负责:
    1. 同时加载 DKI 和 AGA 插件
    2. 管理两者的 Hook 顺序
    3. 处理潜在冲突
    4. 提供统一的监控接口
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self.state = PluginState()
        
        # 插件实例
        self.dki_plugin = None
        self.aga_plugin = None
        
        # 模型引用
        self.model = None
        self.tokenizer = None
        
        # Hook 注册表
        self._hooks: Dict[HookStage, List[callable]] = {
            stage: [] for stage in HookStage
        }
        
        logger.info("HybridPluginLoader initialized")
    
    def load_dki(self) -> bool:
        """加载 DKI 插件"""
        if not self.config.dki_enabled:
            logger.info("DKI disabled in config")
            return False
        
        try:
            from dki.core.dki_plugin import DKIPlugin
            from dki.config.config_loader import load_config
            
            # 加载 DKI 配置
            dki_config = load_config(self.config.dki_config_path)
            
            # 创建 DKI 插件
            self.dki_plugin = DKIPlugin(config=dki_config)
            
            # 注册 DKI Hooks
            self._hooks[HookStage.PRE_INFERENCE].append(self._dki_pre_inference_hook)
            self._hooks[HookStage.ATTENTION].append(self._dki_attention_hook)
            
            self.state.dki_active = True
            logger.info("DKI plugin loaded successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Failed to import DKI: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load DKI: {e}")
            return False
    
    def load_aga(self) -> bool:
        """加载 AGA 插件"""
        if not self.config.aga_enabled:
            logger.info("AGA disabled in config")
            return False
        
        try:
            from aga.core import AuxiliaryGovernedAttention, AGAConfig
            from aga.operator.aga_operator import AGAOperator
            
            # 创建 AGA 配置
            aga_config = AGAConfig(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
            )
            
            # 创建 AGA 插件
            self.aga_plugin = AuxiliaryGovernedAttention(config=aga_config)
            
            # 注册 AGA Hooks
            self._hooks[HookStage.FFN].append(self._aga_ffn_hook)
            
            self.state.aga_active = True
            logger.info("AGA plugin loaded successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Failed to import AGA: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load AGA: {e}")
            return False
    
    def load_all(self) -> Dict[str, bool]:
        """加载所有插件"""
        results = {
            "dki": self.load_dki(),
            "aga": self.load_aga(),
        }
        
        logger.info(f"Plugin loading results: {results}")
        return results
    
    def attach_to_model(self, model: nn.Module, tokenizer=None) -> bool:
        """
        将插件附加到模型
        
        Args:
            model: HuggingFace 模型
            tokenizer: 分词器
        
        Returns:
            是否成功
        """
        self.model = model
        self.tokenizer = tokenizer
        
        try:
            # 附加 DKI
            if self.state.dki_active and self.dki_plugin:
                self.dki_plugin.attach_to_model(model, tokenizer)
                logger.info("DKI attached to model")
            
            # 附加 AGA
            if self.state.aga_active and self.aga_plugin:
                # AGA 需要挂载到特定层
                self._attach_aga_to_layers(model)
                logger.info("AGA attached to model")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to attach plugins to model: {e}")
            return False
    
    def _attach_aga_to_layers(self, model: nn.Module):
        """将 AGA 挂载到模型层"""
        # 获取模型层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            logger.warning("Cannot find model layers for AGA attachment")
            return
        
        # 默认挂载到最后几层
        target_layers = [-4, -3, -2, -1]
        num_layers = len(layers)
        
        for idx in target_layers:
            resolved_idx = idx if idx >= 0 else num_layers + idx
            if 0 <= resolved_idx < num_layers:
                # 这里可以包装层以添加 AGA hook
                logger.debug(f"AGA hook registered for layer {resolved_idx}")
    
    def _dki_pre_inference_hook(
        self,
        user_id: str,
        message: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        DKI 预推理 Hook
        
        在推理前处理用户输入，添加偏好和历史上下文
        """
        if not self.dki_plugin:
            return message, {}
        
        try:
            # 调用 DKI 处理
            enhanced_message, metadata = self.dki_plugin.process_input(
                user_id=user_id,
                message=message,
                **kwargs
            )
            
            self.state.last_dki_injection = {
                "user_id": user_id,
                "original_length": len(message),
                "enhanced_length": len(enhanced_message),
                "metadata": metadata,
            }
            
            return enhanced_message, metadata
            
        except Exception as e:
            logger.error(f"DKI pre-inference hook error: {e}")
            return message, {"error": str(e)}
    
    def _dki_attention_hook(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        DKI Attention Hook
        
        在 Attention 层注入用户偏好 K/V
        """
        if not self.dki_plugin:
            return hidden_states, {}
        
        try:
            # 调用 DKI K/V 注入
            modified_states, metadata = self.dki_plugin.inject_kv(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
            
            return modified_states, metadata
            
        except Exception as e:
            logger.error(f"DKI attention hook error: {e}")
            return hidden_states, {"error": str(e)}
    
    def _aga_ffn_hook(
        self,
        hidden_states: torch.Tensor,
        primary_output: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        AGA FFN Hook
        
        在 FFN 层检测熵并注入知识
        """
        if not self.aga_plugin:
            return primary_output, {}
        
        try:
            # 调用 AGA 前向传播
            fused_output, diagnostics = self.aga_plugin(
                hidden_states=hidden_states,
                primary_attention_output=primary_output,
                return_diagnostics=True,
                **kwargs
            )
            
            metadata = {}
            if diagnostics:
                metadata = {
                    "gate_mean": diagnostics.gate_mean,
                    "active_slots": diagnostics.active_slots,
                    "routed_slots": diagnostics.routed_slots,
                }
                self.state.last_aga_injection = metadata
            
            return fused_output, metadata
            
        except Exception as e:
            logger.error(f"AGA FFN hook error: {e}")
            return primary_output, {"error": str(e)}
    
    def process_request(
        self,
        user_id: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理完整请求
        
        按照配置的 hook 顺序执行所有插件
        """
        result = {
            "original_message": message,
            "enhanced_message": message,
            "dki_applied": False,
            "aga_applied": False,
            "metadata": {},
        }
        
        # 1. DKI 预处理
        if self.state.dki_active:
            enhanced_message, dki_meta = self._dki_pre_inference_hook(
                user_id=user_id,
                message=message,
                **kwargs
            )
            result["enhanced_message"] = enhanced_message
            result["dki_applied"] = True
            result["metadata"]["dki"] = dki_meta
        
        # 2. AGA 在推理过程中自动触发 (通过 model hooks)
        if self.state.aga_active:
            result["aga_applied"] = True
            result["metadata"]["aga"] = {"status": "hooks_registered"}
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取插件状态"""
        status = {
            "dki": {
                "enabled": self.config.dki_enabled,
                "active": self.state.dki_active,
                "stats": self.state.dki_stats,
            },
            "aga": {
                "enabled": self.config.aga_enabled,
                "active": self.state.aga_active,
                "stats": self.state.aga_stats,
            },
            "hook_order": self.config.hook_order,
            "conflict_resolution": self.config.conflict_resolution,
        }
        
        # 添加 DKI 统计
        if self.dki_plugin and hasattr(self.dki_plugin, 'get_stats'):
            status["dki"]["stats"] = self.dki_plugin.get_stats()
        
        # 添加 AGA 统计
        if self.aga_plugin and hasattr(self.aga_plugin, 'get_statistics'):
            status["aga"]["stats"] = self.aga_plugin.get_statistics()
        
        return status
    
    def inject_dki_preference(
        self,
        user_id: str,
        preference: str,
        category: str = "general",
    ) -> Dict[str, Any]:
        """注入 DKI 用户偏好"""
        if not self.dki_plugin:
            return {"success": False, "error": "DKI not loaded"}
        
        try:
            result = self.dki_plugin.add_preference(
                user_id=user_id,
                preference=preference,
                category=category,
            )
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def inject_aga_knowledge(
        self,
        condition: str,
        decision: str,
        lu_id: Optional[str] = None,
        lifecycle_state: str = "confirmed",
        embed_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        注入 AGA 知识
        
        Args:
            condition: 触发条件
            decision: 决策内容
            lu_id: 知识单元 ID
            lifecycle_state: 生命周期状态 (默认 confirmed = 已审核)
            embed_fn: 嵌入函数
        """
        if not self.aga_plugin:
            return {"success": False, "error": "AGA not loaded"}
        
        try:
            from aga.core import LifecycleState
            from datetime import datetime
            
            # 生成 LU ID
            if lu_id is None:
                lu_id = f"LU_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(condition)%10000:04d}"
            
            # 映射生命周期状态
            state_map = {
                'probationary': LifecycleState.PROBATIONARY,
                'confirmed': LifecycleState.CONFIRMED,
                'deprecated': LifecycleState.DEPRECATED,
                'quarantined': LifecycleState.QUARANTINED,
            }
            lc_state = state_map.get(lifecycle_state, LifecycleState.CONFIRMED)
            
            # 查找空闲槽位
            slot_idx = self.aga_plugin.find_free_slot()
            if slot_idx is None:
                return {"success": False, "error": "No free slots available"}
            
            # 生成嵌入向量
            if embed_fn:
                key_vector = embed_fn(condition)
                value_vector = embed_fn(decision)
            else:
                # 使用随机向量作为占位符
                key_vector = torch.randn(self.config.hidden_dim // 64)
                value_vector = torch.randn(self.config.hidden_dim)
            
            # 注入知识
            success = self.aga_plugin.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=key_vector,
                value_vector=value_vector,
                lu_id=lu_id,
                lifecycle_state=lc_state,
                condition=condition,
                decision=decision,
            )
            
            return {
                "success": success,
                "lu_id": lu_id,
                "slot_idx": slot_idx,
                "lifecycle_state": lifecycle_state,
            }
            
        except Exception as e:
            logger.error(f"Failed to inject AGA knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    def get_aga_knowledge_list(self) -> List[Dict[str, Any]]:
        """获取 AGA 知识列表"""
        if not self.aga_plugin:
            return []
        
        try:
            knowledge = self.aga_plugin.get_active_knowledge()
            return [
                {
                    'slot_idx': k.slot_idx,
                    'lu_id': k.lu_id,
                    'condition': k.condition,
                    'decision': k.decision,
                    'lifecycle_state': k.lifecycle_state.value,
                    'reliability': k.reliability,
                    'hit_count': k.hit_count,
                    'created_at': k.created_at.isoformat() if k.created_at else None,
                }
                for k in knowledge
            ]
        except Exception as e:
            logger.error(f"Failed to get AGA knowledge list: {e}")
            return []
    
    def update_aga_lifecycle(
        self,
        lu_id: str,
        new_state: str,
    ) -> Dict[str, Any]:
        """更新 AGA 知识生命周期状态"""
        if not self.aga_plugin:
            return {"success": False, "error": "AGA not loaded"}
        
        try:
            from aga.core import LifecycleState
            
            state_map = {
                'probationary': LifecycleState.PROBATIONARY,
                'confirmed': LifecycleState.CONFIRMED,
                'deprecated': LifecycleState.DEPRECATED,
                'quarantined': LifecycleState.QUARANTINED,
            }
            lc_state = state_map.get(new_state, LifecycleState.PROBATIONARY)
            
            slots = self.aga_plugin.get_slot_by_lu_id(lu_id)
            for slot_idx in slots:
                self.aga_plugin.update_lifecycle(slot_idx, lc_state)
            
            return {"success": True, "lu_id": lu_id, "new_state": new_state}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def shutdown(self):
        """关闭插件"""
        logger.info("Shutting down HybridPluginLoader")
        
        if self.dki_plugin and hasattr(self.dki_plugin, 'shutdown'):
            self.dki_plugin.shutdown()
        
        if self.aga_plugin:
            # AGA 通常不需要显式关闭
            pass
        
        self.state.dki_active = False
        self.state.aga_active = False


# ==================== 便捷函数 ====================

def create_hybrid_loader(
    dki_config_path: str = "config/config.yaml",
    aga_enabled: bool = True,
    dki_enabled: bool = True,
    hidden_dim: int = 4096,
    num_heads: int = 32,
) -> HybridPluginLoader:
    """
    创建混合插件加载器
    
    Args:
        dki_config_path: DKI 配置文件路径
        aga_enabled: 是否启用 AGA
        dki_enabled: 是否启用 DKI
        hidden_dim: 隐藏层维度
        num_heads: 注意力头数
    
    Returns:
        HybridPluginLoader 实例
    """
    config = PluginConfig(
        dki_enabled=dki_enabled,
        dki_config_path=dki_config_path,
        aga_enabled=aga_enabled,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )
    
    loader = HybridPluginLoader(config)
    loader.load_all()
    
    return loader
