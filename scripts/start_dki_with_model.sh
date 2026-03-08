#!/bin/bash

# DKI 内置模型启动脚本
# DKI 会自动加载模型用于 K/V 计算和注入
#
# 用法:
#   bash scripts/start_dki_with_model.sh [MODEL] [PORT] [MODE]
#
# 参数:
#   MODEL — 模型名称 (默认: qianwen_14b)
#     vLLM 引擎:   deepseek_14b | qianwen_14b | qianwen_27b | llama_8b
#     SGLang 引擎: qianwen35_27b_sglang | qianwen35_7b_sglang
#   PORT  — 服务端口 (默认: 8000)
#   MODE  — 启动模式: api | demo | web (默认: demo)
#
# 示例:
#   bash scripts/start_dki_with_model.sh llama_8b 8000 demo
#   bash scripts/start_dki_with_model.sh qianwen_14b 8080 api
#   bash scripts/start_dki_with_model.sh qianwen_27b 8000 demo
#   bash scripts/start_dki_with_model.sh qianwen35_27b_sglang 8000 demo  # SGLang + Qwen3.5
#   bash scripts/start_dki_with_model.sh qianwen35_7b_sglang 8000 api    # SGLang + Qwen3.5 7B

source /opt/ai-demo/venv/bin/activate
cd /opt/ai-demo/dki

# 模型配置 (可通过参数覆盖)
MODEL=${1:-"qianwen_14b"}
PORT=${2:-8000}
MODE=${3:-"demo"}

echo "=========================================="
echo "Starting DKI with built-in model: $MODEL"
echo "=========================================="

# 根据模型设置环境变量
# DKI_MODEL_IMPL: vLLM 模型实现后端
#   - "auto" (默认): vLLM 自动选择最优实现
#   - "transformers": 强制使用 Transformers backend (新架构兼容, 如 Qwen-3.5)
case $MODEL in
    "deepseek_14b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/deepseek-r1-distrill-qwen-14b"
        export DKI_MODEL_ENGINE="vllm"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="4bit"       # 4bit 8bit none
        export DKI_MODEL_IMPL="auto"         # vLLM 原生支持
        ;;
    "qianwen_14b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qwen3-14b-instruct"
        export DKI_MODEL_ENGINE="vllm"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="4bit"       # 4bit 8bit none
        export DKI_MODEL_IMPL="auto"         # vLLM 原生支持 Qwen3
        ;;
    "qianwen_27b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qwen3.5-27b-gptq-int4"
        export DKI_MODEL_ENGINE="vllm"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="gptq"       # 预量化 GPTQ 模型
        # Qwen-3.5 架构 vLLM 原生尚不支持, 需要 Transformers backend
        export DKI_MODEL_IMPL="transformers"
        ;;
    "llama_8b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/llama-3.1-8b-instruct"
        export DKI_MODEL_ENGINE="llama"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="4bit"       # 4bit 8bit none
        export DKI_MODEL_IMPL="auto"         # HF 引擎不使用此参数
        ;;

    # ============ SGLang 引擎模型 ============
    # SGLang 原生支持 Qwen3.5 等新架构, 无需 model_impl="transformers" 回退
    # RadixAttention 提供更高效的前缀复用, 适合 DKI 偏好注入场景
    "qianwen35_27b_sglang")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qwen3.5-27b-instruct"
        export DKI_MODEL_ENGINE="sglang"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="none"       # none | gptq | awq
        # SGLang 特有配置
        export DKI_SGLANG_MEM_FRACTION="0.88"
        export DKI_SGLANG_SCHEDULE="lpm"     # lpm = Longest Prefix Match (推荐)
        export DKI_SGLANG_CHUNKED_PREFILL="8192"
        ;;
    "qianwen35_27b_sglang_gptq")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qwen3.5-27b-gptq-int4"
        export DKI_MODEL_ENGINE="sglang"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="gptq"       # 预量化 GPTQ 模型
        export DKI_SGLANG_MEM_FRACTION="0.88"
        export DKI_SGLANG_SCHEDULE="lpm"
        export DKI_SGLANG_CHUNKED_PREFILL="8192"
        ;;
    "qianwen35_7b_sglang")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qwen3.5-7b-instruct"
        export DKI_MODEL_ENGINE="sglang"
        export CUDA_VISIBLE_DEVICES=0
        export DKI_QUANTIZATION="none"       # none | 4bit | gptq | awq
        export DKI_SGLANG_MEM_FRACTION="0.88"
        export DKI_SGLANG_SCHEDULE="lpm"
        export DKI_SGLANG_CHUNKED_PREFILL="8192"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo ""
        echo "Available models:"
        echo "  vLLM 引擎:"
        echo "    deepseek_14b, qianwen_14b, qianwen_27b, llama_8b"
        echo "  SGLang 引擎 (Qwen3.5):"
        echo "    qianwen35_27b_sglang, qianwen35_27b_sglang_gptq, qianwen35_7b_sglang"
        exit 1
        ;;
esac

echo "Model path: $DKI_MODEL_PATH"
echo "Engine: $DKI_MODEL_ENGINE"
echo "Mode: $MODE"
echo ""

# 使用环境变量驱动的配置文件 (config_env.yaml)
# 该配置文件通过 ${DKI_MODEL_PATH} 和 ${DKI_MODEL_ENGINE} 自动对齐模型
# 如需使用固定配置, 改为 config/config.yaml
export DKI_CONFIG_PATH="${DKI_CONFIG_PATH:-config/config_env.yaml}"
echo "Config: $DKI_CONFIG_PATH"
echo ""

# 根据模式启动服务
case $MODE in
    "demo")
        echo "Launching Demo App (independent upper-layer application)..."
        echo ""
        python main.py demo \
            --host 0.0.0.0 \
            --port $PORT \
            --config $DKI_CONFIG_PATH \
            2>&1 | tee /opt/ai-demo/logs/dki_demo_${MODEL}.log
        ;;
    "api")
        echo "Launching DKI API Server..."
        echo ""
        python main.py api \
            --host 0.0.0.0 \
            --port $PORT \
            --engine $DKI_MODEL_ENGINE \
            --config $DKI_CONFIG_PATH \
            2>&1 | tee /opt/ai-demo/logs/dki_${MODEL}.log
        ;;
    "web")
        echo "Launching DKI Web UI (experiment system)..."
        echo ""
        python main.py web \
            --host 0.0.0.0 \
            --port $PORT \
            --engine $DKI_MODEL_ENGINE \
            --config $DKI_CONFIG_PATH \
            2>&1 | tee /opt/ai-demo/logs/dki_web_${MODEL}.log
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available: api, demo, web"
        exit 1
        ;;
esac
