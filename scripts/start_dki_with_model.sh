#!/bin/bash

# DKI 内置模型启动脚本
# DKI 会自动加载 HuggingFace 模型用于 K/V 计算和注入

source /opt/ai-demo/venv/bin/activate
cd /opt/ai-demo/dki

# 模型配置 (可通过参数覆盖)
MODEL=${1:-"qianwen_14b"}
PORT=${2:-8000}

echo "=========================================="
echo "Starting DKI with built-in model: $MODEL"
echo "=========================================="

# 根据模型设置环境变量
case $MODEL in
    "deepseek_14b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/deepseek-r1-distill-qwen-14b"
        export DKI_MODEL_ENGINE="vllm"
	export CUDA_VISIBLE_DEVICES=0
        ;;
    "qianwen_14b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/qianwen-14b-chat"
        export DKI_MODEL_ENGINE="vllm"
        export CUDA_VISIBLE_DEVICES=0
        ;;
    "llama_8b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/llama-3.1-8b-instruct"
        export DKI_MODEL_ENGINE="llama"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: deepseek_14b, qianwen_14b, llama_8b"
        exit 1
        ;;
esac

echo "Model path: $DKI_MODEL_PATH"
echo "Engine: $DKI_MODEL_ENGINE"
echo ""

# 启动 DKI API 服务 (会自动加载模型)
python main.py api \
    --host 0.0.0.0 \
    --port $PORT \
    --engine $DKI_MODEL_ENGINE \
    --config config/config.yaml \
    2>&1 | tee /opt/ai-demo/logs/dki_${MODEL}.log
