#!/bin/bash

# DKI + AGA 混合部署启动脚本
# 同时启动 DKI 和 AGA 服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DKI_PORT=${DKI_PORT:-8000}
AGA_PORT=${AGA_PORT:-8081}
AGA_EXPERIMENT_PORT=${AGA_EXPERIMENT_PORT:-8765}
MODEL=${MODEL:-"deepseek_7b"}
VENV_PATH=${VENV_PATH:-"/opt/ai-demo/venv"}
DKI_PATH=${DKI_PATH:-"/opt/ai-demo/DKI"}
AGA_PATH=${AGA_PATH:-"/opt/ai-demo/AGA"}
LOG_DIR=${LOG_DIR:-"/opt/ai-demo/logs"}

# 帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL       Model to use (deepseek_7b, llama_8b, deepseek_33b)"
    echo "  --dki-port PORT     DKI API port (default: 8000)"
    echo "  --aga-port PORT     AGA API port (default: 8081)"
    echo "  --exp-port PORT     AGA Experiment Tool port (default: 8765)"
    echo "  --dki-only          Only start DKI"
    echo "  --aga-only          Only start AGA"
    echo "  --no-experiment     Don't start AGA Experiment Tool"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --model deepseek_7b"
    echo "  $0 --model llama_8b --dki-port 8000 --aga-port 8081"
    echo "  $0 --dki-only --model deepseek_7b"
}

# 解析参数
DKI_ONLY=false
AGA_ONLY=false
NO_EXPERIMENT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dki-port)
            DKI_PORT="$2"
            shift 2
            ;;
        --aga-port)
            AGA_PORT="$2"
            shift 2
            ;;
        --exp-port)
            AGA_EXPERIMENT_PORT="$2"
            shift 2
            ;;
        --dki-only)
            DKI_ONLY=true
            shift
            ;;
        --aga-only)
            AGA_ONLY=true
            shift
            ;;
        --no-experiment)
            NO_EXPERIMENT=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 创建日志目录
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DKI + AGA Hybrid Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Model:              ${GREEN}$MODEL${NC}"
echo -e "DKI Port:           ${GREEN}$DKI_PORT${NC}"
echo -e "AGA API Port:       ${GREEN}$AGA_PORT${NC}"
echo -e "AGA Experiment Port: ${GREEN}$AGA_EXPERIMENT_PORT${NC}"
echo ""

# 激活虚拟环境
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment not found at $VENV_PATH${NC}"
fi

# 停止现有服务
echo ""
echo -e "${YELLOW}Stopping existing services...${NC}"
tmux kill-session -t dki 2>/dev/null || true
tmux kill-session -t aga_api 2>/dev/null || true
tmux kill-session -t aga_exp 2>/dev/null || true
sleep 2

# 设置模型路径
case $MODEL in
    "deepseek_7b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/deepseek-llm-7b-chat"
        export DKI_MODEL_ENGINE="vllm"
        ;;
    "deepseek_33b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/deepseek-coder-v2-instruct"
        export DKI_MODEL_ENGINE="vllm"
        export CUDA_VISIBLE_DEVICES=0,1
        ;;
    "llama_8b")
        export DKI_MODEL_PATH="/opt/ai-demo/models/llama-3.1-8b-instruct"
        export DKI_MODEL_ENGINE="llama"
        ;;
    *)
        echo -e "${RED}Unknown model: $MODEL${NC}"
        exit 1
        ;;
esac

echo -e "Model Path: ${GREEN}$DKI_MODEL_PATH${NC}"
echo ""

# 启动 DKI
if [ "$AGA_ONLY" = false ]; then
    echo -e "${BLUE}Starting DKI service...${NC}"
    
    tmux new-session -d -s dki
    tmux send-keys -t dki "source $VENV_PATH/bin/activate && cd $DKI_PATH && python main.py api --host 0.0.0.0 --port $DKI_PORT --engine $DKI_MODEL_ENGINE --config config/config.yaml 2>&1 | tee $LOG_DIR/dki_hybrid.log" Enter
    
    echo -e "${GREEN}✓ DKI service started (tmux session: dki)${NC}"
fi

# 启动 AGA API
if [ "$DKI_ONLY" = false ]; then
    echo -e "${BLUE}Starting AGA API service...${NC}"
    
    tmux new-session -d -s aga_api
    tmux send-keys -t aga_api "source $VENV_PATH/bin/activate && cd $AGA_PATH && python -m aga.api --host 0.0.0.0 --port $AGA_PORT 2>&1 | tee $LOG_DIR/aga_api_hybrid.log" Enter
    
    echo -e "${GREEN}✓ AGA API service started (tmux session: aga_api)${NC}"
    
    # 启动 AGA Experiment Tool
    if [ "$NO_EXPERIMENT" = false ]; then
        echo -e "${BLUE}Starting AGA Experiment Tool...${NC}"
        
        tmux new-session -d -s aga_exp
        tmux send-keys -t aga_exp "source $VENV_PATH/bin/activate && cd $AGA_PATH/aga_experiment_tool && python app.py --port $AGA_EXPERIMENT_PORT 2>&1 | tee $LOG_DIR/aga_experiment_hybrid.log" Enter
        
        echo -e "${GREEN}✓ AGA Experiment Tool started (tmux session: aga_exp)${NC}"
    fi
fi

# 等待服务启动
echo ""
echo -e "${YELLOW}Waiting for services to start (60 seconds)...${NC}"
sleep 60

# 验证服务
echo ""
echo -e "${BLUE}Verifying services...${NC}"

if [ "$AGA_ONLY" = false ]; then
    if curl -s http://localhost:$DKI_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ DKI service is running on port $DKI_PORT${NC}"
    else
        echo -e "${RED}✗ DKI service failed to start${NC}"
        echo -e "  Check logs: tail -f $LOG_DIR/dki_hybrid.log"
    fi
fi

if [ "$DKI_ONLY" = false ]; then
    if curl -s http://localhost:$AGA_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ AGA API service is running on port $AGA_PORT${NC}"
    else
        echo -e "${YELLOW}⚠ AGA API service may not be ready yet${NC}"
        echo -e "  Check logs: tail -f $LOG_DIR/aga_api_hybrid.log"
    fi
    
    if [ "$NO_EXPERIMENT" = false ]; then
        if curl -s http://localhost:$AGA_EXPERIMENT_PORT > /dev/null 2>&1; then
            echo -e "${GREEN}✓ AGA Experiment Tool is running on port $AGA_EXPERIMENT_PORT${NC}"
        else
            echo -e "${YELLOW}⚠ AGA Experiment Tool may not be ready yet${NC}"
            echo -e "  Check logs: tail -f $LOG_DIR/aga_experiment_hybrid.log"
        fi
    fi
fi

# 输出访问信息
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Service Endpoints${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$AGA_ONLY" = false ]; then
    echo -e "${GREEN}DKI Services:${NC}"
    echo -e "  Health:     http://localhost:$DKI_PORT/health"
    echo -e "  Chat API:   http://localhost:$DKI_PORT/v1/dki/chat"
    echo -e "  Stats:      http://localhost:$DKI_PORT/api/stats"
    echo -e "  Docs:       http://localhost:$DKI_PORT/docs"
    echo ""
fi

if [ "$DKI_ONLY" = false ]; then
    echo -e "${GREEN}AGA Services:${NC}"
    echo -e "  API Health: http://localhost:$AGA_PORT/health"
    echo -e "  API Docs:   http://localhost:$AGA_PORT/docs"
    if [ "$NO_EXPERIMENT" = false ]; then
        echo -e "  Experiment: http://localhost:$AGA_EXPERIMENT_PORT"
        echo -e "  (Password:  aga_experiment_2026)"
    fi
    echo ""
fi

echo -e "${GREEN}Session Management:${NC}"
if [ "$AGA_ONLY" = false ]; then
    echo -e "  DKI:        tmux attach -t dki"
fi
if [ "$DKI_ONLY" = false ]; then
    echo -e "  AGA API:    tmux attach -t aga_api"
    if [ "$NO_EXPERIMENT" = false ]; then
        echo -e "  AGA Exp:    tmux attach -t aga_exp"
    fi
fi
echo ""
echo -e "${GREEN}Logs:${NC}"
echo -e "  tail -f $LOG_DIR/dki_hybrid.log"
echo -e "  tail -f $LOG_DIR/aga_api_hybrid.log"
echo -e "  tail -f $LOG_DIR/aga_experiment_hybrid.log"
echo ""
echo -e "${BLUE}========================================${NC}"
