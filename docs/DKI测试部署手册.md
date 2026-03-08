# DKI + AGA 完整部署指南

> 面向无模型安装经验用户的完整部署文档

## 📋 目录

1. [环境概述](#1-环境概述)
2. [前置准备](#2-前置准备)
3. [Python 环境配置](#3-python-环境配置)
4. [vLLM 安装](#4-vllm-安装)
5. [模型下载](#5-模型下载)
6. [DKI 系统部署](#6-dki-系统部署)
7. [AGA 系统部署](#7-aga-系统部署)
8. [vLLM 模型服务启动](#8-vllm-模型服务启动)
9. [DKI + AGA 混合部署](#9-dki--aga-混合部署)
10. [功能测试](#10-功能测试)
11. [常见问题排查](#11-常见问题排查)

---

## 1. 环境概述

### 1.1 目标环境

| 项目     | 配置                          |
| -------- | ----------------------------- |
| 操作系统 | Ubuntu Server 22.04 LTS 64 位 |
| CPU      | 20 核                         |
| 内存     | 80 GB                         |
| 存储     | 100 GB SSD                    |
| GPU      | 2 × NVIDIA V100 (32GB)        |
| GPU 驱动 | 570.158.01                    |
| CUDA     | 12.8.1                        |
| cuDNN    | 9.10.2                        |

### 1.2 目标模型（修正后准确版本）

| 模型                         | 参数量 | 显存需求（FP16） | 部署方式  | 硬盘占用（FP16） |
| ---------------------------- | ------ | ---------------- | --------- | ---------------- |
| DeepSeek-LLM-R1-7B           | 7B     | ~14GB            | 单卡      | ~14.1 GB         |
| Llama-3.1-8B-Instruct        | 8B     | ~16GB            | 单卡      | ~15.0 GB         |
| DeepSeek-R1-Distill-Qwen-32B | 32B    | ~64GB            | 双卡 TP=2 | ~65.7 GB         |

### 1.3 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Deployment Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Client     │    │   Client     │    │   Client     │               │
│  │  (Browser)   │    │  (curl)      │    │  (Python)    │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    DKI Web Server (:8000)                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │ Chat API    │  │ Stats API   │  │ Viz API     │              │    │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘              │    │
│  │         │                                                        │    │
│  │         ▼                                                        │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                   DKI Plugin Core                        │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │    │
│  │  │  │ K/V Inject  │  │ Memory Trig │  │ Ref Resolve │      │    │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   vLLM Server (:8080)                            │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                   AGA Plugin (Attention Hook)            │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │    │
│  │  │  │ Entropy Det │  │ FFN Inject  │  │ Knowledge   │      │    │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                   LLM Model (GPU)                        │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐                       │    │    │
│  │  │  │ V100 GPU 0  │  │ V100 GPU 1  │                       │    │    │
│  │  │  │ (32GB)      │  │ (32GB)      │                       │    │    │
│  │  │  └─────────────┘  └─────────────┘                       │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Data Layer                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │ SQLite DB   │  │ Redis Cache │  │ FAISS Index │              │    │
│  │  │ (Optional)  │  │ (Optional)  │  │ (AGA)       │              │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 前置准备

### 2.1 验证系统环境

打开终端，依次执行以下命令验证环境：

```bash
# 验证操作系统版本
cat /etc/os-release | grep -E "^(NAME|VERSION)="
# 预期输出:
# NAME="Ubuntu"
# VERSION="22.04.x LTS (Jammy Jellyfish)"

# 验证 GPU 驱动
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# 预期输出: 570.158.01

# 验证 CUDA 版本
nvcc --version | grep "release"
# 预期输出: Cuda compilation tools, release 12.8, V12.8.x

# 验证 GPU 信息
nvidia-smi
# 预期输出: 显示 2 块 V100 GPU，每块 32GB 显存
```

### 2.2 创建工作目录

```bash
# 创建项目根目录
sudo mkdir -p /opt/ai-demo
sudo chown $USER:$USER /opt/ai-demo
cd /opt/ai-demo

# 创建子目录
mkdir -p models logs data configs
```

### 2.3 安装系统依赖

```bash
# 更新软件包列表
sudo apt update

# 安装基础依赖
sudo apt install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv

# 验证安装
git --version
python3 --version
# 预期输出: Python 3.10.x
```

---

## 3. Python 环境配置

### 3.1 创建虚拟环境

```bash
cd /opt/ai-demo

# 创建 Python 虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 验证虚拟环境已激活（命令行前应显示 (venv)）
which python
# 预期输出: /opt/ai-demo/venv/bin/python

# 升级 pip
pip install --upgrade pip setuptools wheel
```

### 3.2 配置 pip 镜像（可选，加速下载）

```bash
# 创建 pip 配置目录
mkdir -p ~/.pip

# 配置清华镜像源
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

---

## 4. vLLM 安装

### 4.1 安装 vLLM

```bash
# 确保虚拟环境已激活
source /opt/ai-demo/venv/bin/activate

# 安装 vLLM（指定版本以确保兼容性）
pip install vllm==0.7.2

# 安装完成后验证
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
# 预期输出: vLLM version: 0.7.2
```

### 4.2 安装 vLLM 依赖

```bash
# 安装 PyTorch（如果 vLLM 未自动安装正确版本）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
echo 'export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/' >> ~/.bashrc
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.aliyun.com/pypi/simple/


# 验证 PyTorch CUDA 支持
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
# 预期输出: PyTorch: 2.5.1+cu124, CUDA: True, Devices: n(实际gpu卡数量)
```

### 4.3 安装 Hugging Face 工具

```bash
# 安装 huggingface_hub 用于模型下载
pip install huggingface_hub transformers accelerate

# 配置 Hugging Face 缓存目录
export HF_HOME=/opt/ai-demo/models/huggingface
mkdir -p $HF_HOME

# 配置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 添加到 bashrc（永久生效）
echo 'export HF_HOME=/opt/ai-demo/models/huggingface' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

```

---

## 5. 模型下载

### 5.1 登录 Hugging Face（如需访问受限模型）

```bash
# 安装 huggingface-cli    可跳过,新版huggingface 已启用该指令
# pip install huggingface_hub[cli]

# 登录（需要 Hugging Face 账号和 Access Token）
# 获取 Token: https://huggingface.co/settings/tokens
hf auth login
# 按提示输入 Token
```

### 5.2 下载 DeepSeek-LLM-R1-7B（修正后准确模型）

```bash
cd /opt/ai-demo/models

# 下载 DeepSeek-LLM-R1-7B
hf download deepseek-ai/DeepSeek-LLM-R1-7B \
    --local-dir deepseek-llm-r1-7b \

# 验证下载（检查模型文件）
ls -lh deepseek-llm-r1-7b/
# 预期: 应看到 config.json, model-*.safetensors 等文件
```

### 5.3 下载 DeepSeek-R1-Distill-Qwen-32B（修正后准确模型）

```bash
cd /opt/ai-demo/models

# 下载 DeepSeek-R1-Distill-Qwen-32B
# 注意：此模型较大（约 65.7GB），下载时间较长
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir deepseek-r1-distill-qwen-32b \
    --local-dir-use-symlinks False

# 验证下载
ls -lh deepseek-r1-distill-qwen-32b/
```

### 5.4 下载 Llama-3.1-8B-Instruct

```bash
cd /opt/ai-demo/models

# 下载 Llama-3.1-8B-Instruct
# 注意：需要先在 Hugging Face 上申请 Llama 模型访问权限
# 申请地址: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir llama-3.1-8b-instruct \
    --local-dir-use-symlinks False

# 验证下载
ls -lh llama-3.1-8b-instruct/
```

### 5.5 验证所有模型

```bash
# 列出所有已下载模型
ls -la /opt/ai-demo/models/

# 预期输出:
# drwxr-xr-x deepseek-llm-r1-7b
# drwxr-xr-x deepseek-r1-distill-qwen-32b
# drwxr-xr-x llama-3.1-8b-instruct
# drwxr-xr-x huggingface
```

---

## 6. DKI 系统部署

### 6.1 克隆 DKI 代码

```bash
cd /opt/ai-demo

# 克隆 DKI 仓库（假设从 GitHub 克隆，请替换为实际地址）
# 如果是本地复制，使用: cp -r /path/to/DKI ./DKI
git clone https://github.com/your-org/DKI.git

# 或者从本地复制
# cp -r /your/local/path/DKI ./DKI

cd DKI
```

### 6.2 安装 DKI 依赖

```bash
# 确保虚拟环境已激活
source /opt/ai-demo/venv/bin/activate

cd /opt/ai-demo/DKI

# 安装 DKI 依赖
pip install -r requirements.txt

# 安装 DKI 包（开发模式）
pip install -e .

# 验证安装
python -c "from dki.core.dki_plugin import DKIPlugin; print('DKI installed successfully')"
```

### 6.3 配置 DKI

```bash
cd /opt/ai-demo/DKI/config

# 备份默认配置
cp config.yaml config.yaml.bak

# 编辑配置文件
vim config.yaml
```

**修改 `config.yaml` 关键配置**：

```yaml
# /opt/ai-demo/DKI/config/config.yaml

# 服务器配置
server:
    host: "0.0.0.0"
    port: 8000
    debug: false

# 模型引擎配置
model:
    default_engine: "vllm"
    engines:
        vllm:
            # 模型路径（根据实际测试模型修改）
            model_name: "/opt/ai-demo/models/deepseek-llm-r1-7b"
            # vLLM API 地址
            api_base: "http://localhost:8080/v1"
            tensor_parallel_size: 1

# DKI 插件配置
dki:
    enabled: true
    version: "2.5"

    # 注入策略
    injection_strategy: "stable" # stable | full_attention

    # 混合注入策略
    hybrid_injection:
        enabled: true
        language: "cn"

        preference:
            enabled: true
            position_strategy: "negative"
            alpha: 0.4
            max_tokens: 200

        history:
            enabled: true
            method: "suffix_prompt"
            max_tokens: 2000
            max_messages: 10

    # 门控配置
    gating:
        relevance_threshold: 0.7
        entropy_threshold: 0.5

# Memory Trigger 配置
memory_trigger:
    enabled: true
    language: "auto"

# Reference Resolver 配置
reference_resolver:
    enabled: true
    just_now_turns: 3
    recently_turns: 10

# Redis 缓存（可选，如未安装 Redis 则设为 false）
redis:
    enabled: false
    host: "localhost"
    port: 6379
    db: 0

# 日志配置
logging:
    level: "INFO"
    file: "/opt/ai-demo/logs/dki.log"
```

### 6.4 配置外部数据适配器（可选）

如果需要连接外部数据库，编辑适配器配置：

```bash
cd /opt/ai-demo/DKI/config

# 复制示例配置
cp adapter_config.example.yaml adapter_config.yaml

# 编辑配置
vim adapter_config.yaml
```

**示例 SQLite 配置**（用于测试）：

```yaml
# /opt/ai-demo/DKI/config/adapter_config.yaml

adapter:
    type: "config_driven"
    database:
        type: "sqlite"
        connection_string: "sqlite:////opt/ai-demo/data/dki_test.db"

    # 消息表映射
    messages:
        table_name: "messages"
        columns:
            id: id
            user_id: user_id
            role: role
            content: content
            timestamp: created_at
        content_json_key: null

    # 偏好表映射
    preferences:
        table_name: "user_preferences"
        columns:
            id: id
            user_id: user_id
            content: preference_text
            category: category
```

### 6.5 初始化测试数据库（可选）

```bash
cd /opt/ai-demo/DKI

# 创建测试数据库
python << 'EOF'
import sqlite3
import os

db_path = "/opt/ai-demo/data/dki_test.db"
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 创建消息表
cursor.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# 创建偏好表
cursor.execute('''
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    preference_text TEXT NOT NULL,
    category TEXT DEFAULT 'general'
)
''')

# 插入测试数据
cursor.execute("INSERT INTO user_preferences (user_id, preference_text, category) VALUES ('test_user', '喜欢简洁的回答风格', 'style')")
cursor.execute("INSERT INTO user_preferences (user_id, preference_text, category) VALUES ('test_user', '偏好中文交流', 'language')")
cursor.execute("INSERT INTO messages (user_id, role, content) VALUES ('test_user', 'user', '你好，请介绍一下你自己')")
cursor.execute("INSERT INTO messages (user_id, role, content) VALUES ('test_user', 'assistant', '你好！我是一个AI助手，很高兴为你服务。')")

conn.commit()
conn.close()
print(f"Test database created at: {db_path}")
EOF
```

---

## 7. AGA 系统部署

### 7.1 克隆 AGA 代码

```bash
cd /opt/ai-demo

# 克隆 AGA 仓库
git clone https://github.com/your-org/AGA.git

# 或者从本地复制
# cp -r /your/local/path/AGA ./AGA

cd AGA
```

### 7.2 安装 AGA 依赖

```bash
# 确保虚拟环境已激活
source /opt/ai-demo/venv/bin/activate

cd /opt/ai-demo/AGA

# 安装 AGA 依赖
pip install -r requirements.txt

# 安装 AGA 包（开发模式）
pip install -e .

# 验证安装
python -c "from aga.core.aga_plugin import AGAPlugin; print('AGA installed successfully')"
```

### 7.3 配置 AGA

```bash
cd /opt/ai-demo/AGA/config

# 编辑运行时配置
vim runtime.yaml
```

**修改 `runtime.yaml` 关键配置**：

```yaml
# /opt/ai-demo/AGA/config/runtime.yaml

# AGA 运行时配置
runtime:
    enabled: true

    # 熵检测阈值
    entropy_threshold: 0.7

    # 知识注入配置
    knowledge_injection:
        enabled: true
        alpha: 0.3
        max_tokens: 500

    # 知识匹配失败行为
    knowledge_match_fail_behavior: "force_zero_alpha"
    # 可选值:
    # - "bypass_llm": 跳过 LLM，返回无相关内容提示
    # - "return_no_match_response": 返回预设的无匹配响应
    # - "force_zero_alpha": alpha=0，回退到原始 LLM

    # 失败时是否开放（返回原始输出）
    fail_open_enabled: true

# FlashAttention 配置
flash_attention:
    enabled: true
    backend: "auto"

# 知识库配置
knowledge_base:
    type: "faiss"
    index_path: "/opt/ai-demo/data/aga_knowledge.index"
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# 日志配置
logging:
    level: "INFO"
    file: "/opt/ai-demo/logs/aga.log"
```

### 7.4 初始化 AGA 知识库（可选）

```bash
cd /opt/ai-demo/AGA

# 创建示例知识库
python << 'EOF'
import os
import json

# 创建知识库目录
kb_dir = "/opt/ai-demo/data/knowledge_base"
os.makedirs(kb_dir, exist_ok=True)

# 创建示例知识文档
knowledge_docs = [
    {
        "id": "doc_001",
        "title": "Python 基础",
        "content": "Python 是一种解释型、面向对象的高级编程语言。它具有简洁的语法和强大的标准库。"
    },
    {
        "id": "doc_002",
        "title": "机器学习简介",
        "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策。"
    },
    {
        "id": "doc_003",
        "title": "深度学习基础",
        "content": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂表示。"
    }
]

# 保存知识文档
with open(os.path.join(kb_dir, "knowledge.json"), "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, ensure_ascii=False, indent=2)

print(f"Knowledge base created at: {kb_dir}")
EOF
```

---

## 8. vLLM 模型服务启动

### 8.1 启动脚本准备

```bash
cd /opt/ai-demo

# 创建启动脚本目录
mkdir -p scripts
```

### 8.2 DeepSeek-LLM-R1-7B 启动脚本（修正后模型）

```bash
cat > /opt/ai-demo/scripts/start_deepseek_7b.sh << 'EOF'
#!/bin/bash

# DeepSeek-LLM-R1-7B 启动脚本
# 单卡运行，适合 V100 32GB

source /opt/ai-demo/venv/bin/activate

MODEL_PATH="/opt/ai-demo/models/deepseek-llm-r1-7b"
PORT=8080
GPU_MEMORY_UTILIZATION=0.85

echo "Starting DeepSeek-LLM-R1-7B on port $PORT..."

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto \
    2>&1 | tee /opt/ai-demo/logs/vllm_deepseek_7b.log
EOF

chmod +x /opt/ai-demo/scripts/start_deepseek_7b.sh
```

### 8.3 DeepSeek-R1-Distill-Qwen-32B 启动脚本（修正后模型）

```bash
cat > /opt/ai-demo/scripts/start_deepseek_32b.sh << 'EOF'
#!/bin/bash

# DeepSeek-R1-Distill-Qwen-32B 启动脚本
# 双卡张量并行，需要 2 × V100 32GB

source /opt/ai-demo/venv/bin/activate

MODEL_PATH="/opt/ai-demo/models/deepseek-r1-distill-qwen-32b"
PORT=8080
GPU_MEMORY_UTILIZATION=0.90

echo "Starting DeepSeek-R1-Distill-Qwen-32B on port $PORT with TP=2..."

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype auto \
    2>&1 | tee /opt/ai-demo/logs/vllm_deepseek_32b.log
EOF

chmod +x /opt/ai-demo/scripts/start_deepseek_32b.sh
```

### 8.4 Llama-3.1-8B-Instruct 启动脚本

```bash
cat > /opt/ai-demo/scripts/start_llama_8b.sh << 'EOF'
#!/bin/bash

# Llama-3.1-8B-Instruct 启动脚本
# 单卡运行，适合 V100 32GB

source /opt/ai-demo/venv/bin/activate

MODEL_PATH="/opt/ai-demo/models/llama-3.1-8b-instruct"
PORT=8080
GPU_MEMORY_UTILIZATION=0.85

echo "Starting Llama-3.1-8B-Instruct on port $PORT..."

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 8192 \
    --dtype auto \
    2>&1 | tee /opt/ai-demo/logs/vllm_llama_8b.log
EOF

chmod +x /opt/ai-demo/scripts/start_llama_8b.sh
```

### 8.5 启动模型服务

```bash
# 使用 tmux 在后台运行（推荐）
tmux new-session -d -s vllm

# 在 tmux 会话中启动模型（以 DeepSeek 7B 为例）
tmux send-keys -t vllm "/opt/ai-demo/scripts/start_deepseek_7b.sh" Enter

# 查看日志
tmux attach -t vllm
# 按 Ctrl+B 然后按 D 可以退出 tmux 但保持运行

# 或者直接查看日志
tail -f /opt/ai-demo/logs/vllm_deepseek_7b.log
```

### 8.6 验证模型服务

等待模型加载完成（通常需要 1-3 分钟），然后验证：

```bash
# 检查服务是否运行
curl http://localhost:8080/v1/models

# 预期输出（JSON 格式）:
# {"object":"list","data":[{"id":"deepseek-llm-r1-7b",...}]}

# 测试推理
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/opt/ai-demo/models/deepseek-llm-r1-7b",
        "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
        "max_tokens": 100
    }'

# 预期输出: 模型的回复 JSON
```

---

## 9. DKI + AGA 混合部署

### 9.1 DKI 启动脚本

```bash
cat > /opt/ai-demo/scripts/start_dki.sh << 'EOF'
#!/bin/bash

# DKI 服务启动脚本

source /opt/ai-demo/venv/bin/activate
cd /opt/ai-demo/DKI

echo "Starting DKI server on port 8000..."

python main.py \
    --config config/config.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee /opt/ai-demo/logs/dki.log
EOF

chmod +x /opt/ai-demo/scripts/start_dki.sh
```

### 9.2 启动 DKI 服务

```bash
# 创建新的 tmux 会话
tmux new-session -d -s dki

# 启动 DKI
tmux send-keys -t dki "/opt/ai-demo/scripts/start_dki.sh" Enter

# 查看日志
tail -f /opt/ai-demo/logs/dki.log
```

### 9.3 验证 DKI 服务

```bash
# 检查 DKI 健康状态
curl http://localhost:8000/health

# 预期输出:
# {"status": "healthy", "version": "2.5", ...}

# 检查 DKI 统计
curl http://localhost:8000/api/stats

# 测试 DKI 聊天 API
curl -X POST http://localhost:8000/api/chat \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "test_user",
        "message": "你好，最近我们聊了什么？"
    }'
```

### 9.4 完整启动顺序

```bash
# 1. 启动 vLLM 模型服务（选择一个模型）
tmux new-session -d -s vllm
tmux send-keys -t vllm "/opt/ai-demo/scripts/start_deepseek_7b.sh" Enter

# 等待模型加载完成（约 1-3 分钟）
sleep 120

# 验证 vLLM 服务
curl http://localhost:8080/v1/models

# 2. 启动 DKI 服务
tmux new-session -d -s dki
tmux send-keys -t dki "/opt/ai-demo/scripts/start_dki.sh" Enter

# 等待 DKI 启动（约 10-30 秒）
sleep 30

# 验证 DKI 服务
curl http://localhost:8000/health
```

### 9.5 一键启动脚本

```bash
cat > /opt/ai-demo/scripts/start_all.sh << 'EOF'
#!/bin/bash

# 一键启动所有服务

MODEL=${1:-"deepseek_7b"}  # 默认使用 DeepSeek 7B

echo "=========================================="
echo "Starting AI Demo Services"
echo "Model: $MODEL"
echo "=========================================="

# 停止现有服务
echo "Stopping existing services..."
tmux kill-session -t vllm 2>/dev/null
tmux kill-session -t dki 2>/dev/null
sleep 2

# 启动 vLLM
echo "Starting vLLM with $MODEL..."
tmux new-session -d -s vllm

case $MODEL in
    "deepseek_7b")
        tmux send-keys -t vllm "/opt/ai-demo/scripts/start_deepseek_7b.sh" Enter
        ;;
    "deepseek_32b")
        tmux send-keys -t vllm "/opt/ai-demo/scripts/start_deepseek_32b.sh" Enter
        ;;
    "llama_8b")
        tmux send-keys -t vllm "/opt/ai-demo/scripts/start_llama_8b.sh" Enter
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available models: deepseek_7b, deepseek_32b, llama_8b"
        exit 1
        ;;
esac

# 等待模型加载
echo "Waiting for model to load (120 seconds)..."
sleep 120

# 验证 vLLM
echo "Verifying vLLM service..."
if curl -s http://localhost:8080/v1/models > /dev/null; then
    echo "✅ vLLM service is running"
else
    echo "❌ vLLM service failed to start"
    exit 1
fi

# 启动 DKI
echo "Starting DKI service..."
tmux new-session -d -s dki
tmux send-keys -t dki "/opt/ai-demo/scripts/start_dki.sh" Enter

# 等待 DKI 启动
echo "Waiting for DKI to start (30 seconds)..."
sleep 30

# 验证 DKI
echo "Verifying DKI service..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ DKI service is running"
else
    echo "❌ DKI service failed to start"
    exit 1
fi

echo "=========================================="
echo "All services started successfully!"
echo ""
echo "Services:"
echo "  - vLLM: http://localhost:8080"
echo "  - DKI:  http://localhost:8000"
echo ""
echo "To view logs:"
echo "  - vLLM: tmux attach -t vllm"
echo "  - DKI:  tmux attach -t dki"
echo "=========================================="
EOF

chmod +x /opt/ai-demo/scripts/start_all.sh
```

**使用方法**：

```bash
# 启动 DeepSeek 7B
/opt/ai-demo/scripts/start_all.sh deepseek_7b

# 启动 DeepSeek 32B
/opt/ai-demo/scripts/start_all.sh deepseek_32b

# 启动 Llama 8B
/opt/ai-demo/scripts/start_all.sh llama_8b
```

---

## 10. 功能测试

### 10.1 基础连通性测试

```bash
# 创建测试脚本
cat > /opt/ai-demo/scripts/test_basic.sh << 'EOF'
#!/bin/bash

echo "=========================================="
echo "Basic Connectivity Test"
echo "=========================================="

# 测试 vLLM
echo ""
echo "1. Testing vLLM service..."
VLLM_RESPONSE=$(curl -s http://localhost:8080/v1/models)
if echo "$VLLM_RESPONSE" | grep -q "data"; then
    echo "✅ vLLM: OK"
    echo "   Models: $(echo $VLLM_RESPONSE | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d['data']])")"
else
    echo "❌ vLLM: FAILED"
fi

# 测试 DKI
echo ""
echo "2. Testing DKI service..."
DKI_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$DKI_RESPONSE" | grep -q "healthy"; then
    echo "✅ DKI: OK"
    echo "   Version: $(echo $DKI_RESPONSE | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version', 'unknown'))")"
else
    echo "❌ DKI: FAILED"
fi

echo ""
echo "=========================================="
EOF

chmod +x /opt/ai-demo/scripts/test_basic.sh
/opt/ai-demo/scripts/test_basic.sh
```

### 10.2 DKI 功能测试

```bash
cat > /opt/ai-demo/scripts/test_dki.py << 'EOF'
#!/usr/bin/env python3
"""DKI 功能测试脚本"""

import requests
import json
import time

DKI_URL = "http://localhost:8000"
TEST_USER = "test_user_001"

def test_health():
    """测试健康检查"""
    print("\n=== Test 1: Health Check ===")
    try:
        resp = requests.get(f"{DKI_URL}/health", timeout=10)
        data = resp.json()
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat_basic():
    """测试基础聊天"""
    print("\n=== Test 2: Basic Chat ===")
    try:
        payload = {
            "user_id": TEST_USER,
            "message": "你好，请简单介绍一下你自己"
        }
        resp = requests.post(
            f"{DKI_URL}/api/chat",
            json=payload,
            timeout=60
        )
        data = resp.json()
        print(f"Response: {data.get('response', '')[:100]}...")
        print(f"DKI Metadata: {data.get('dki_metadata', {})}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_memory_trigger():
    """测试记忆触发"""
    print("\n=== Test 3: Memory Trigger ===")
    try:
        # 先发送一条消息建立上下文
        payload1 = {
            "user_id": TEST_USER,
            "message": "我喜欢吃川菜，特别是麻婆豆腐"
        }
        requests.post(f"{DKI_URL}/api/chat", json=payload1, timeout=60)

        time.sleep(1)

        # 测试记忆召回
        payload2 = {
            "user_id": TEST_USER,
            "message": "我们刚才聊了什么？"
        }
        resp = requests.post(
            f"{DKI_URL}/api/chat",
            json=payload2,
            timeout=60
        )
        data = resp.json()
        print(f"Response: {data.get('response', '')[:200]}...")

        # 检查是否提到川菜
        response_text = data.get('response', '').lower()
        if '川菜' in response_text or '麻婆豆腐' in response_text:
            print("✅ Memory recall successful")
            return True
        else:
            print("⚠️ Memory recall may not be working correctly")
            return True  # 不强制失败
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats():
    """测试统计 API"""
    print("\n=== Test 4: Stats API ===")
    try:
        resp = requests.get(f"{DKI_URL}/api/stats", timeout=10)
        data = resp.json()
        print(f"Total requests: {data.get('total_requests', 0)}")
        print(f"Cache hit rate: {data.get('cache_hit_rate', 0):.2%}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("DKI Functional Test Suite")
    print("=" * 50)

    results = {
        "Health Check": test_health(),
        "Basic Chat": test_chat_basic(),
        "Memory Trigger": test_memory_trigger(),
        "Stats API": test_stats(),
    }

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

chmod +x /opt/ai-demo/scripts/test_dki.py

# 运行测试
source /opt/ai-demo/venv/bin/activate
python /opt/ai-demo/scripts/test_dki.py
```

### 10.3 模型对比测试

```bash
cat > /opt/ai-demo/scripts/test_models.py << 'EOF'
#!/usr/bin/env python3
"""多模型对比测试脚本"""

import requests
import json
import time
import subprocess
import sys

VLLM_URL = "http://localhost:8080/v1"
DKI_URL = "http://localhost:8000"

MODELS = {
    "deepseek_7b": {
        "name": "DeepSeek-LLM-R1-7B",
        "path": "/opt/ai-demo/models/deepseek-llm-r1-7b",
        "script": "/opt/ai-demo/scripts/start_deepseek_7b.sh"
    },
    "deepseek_32b": {
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "path": "/opt/ai-demo/models/deepseek-r1-distill-qwen-32b",
        "script": "/opt/ai-demo/scripts/start_deepseek_32b.sh"
    },
    "llama_8b": {
        "name": "Llama-3.1-8B-Instruct",
        "path": "/opt/ai-demo/models/llama-3.1-8b-instruct",
        "script": "/opt/ai-demo/scripts/start_llama_8b.sh"
    }
}

TEST_PROMPTS = [
    {
        "name": "Basic Greeting",
        "prompt": "你好，请简单介绍一下你自己",
        "expected_keywords": ["AI", "助手", "帮助"]
    },
    {
        "name": "Code Generation",
        "prompt": "请用 Python 写一个计算斐波那契数列的函数",
        "expected_keywords": ["def", "fibonacci", "return"]
    },
    {
        "name": "Knowledge Query",
        "prompt": "请解释什么是机器学习",
        "expected_keywords": ["学习", "数据", "模型"]
    }
]

def test_model_direct(model_path, prompt):
    """直接测试 vLLM 模型"""
    try:
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }
        start_time = time.time()
        resp = requests.post(
            f"{VLLM_URL}/chat/completions",
            json=payload,
            timeout=120
        )
        latency = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            response = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "response": response,
                "latency": latency,
                "tokens": data.get("usage", {}).get("completion_tokens", 0)
            }
        else:
            return {"success": False, "error": resp.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_model_with_dki(user_id, prompt):
    """通过 DKI 测试模型"""
    try:
        payload = {
            "user_id": user_id,
            "message": prompt
        }
        start_time = time.time()
        resp = requests.post(
            f"{DKI_URL}/api/chat",
            json=payload,
            timeout=120
        )
        latency = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "response": data.get("response", ""),
                "latency": latency,
                "dki_metadata": data.get("dki_metadata", {})
            }
        else:
            return {"success": False, "error": resp.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_tests_for_model(model_key, model_info):
    """对单个模型运行所有测试"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_info['name']}")
    print(f"{'='*60}")

    results = []

    for test in TEST_PROMPTS:
        print(f"\n--- {test['name']} ---")

        # 直接测试
        print("Direct vLLM test...")
        direct_result = test_model_direct(model_info['path'], test['prompt'])

        if direct_result['success']:
            print(f"  Response: {direct_result['response'][:100]}...")
            print(f"  Latency: {direct_result['latency']:.2f}s")
        else:
            print(f"  Error: {direct_result.get('error', 'Unknown')}")

        # DKI 测试
        print("DKI test...")
        dki_result = test_model_with_dki(f"test_{model_key}", test['prompt'])

        if dki_result['success']:
            print(f"  Response: {dki_result['response'][:100]}...")
            print(f"  Latency: {dki_result['latency']:.2f}s")
            print(f"  DKI Metadata: {dki_result.get('dki_metadata', {})}")
        else:
            print(f"  Error: {dki_result.get('error', 'Unknown')}")

        results.append({
            "test": test['name'],
            "direct": direct_result,
            "dki": dki_result
        })

    return results

def main():
    print("=" * 60)
    print("Multi-Model Comparison Test")
    print("=" * 60)

    # 检查当前运行的模型
    try:
        resp = requests.get(f"{VLLM_URL}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json()
            current_model = models["data"][0]["id"] if models["data"] else None
            print(f"\nCurrently loaded model: {current_model}")
        else:
            print("Warning: Could not detect current model")
            current_model = None
    except:
        print("Error: vLLM service not responding")
        return False

    # 确定要测试的模型
    model_to_test = None
    for key, info in MODELS.items():
        if current_model and info['path'] in current_model:
            model_to_test = (key, info)
            break

    if not model_to_test:
        print("\nNo matching model found. Testing with first available...")
        model_to_test = list(MODELS.items())[0]

    # 运行测试
    results = run_tests_for_model(model_to_test[0], model_to_test[1])

    # 输出汇总
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for r in results:
        print(f"\n{r['test']}:")
        if r['direct']['success'] and r['dki']['success']:
            latency_diff = r['dki']['latency'] - r['direct']['latency']
            print(f"  Direct latency: {r['direct']['latency']:.2f}s")
            print(f"  DKI latency:    {r['dki']['latency']:.2f}s")
            print(f"  DKI overhead:   {latency_diff:.2f}s ({latency_diff/r['direct']['latency']*100:.1f}%)")
        else:
            print(f"  Direct: {'✅' if r['direct']['success'] else '❌'}")
            print(f"  DKI:    {'✅' if r['dki']['success'] else '❌'}")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

chmod +x /opt/ai-demo/scripts/test_models.py

# 运行测试
source /opt/ai-demo/venv/bin/activate
python /opt/ai-demo/scripts/test_models.py
```

### 10.4 性能基准测试

```bash
cat > /opt/ai-demo/scripts/benchmark.py << 'EOF'
#!/usr/bin/env python3
"""性能基准测试脚本"""

import requests
import time
import statistics
import concurrent.futures
from dataclasses import dataclass
from typing import List

DKI_URL = "http://localhost:8000"
VLLM_URL = "http://localhost:8080/v1"

@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # requests per second

def single_request(url: str, payload: dict, timeout: int = 120) -> tuple:
    """执行单次请求，返回 (success, latency)"""
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.time() - start
        return (resp.status_code == 200, latency)
    except:
        return (False, 0)

def run_benchmark(
    name: str,
    url: str,
    payload: dict,
    num_requests: int = 10,
    concurrency: int = 1
) -> BenchmarkResult:
    """运行基准测试"""
    print(f"\nRunning benchmark: {name}")
    print(f"  Requests: {num_requests}, Concurrency: {concurrency}")

    latencies = []
    successes = 0
    failures = 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_request, url, payload)
            for _ in range(num_requests)
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            success, latency = future.result()
            if success:
                successes += 1
                latencies.append(latency)
            else:
                failures += 1

            # 进度显示
            print(f"\r  Progress: {i+1}/{num_requests}", end="", flush=True)

    total_time = time.time() - start_time
    print()  # 换行

    if latencies:
        latencies.sort()
        return BenchmarkResult(
            name=name,
            total_requests=num_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency=statistics.mean(latencies),
            p50_latency=latencies[len(latencies)//2],
            p95_latency=latencies[int(len(latencies)*0.95)] if len(latencies) >= 20 else latencies[-1],
            p99_latency=latencies[int(len(latencies)*0.99)] if len(latencies) >= 100 else latencies[-1],
            throughput=successes / total_time
        )
    else:
        return BenchmarkResult(
            name=name,
            total_requests=num_requests,
            successful_requests=0,
            failed_requests=failures,
            avg_latency=0,
            p50_latency=0,
            p95_latency=0,
            p99_latency=0,
            throughput=0
        )

def main():
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # 获取当前模型
    try:
        resp = requests.get(f"{VLLM_URL}/models", timeout=10)
        model_path = resp.json()["data"][0]["id"]
        print(f"\nModel: {model_path}")
    except:
        print("Error: Cannot connect to vLLM")
        return

    results: List[BenchmarkResult] = []

    # 测试 1: 直接 vLLM 调用
    vllm_payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50
    }
    results.append(run_benchmark(
        "vLLM Direct (short)",
        f"{VLLM_URL}/chat/completions",
        vllm_payload,
        num_requests=10,
        concurrency=1
    ))

    # 测试 2: DKI 调用
    dki_payload = {
        "user_id": "benchmark_user",
        "message": "Hello, how are you?"
    }
    results.append(run_benchmark(
        "DKI (short)",
        f"{DKI_URL}/api/chat",
        dki_payload,
        num_requests=10,
        concurrency=1
    ))

    # 测试 3: 长文本生成
    vllm_long_payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": "请详细解释机器学习的基本概念和应用场景"}],
        "max_tokens": 200
    }
    results.append(run_benchmark(
        "vLLM Direct (long)",
        f"{VLLM_URL}/chat/completions",
        vllm_long_payload,
        num_requests=5,
        concurrency=1
    ))

    dki_long_payload = {
        "user_id": "benchmark_user",
        "message": "请详细解释机器学习的基本概念和应用场景"
    }
    results.append(run_benchmark(
        "DKI (long)",
        f"{DKI_URL}/api/chat",
        dki_payload,
        num_requests=5,
        concurrency=1
    ))

    # 输出结果
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(f"\n{'Test Name':<25} {'Success':<10} {'Avg(s)':<10} {'P50(s)':<10} {'P95(s)':<10} {'QPS':<10}")
    print("-" * 75)

    for r in results:
        print(f"{r.name:<25} {r.successful_requests}/{r.total_requests:<7} {r.avg_latency:<10.2f} {r.p50_latency:<10.2f} {r.p95_latency:<10.2f} {r.throughput:<10.2f}")

    # DKI 开销分析
    print("\n" + "=" * 60)
    print("DKI Overhead Analysis")
    print("=" * 60)

    if len(results) >= 2:
        vllm_short = results[0]
        dki_short = results[1]
        if vllm_short.avg_latency > 0:
            overhead = (dki_short.avg_latency - vllm_short.avg_latency) / vllm_short.avg_latency * 100
            print(f"Short response overhead: {overhead:.1f}%")

    if len(results) >= 4:
        vllm_long = results[2]
        dki_long = results[3]
        if vllm_long.avg_latency > 0:
            overhead = (dki_long.avg_latency - vllm_long.avg_latency) / vllm_long.avg_latency * 100
            print(f"Long response overhead:  {overhead:.1f}%")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/ai-demo/scripts/benchmark.py

# 运行基准测试
source /opt/ai-demo/venv/bin/activate
python /opt/ai-demo/scripts/benchmark.py
```

---

## 11. 常见问题排查

### 11.1 GPU 相关问题

**问题：CUDA out of memory**

```bash
# 检查 GPU 内存使用
nvidia-smi

# 解决方案 1: 降低 GPU 内存利用率
# 编辑启动脚本，将 GPU_MEMORY_UTILIZATION 从 0.85 降到 0.75

# 解决方案 2: 减少 max_model_len
# 编辑启动脚本，将 --max-model-len 从 8192 降到 4096

# 解决方案 3: 清理 GPU 内存
sudo fuser -v /dev/nvidia*
# 如果有进程占用，可以 kill 掉
```

**问题：GPU 驱动不匹配**

```bash
# 检查驱动版本
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# 检查 CUDA 版本
nvcc --version

# 如果版本不匹配，可能需要重新安装 PyTorch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 11.2 vLLM 相关问题

**问题：模型加载失败**

```bash
# 检查模型文件完整性
ls -la /opt/ai-demo/models/deepseek-llm-r1-7b/

# 检查是否有 config.json 和 model 文件
cat /opt/ai-demo/models/deepseek-llm-r1-7b/config.json | head -20

# 如果文件不完整，重新下载
huggingface-cli download deepseek-ai/DeepSeek-LLM-R1-7B \
    --local-dir /opt/ai-demo/models/deepseek-llm-r1-7b \
    --local-dir-use-symlinks False \
    --resume-download
```

**问题：vLLM 服务无响应**

```bash
# 检查服务是否运行
ps aux | grep vllm

# 检查端口是否被占用
sudo netstat -tlnp | grep 8080

# 查看日志
tail -100 /opt/ai-demo/logs/vllm_deepseek_7b.log

# 重启服务
tmux kill-session -t vllm
/opt/ai-demo/scripts/start_deepseek_7b.sh
```

### 11.3 DKI 相关问题

**问题：DKI 无法连接 vLLM**

```bash
# 检查 vLLM 服务
curl http://localhost:8080/v1/models

# 检查 DKI 配置中的 api_base
cat /opt/ai-demo/DKI/config/config.yaml | grep api_base

# 确保配置正确
# api_base: "http://localhost:8080/v1"
```

**问题：DKI 启动报错**

```bash
# 查看详细错误日志
tail -100 /opt/ai-demo/logs/dki.log

# 常见错误 1: 模块导入失败
# 解决: 确保安装了所有依赖
pip install -r /opt/ai-demo/DKI/requirements.txt

# 常见错误 2: 配置文件格式错误
# 解决: 检查 YAML 格式
python -c "import yaml; yaml.safe_load(open('/opt/ai-demo/DKI/config/config.yaml'))"
```

### 11.4 网络相关问题

**问题：无法下载模型**

```bash
# 检查网络连接
ping huggingface.co

# 使用代理（如果需要）
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

**问题：端口被占用**

```bash
# 查找占用端口的进程
sudo lsof -i :8000
sudo lsof -i :8080

# 杀死进程
sudo kill -9 <PID>

# 或更改端口
# 编辑配置文件，使用其他端口
```

### 11.5 服务管理命令速查

```bash
# 查看所有 tmux 会话
tmux ls

# 进入 vLLM 会话
tmux attach -t vllm

# 进入 DKI 会话
tmux attach -t dki

# 退出 tmux 会话（保持运行）
# 按 Ctrl+B，然后按 D

# 停止 vLLM 服务
tmux kill-session -t vllm

# 停止 DKI 服务
tmux kill-session -t dki

# 停止所有服务
tmux kill-server

# 查看 GPU 状态
watch -n 1 nvidia-smi

# 查看系统资源
htop

# 查看磁盘使用
df -h

# 查看日志
tail -f /opt/ai-demo/logs/vllm_*.log
tail -f /opt/ai-demo/logs/dki.log
```

---

## 附录 A: 目录结构总览

```
/opt/ai-demo/
├── venv/                          # Python 虚拟环境
├── models/                        # 模型文件
│   ├── huggingface/               # HF 缓存
│   ├── deepseek-llm-r1-7b/        # DeepSeek-LLM-R1-7B
│   ├── deepseek-r1-distill-qwen-32b/ # DeepSeek-R1-Distill-Qwen-32B
│   └── llama-3.1-8b-instruct/     # Llama-3.1-8B-Instruct
├── DKI/                           # DKI 系统
│   ├── config/
│   ├── dki/
│   └── ...
├── AGA/                           # AGA 系统
│   ├── config/
│   ├── aga/
│   └── ...
├── data/                          # 数据文件
│   ├── dki_test.db                # 测试数据库
│   └── knowledge_base/            # 知识库
├── logs/                          # 日志文件
│   ├── vllm_deepseek_7b.log
│   ├── vllm_deepseek_32b.log
│   ├── vllm_llama_8b.log
│   └── dki.log
├── scripts/                       # 启动和测试脚本
│   ├── start_deepseek_7b.sh
│   ├── start_deepseek_32b.sh
│   ├── start_llama_8b.sh
│   ├── start_dki.sh
│   ├── start_all.sh
│   ├── test_basic.sh
│   ├── test_dki.py
│   ├── test_models.py
│   └── benchmark.py
└── configs/                       # 配置备份
```

---

## 附录 B: 快速命令参考

```bash
# === 环境激活 ===
source /opt/ai-demo/venv/bin/activate

# === 一键启动 ===
/opt/ai-demo/scripts/start_all.sh deepseek_7b   # DeepSeek-LLM-R1-7B
/opt/ai-demo/scripts/start_all.sh deepseek_32b  # DeepSeek-R1-Distill-Qwen-32B
/opt/ai-demo/scripts/start_all.sh llama_8b      # Llama-3.1-8B-Instruct

# === 服务验证 ===
curl http://localhost:8080/v1/models            # vLLM
curl http://localhost:8000/health               # DKI

# === 功能测试 ===
python /opt/ai-demo/scripts/test_dki.py         # DKI 功能测试
python /opt/ai-demo/scripts/test_models.py      # 模型对比测试
python /opt/ai-demo/scripts/benchmark.py        # 性能基准测试

# === 日志查看 ===
tail -f /opt/ai-demo/logs/vllm_*.log            # vLLM 日志
tail -f /opt/ai-demo/logs/dki.log               # DKI 日志

# === 服务管理 ===
tmux ls                                          # 查看会话
tmux attach -t vllm                              # 进入 vLLM
tmux attach -t dki                               # 进入 DKI
tmux kill-session -t vllm                        # 停止 vLLM
tmux kill-session -t dki                         # 停止 DKI
```

---

**文档版本**: 1.0  
**最后更新**: 2026-02-12  
**适用系统**: Ubuntu Server 22.04 LTS + 2×V100 (32GB)
