# DKI Quick Start Guide

快速开始使用 DKI (Dynamic KV Injection) 系统。

## 📋 Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ VRAM (for 7B models)

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd DKI

# Windows
scripts\setup.bat

# Linux/Mac
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Step 2: Start the System

```bash
# Windows
scripts\start.bat web

# Linux/Mac
./scripts/start.sh web
```

### Step 3: Open Web UI

Open http://localhost:8080 in your browser.

## 💻 Quick Python Example

```python
from dki import DKISystem

# Initialize DKI
dki = DKISystem()

# Add some memories about the user
session_id = "demo_user"

dki.add_memory(session_id, "I prefer vegetarian food")
dki.add_memory(session_id, "I'm allergic to seafood")
dki.add_memory(session_id, "I live in Beijing")
dki.add_memory(session_id, "I enjoy hiking on weekends")

# Chat - memories are automatically injected
response = dki.chat(
    query="Where should I eat tonight?",
    session_id=session_id
)

print(f"Response: {response.text}")
print(f"Injection Alpha: {response.gating_decision.alpha:.2f}")
print(f"Memories Used: {len(response.memories_used)}")
print(f"Latency: {response.latency_ms:.0f}ms")
```

## 🔄 Compare DKI vs RAG

```python
from dki import DKISystem, RAGSystem

dki = DKISystem()
rag = RAGSystem()

session_id = "comparison_test"
query = "Recommend a restaurant"

# Add same memories to both systems
memories = [
    "User prefers vegetarian food",
    "User is allergic to seafood",
]

for mem in memories:
    dki.add_memory(session_id, mem)
    rag.add_memory(session_id, mem)

# Compare responses
dki_response = dki.chat(query, session_id)
rag_response = rag.chat(query, session_id)

print("=== DKI Response ===")
print(f"Text: {dki_response.text}")
print(f"Latency: {dki_response.latency_ms:.0f}ms")
print(f"Alpha: {dki_response.gating_decision.alpha:.2f}")

print("\n=== RAG Response ===")
print(f"Text: {rag_response.text}")
print(f"Latency: {rag_response.latency_ms:.0f}ms")
```

## 🎛️ Control Injection Strength

```python
# Force specific alpha value
response = dki.chat(
    query="What should I do this weekend?",
    session_id=session_id,
    force_alpha=0.8  # Strong memory influence
)

# Disable injection entirely
response = dki.chat(
    query="What is 2+2?",
    session_id=session_id,
    allow_injection=False  # Pure LLM response
)
```

## 📊 Run Experiments

```python
from dki.experiment.runner import ExperimentRunner, ExperimentConfig
from dki.experiment.data_generator import ExperimentDataGenerator

# Generate test data
generator = ExperimentDataGenerator("./data")
generator.generate_all()

# Run comparison experiment
runner = ExperimentRunner()
config = ExperimentConfig(
    name="Quick Test",
    modes=["dki", "rag"],
    datasets=["memory_qa"],
    max_samples=20
)

results = runner.run_experiment(config)
print(f"DKI avg latency: {results['aggregated_metrics']['dki']['latency_p50']:.0f}ms")
print(f"RAG avg latency: {results['aggregated_metrics']['rag']['latency_p50']:.0f}ms")
```

## 🔧 Configuration Tips

### Use Different Models

Edit `config/config.yaml`:

```yaml
model:
  default_engine: "llama"  # Change from vllm to llama
  engines:
    llama:
      model_name: "meta-llama/Llama-3.2-3B-Instruct"
```

### Tune Gating Thresholds

```yaml
dki:
  gating:
    entropy_threshold: 0.3    # Lower = more likely to inject
    relevance_threshold: 0.5  # Lower = accept less similar memories
```

### Adjust Cache Settings

```yaml
dki:
  cache:
    max_size: 200       # Increase cache size
    strategy: "lru"     # Use simple LRU instead of weighted
    ttl_seconds: 7200   # Cache valid for 2 hours
```

## 🌐 REST API Usage

### Add Memory

```bash
curl -X POST http://localhost:8080/api/memory \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "content": "I like coffee"}'
```

### Chat

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What drink should I order?", "session_id": "test", "mode": "dki"}'
```

### Get Stats

```bash
curl http://localhost:8080/api/stats
```

## ❓ Troubleshooting

### CUDA Out of Memory

- Use smaller model (3B instead of 7B)
- Enable 8-bit quantization: `load_in_8bit: true`
- Reduce `max_model_len` in config

### Slow First Response

- First turn computes K/V cache
- Subsequent turns use cached K/V
- This is expected behavior

### No Memory Injection

Check gating decision:
```python
response = dki.chat(query, session_id)
print(response.gating_decision.reasoning)
# "High uncertainty but no relevant memory (open-ended question)"
```

Lower thresholds if needed:
```python
dki.gating.update_thresholds(
    entropy_threshold=0.3,
    relevance_threshold=0.5
)
```

## 📚 Next Steps

1. Read the full [README.md](README.md)
2. Explore the [DKI Paper](../AGAPaper/DKI_Paper_v1.md)
3. Run the full experiment suite
4. Try different model engines
5. Customize for your use case

---

Happy experimenting with DKI! 🚀
