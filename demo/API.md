The **DynamicCache** API is part of Hugging Face’s **Transformers** library, specifically introduced for efficient **key-value caching** in **decoder-only** language models (like GPT-style models). It dynamically manages the cache size during generation, avoiding recomputation of past key-value states.

### 🔍 **Key Details from HF Docs**

-   **Purpose**: Optimizes memory usage by dynamically growing the cache instead of pre-allocating fixed-size tensors.
-   **Usage**: Automatically used in models like `LlamaForCausalLM`, `GPT2LMHeadModel`, etc., when `use_cache=True` (default in `generate()`).
-   **API Reference**:  
    [DynamicCache Docs](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) (check the latest `transformers` docs).

### 📌 **Code Example**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is", return_tensors="pt")
outputs = model.generate(**inputs, use_cache=True)  # DynamicCache used internally
```

### ⚙️ **How It Works**

-   Instead of static caches (`DynamicCache` replaces older `past_key_values` tuples), it appends new states to a growing list.
-   Reduces **GPU memory spikes** compared to static caches for long sequences.

### 🚨 **Notes**

-   Requires `transformers >= 4.36.0` (check your version via `pip show transformers`).
-   For custom implementations, inherit from `transformers.cache_utils.DynamicCache`.

Would you like a deeper dive into its source code or benchmarking against static caches?
