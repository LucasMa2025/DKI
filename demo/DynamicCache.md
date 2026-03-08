class transformers.DynamicCache
<
source

> ( ddp_cache_data: collections.abc.Iterable[tuple[torch.Tensor | None, ...]] | None = Noneconfig: transformers.configuration_utils.PreTrainedConfig | None = Noneoffloading: bool = Falseoffload_only_non_sliding: bool = False )

Parameters

ddp_cache_data (Iterable[tuple[torch.Tensor, torch.Tensor]], optional) — It was originally added for compatibility with torch.distributed (DDP). In a nutshell, it is map(gather_map, zip(\*caches)), i.e. each item in the iterable contains the key and value states for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]). Note: it needs to be the 1st arg as well to work correctly
config (PreTrainedConfig, optional) — The config of the model for which this Cache will be used. If passed, it will be used to check for sliding or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to [batch_size, num_heads, min(seq_len, sliding_window), head_dim].
offloading (bool, optional, defaults to False) — Whether to perform offloading of the layers to cpu, to save GPU memory.
offload_only_non_sliding (bool, optional, defaults to False) — If offloading is True, this further decides if only the non-sliding layers will be offloaded (because usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).
A cache that grows dynamically as more tokens are generated. This is the default for generative models. It stores the key and value states as a list of CacheLayer, one for each layer. The expected shape for each tensor in the CacheLayers is [batch_size, num_heads, seq_len, head_dim]. If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the memory requirement of the cached tensors to [batch_size, num_heads, min(seq_len, sliding_window), head_dim].

See Cache for details on common methods that are implemented by all cache classes.

Example:

Copied
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

# Prepare a cache class and pass it to model's forward

past_key_values = DynamicCache(config=model.config)
outputs = model(\*\*inputs, past_key_values=past_key_values, use_cache=True)
outputs.past_key_values # access cache filled with key/values from generation
