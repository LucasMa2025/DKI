import json

with open('experiment_results/experiment_exp_ddc8417fa14041c1_20260228_031423.json','r') as f:
    data = json.load(f)

print("Experiment ID:", data["experiment_id"])
print("Config:", data["config"])
print("Started:", data["started_at"])
print("Completed:", data["completed_at"])
print()

for mode in ['dki','rag','baseline']:
    m = data['results_by_mode'][mode]['metrics']
    samples = data['results_by_mode'][mode]['samples']
    print(f"=== {mode.upper()} ===")
    cnt = m.get("count", len(samples))
    print(f"  Samples: {cnt}")
    lat = m.get("latency", {})
    print(f"  Latency p50: {lat.get('p50',0):.0f}ms, p95: {lat.get('p95',0):.0f}ms, mean: {lat.get('mean',0):.0f}ms")
    mem = m.get("memory_usage", {})
    print(f"  Avg memories used: {mem.get('avg_memories_per_query',0):.2f}")
    rlen = m.get("response_length", {})
    print(f"  Response len: mean={rlen.get('mean',0):.0f}, std={rlen.get('std',0):.0f}, min={rlen.get('min',0)}, max={rlen.get('max',0)}")
    
    pref_count = sum(1 for s in samples if s.get('injection_info',{}).get('preference_text'))
    hist_count = sum(1 for s in samples if s.get('injection_info',{}).get('history_messages'))
    cache_hit = sum(1 for s in samples if s.get('cache_hit'))
    hist_tokens = [s.get('injection_info',{}).get('history_tokens',0) for s in samples]
    mem_used = [len(s.get('memories_used',[])) for s in samples]
    print(f"  Pref injection: {pref_count}/{len(samples)}")
    print(f"  History present: {hist_count}/{len(samples)}")
    print(f"  Cache hits: {cache_hit}/{len(samples)}")
    if hist_tokens:
        print(f"  History tokens: min={min(hist_tokens)}, max={max(hist_tokens)}, mean={sum(hist_tokens)/len(hist_tokens):.0f}")
    print(f"  Memories used: max={max(mem_used)}, mean={sum(mem_used)/len(mem_used):.2f}")
    
    # Unique session IDs
    sessions = set(s.get('sample_id','') for s in samples)
    print(f"  Unique sessions: {len(sessions)}")
    print()

print("=== AGGREGATED ===")
for mode, agg in data.get("aggregated_metrics", {}).items():
    print(f"  {mode}: {agg}")
