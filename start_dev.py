#!/usr/bin/env python3
"""
DKI Development Server Startup Script

Starts both the backend API server and frontend dev server.
The frontend is an example Chat UI that demonstrates DKI integration.

Usage:
    python start_dev.py              # Start both servers
    python start_dev.py backend      # Start only backend
    python start_dev.py frontend     # Start only frontend
    python start_dev.py --redis      # Start with Redis enabled
    python start_dev.py --check-redis # Check Redis connection

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Example Chat UI (Vue3)                                 │
    │  └── Only passes user_id + raw input to DKI            │
    └─────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────┐
    │  DKI Plugin API (FastAPI)                               │
    │  ├── /v1/dki/chat - DKI enhanced chat                  │
    │  └── /v1/dki/info - DKI plugin status                  │
    └─────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────┐
    │  DKI Plugin Core                                        │
    │  ├── Config-driven adapter reads Chat UI's database    │
    │  ├── Preferences → K/V injection (Attention Hook)      │
    │  ├── History → Suffix prompt                           │
    │  └── L2 Cache → Redis (optional, for multi-instance)   │
    └─────────────────────────────────────────────────────────┘

Redis Integration:
    - L1 (Memory): Per-instance hot cache, < 1ms
    - L2 (Redis): Distributed warm cache, 1-5ms
    - Without Redis: cache hit rate = 70%/N (N = instances)
    - With Redis: cache hit rate = 70% (constant)
"""

import subprocess
import sys
import os
import time
import asyncio
from pathlib import Path

# Configuration
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8080
FRONTEND_PORT = 3000

# Paths
ROOT_DIR = Path(__file__).parent
UI_DIR = ROOT_DIR / "ui"


def start_backend():
    """Start the FastAPI backend server with DKI plugin."""
    print(f"🚀 Starting DKI backend server on http://localhost:{BACKEND_PORT}")
    print("   - DKI Plugin API: /v1/dki/chat")
    print("   - API Docs: /docs")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "dki.web.app:create_app",
            "--factory",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--reload",
        ],
        cwd=ROOT_DIR,
        env=env,
    )


def start_frontend():
    """Start the Vue3 example Chat UI dev server."""
    print(f"🎨 Starting example Chat UI on http://localhost:{FRONTEND_PORT}")
    print("   - This is an example app demonstrating DKI integration")
    print("   - Chat UI only passes user_id + raw input to DKI")
    
    # Check if node_modules exists
    if not (UI_DIR / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=UI_DIR,
            shell=True,
            check=True,
        )
    
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=UI_DIR,
        shell=True,
    )


async def check_redis():
    """Check Redis connection and display status."""
    print("\n🔍 Checking Redis connection...")
    
    try:
        from dki.cache.redis_client import DKIRedisClient, RedisConfig, REDIS_AVAILABLE
        
        if not REDIS_AVAILABLE:
            print("❌ Redis library not installed. Install with: pip install redis")
            return False
        
        # Load config
        import yaml
        config_path = ROOT_DIR / "config" / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            redis_config = RedisConfig.from_dict(config_data.get('redis', {}))
        else:
            redis_config = RedisConfig()
        
        if not redis_config.enabled:
            print("⚠️  Redis is disabled in config.yaml")
            print("   To enable, set redis.enabled: true in config/config.yaml")
            return False
        
        # Try to connect
        client = DKIRedisClient(redis_config)
        connected = await client.connect()
        
        if connected:
            info = await client.info()
            print(f"✅ Redis connected successfully!")
            print(f"   Host: {redis_config.host}:{redis_config.port}")
            print(f"   Version: {info.get('redis_version', 'unknown')}")
            print(f"   Memory: {info.get('used_memory_human', 'unknown')}")
            await client.close()
            return True
        else:
            print(f"❌ Failed to connect to Redis at {redis_config.host}:{redis_config.port}")
            return False
            
    except Exception as e:
        print(f"❌ Redis check failed: {e}")
        return False


def enable_redis_in_config():
    """Enable Redis in config.yaml."""
    import yaml
    
    config_path = ROOT_DIR / "config" / "config.yaml"
    
    if not config_path.exists():
        print("❌ config.yaml not found")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Enable Redis
    if 'redis' not in config_data:
        config_data['redis'] = {}
    config_data['redis']['enabled'] = True
    
    if 'preference_cache' not in config_data:
        config_data['preference_cache'] = {}
    config_data['preference_cache']['l2_enabled'] = True
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Redis enabled in config.yaml")
    print("   - redis.enabled: true")
    print("   - preference_cache.l2_enabled: true")


def main():
    """Main entry point."""
    processes = []
    
    # Parse arguments
    args = sys.argv[1:]
    mode = "all"
    
    # Handle special flags
    if "--check-redis" in args:
        asyncio.run(check_redis())
        return
    
    if "--redis" in args:
        enable_redis_in_config()
        args.remove("--redis")
    
    if args:
        mode = args[0]
    
    print("\n" + "=" * 60)
    print("  DKI - Dynamic KV Injection Development Environment")
    print("  Attention-Level User Memory Plugin for LLMs")
    print("=" * 60 + "\n")
    
    # Check Redis status
    redis_status = "disabled"
    try:
        import yaml
        config_path = ROOT_DIR / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            if config_data.get('redis', {}).get('enabled', False):
                redis_status = "enabled"
    except Exception:
        pass
    
    try:
        if mode in ("all", "backend"):
            backend_proc = start_backend()
            processes.append(backend_proc)
            time.sleep(2)  # Wait for backend to start
        
        if mode in ("all", "frontend"):
            frontend_proc = start_frontend()
            processes.append(frontend_proc)
        
        print("\n" + "=" * 60)
        print("🎉 DKI Development Servers Started!")
        print("=" * 60)
        
        if mode in ("all", "backend"):
            print(f"  Backend API:      http://localhost:{BACKEND_PORT}")
            print(f"  DKI Chat API:     http://localhost:{BACKEND_PORT}/v1/dki/chat")
            print(f"  API Docs:         http://localhost:{BACKEND_PORT}/docs")
        
        if mode in ("all", "frontend"):
            print(f"  Example Chat UI:  http://localhost:{FRONTEND_PORT}")
        
        print("=" * 60)
        print(f"\n📦 Cache Status:")
        print(f"   - L1 (Memory): enabled")
        print(f"   - L2 (Redis):  {redis_status}")
        if redis_status == "disabled":
            print(f"   💡 Enable Redis for multi-instance: python start_dev.py --redis")
        
        print("\n📝 Integration Notes:")
        print("   - Chat UI is an EXAMPLE app demonstrating DKI integration")
        print("   - DKI adapter reads Chat UI's database for preferences/history")
        print("   - Upstream apps only need to pass user_id + raw input")
        print("\nPress Ctrl+C to stop all servers")
        print("=" * 60 + "\n")
        
        # Wait for processes
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        print("✅ All servers stopped")


if __name__ == "__main__":
    main()
