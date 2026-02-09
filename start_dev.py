#!/usr/bin/env python3
"""
DKI Development Server Startup Script

Starts both the backend API server and frontend dev server.
The frontend is an example Chat UI that demonstrates DKI integration.

Usage:
    python start_dev.py          # Start both servers
    python start_dev.py backend  # Start only backend
    python start_dev.py frontend # Start only frontend

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Example Chat UI (Vue3)                                 │
    │  └── Only passes user_id + raw input to DKI             │
    └─────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────┐
    │  DKI Plugin API (FastAPI)                               │
    │  ├── /v1/dki/chat - DKI enhanced chat                   │
    │  └── /v1/dki/info - DKI plugin status                   │
    └─────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────┐
    │  DKI Plugin Core                                        │
    │  ├── Config-driven adapter reads Chat UI's database     │
    │  ├── Preferences → K/V injection (Attention Hook)       │
    │  └── History → Suffix prompt                            │
    └─────────────────────────────────────────────────────────┘
"""

import subprocess
import sys
import os
import time
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


def main():
    """Main entry point."""
    processes = []
    
    # Parse arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("\n" + "=" * 60)
    print("  DKI - Dynamic KV Injection Development Environment")
    print("  Attention-Level User Memory Plugin for LLMs")
    print("=" * 60 + "\n")
    
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

