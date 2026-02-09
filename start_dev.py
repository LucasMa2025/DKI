#!/usr/bin/env python3
"""
Development Server Startup Script
Starts both the backend API server and frontend dev server

Usage:
    python start_dev.py          # Start both servers
    python start_dev.py backend  # Start only backend
    python start_dev.py frontend # Start only frontend
"""

import subprocess
import sys
import os
import signal
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
    """Start the FastAPI backend server."""
    print(f"🚀 Starting backend server on http://localhost:{BACKEND_PORT}")
    
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
    """Start the Vue3 frontend dev server."""
    print(f"🎨 Starting frontend server on http://localhost:{FRONTEND_PORT}")
    
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
    
    try:
        if mode in ("all", "backend"):
            backend_proc = start_backend()
            processes.append(backend_proc)
            time.sleep(2)  # Wait for backend to start
        
        if mode in ("all", "frontend"):
            frontend_proc = start_frontend()
            processes.append(frontend_proc)
        
        print("\n" + "=" * 50)
        print("🎉 DKI Development Servers Started!")
        print("=" * 50)
        
        if mode in ("all", "backend"):
            print(f"  Backend API:  http://localhost:{BACKEND_PORT}")
            print(f"  API Docs:     http://localhost:{BACKEND_PORT}/docs")
        
        if mode in ("all", "frontend"):
            print(f"  Frontend UI:  http://localhost:{FRONTEND_PORT}")
        
        print("=" * 50)
        print("Press Ctrl+C to stop all servers")
        print("=" * 50 + "\n")
        
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
