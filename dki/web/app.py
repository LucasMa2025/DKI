"""
FastAPI Web Application for DKI System
Provides REST API and Web UI for testing
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from loguru import logger

from dki.core.dki_system import DKISystem
from dki.core.rag_system import RAGSystem
from dki.database.connection import get_db, DatabaseManager
from dki.database.repository import SessionRepository, MemoryRepository, ExperimentRepository
from dki.experiment.runner import ExperimentRunner, ExperimentConfig
from dki.experiment.data_generator import ExperimentDataGenerator
from dki.config.config_loader import ConfigLoader


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "dki"  # dki, rag, baseline
    force_alpha: Optional[float] = None
    max_new_tokens: int = 256
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    mode: str
    session_id: str
    latency_ms: float
    memories_used: List[Dict[str, Any]]
    alpha: Optional[float] = None
    cache_hit: bool = False
    metadata: Dict[str, Any] = {}


class MemoryRequest(BaseModel):
    session_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    memory_id: str
    session_id: str
    content: str


class ExperimentRequest(BaseModel):
    name: str
    description: str = ""
    modes: List[str] = ["dki", "rag", "baseline"]
    datasets: List[str] = ["persona_chat", "memory_qa"]
    max_samples: int = 50


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    config = ConfigLoader().config
    
    app = FastAPI(
        title="DKI System",
        description="Dynamic KV Injection - Attention-Level Memory Augmentation",
        version="1.0.0",
    )
    
    # Initialize systems (lazy)
    _systems = {}
    
    def get_dki_system() -> DKISystem:
        if 'dki' not in _systems:
            _systems['dki'] = DKISystem()
        return _systems['dki']
    
    def get_rag_system() -> RAGSystem:
        if 'rag' not in _systems:
            _systems['rag'] = RAGSystem()
        return _systems['rag']
    
    # API Routes
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Chat endpoint supporting DKI, RAG, and baseline modes.
        """
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        try:
            if request.mode == "dki":
                dki = get_dki_system()
                response = dki.chat(
                    query=request.query,
                    session_id=session_id,
                    force_alpha=request.force_alpha,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                return ChatResponse(
                    response=response.text,
                    mode="dki",
                    session_id=session_id,
                    latency_ms=response.latency_ms,
                    memories_used=[m.to_dict() for m in response.memories_used],
                    alpha=response.gating_decision.alpha,
                    cache_hit=response.cache_hit,
                    metadata=response.metadata,
                )
                
            elif request.mode == "rag":
                rag = get_rag_system()
                response = rag.chat(
                    query=request.query,
                    session_id=session_id,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                return ChatResponse(
                    response=response.text,
                    mode="rag",
                    session_id=session_id,
                    latency_ms=response.latency_ms,
                    memories_used=[m.to_dict() for m in response.memories_used],
                    metadata=response.metadata,
                )
                
            else:  # baseline
                dki = get_dki_system()
                output = dki.model.generate(
                    prompt=request.query,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                return ChatResponse(
                    response=output.text,
                    mode="baseline",
                    session_id=session_id,
                    latency_ms=output.latency_ms,
                    memories_used=[],
                )
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/memory", response_model=MemoryResponse)
    async def add_memory(request: MemoryRequest):
        """Add memory to both DKI and RAG systems."""
        try:
            dki = get_dki_system()
            rag = get_rag_system()
            
            memory_id = dki.add_memory(
                session_id=request.session_id,
                content=request.content,
                metadata=request.metadata,
            )
            
            # Also add to RAG for comparison
            rag.add_memory(
                session_id=request.session_id,
                content=request.content,
                memory_id=memory_id,
                metadata=request.metadata,
            )
            
            return MemoryResponse(
                memory_id=memory_id,
                session_id=request.session_id,
                content=request.content,
            )
            
        except Exception as e:
            logger.error(f"Add memory error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/memories/{session_id}")
    async def get_memories(session_id: str):
        """Get all memories for a session."""
        try:
            db_manager = DatabaseManager()
            with db_manager.session_scope() as db:
                memory_repo = MemoryRepository(db)
                memories = memory_repo.get_by_session(session_id)
                return {
                    "session_id": session_id,
                    "memories": [m.to_dict() for m in memories],
                }
        except Exception as e:
            logger.error(f"Get memories error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/search")
    async def search_memories(query: str, session_id: Optional[str] = None, top_k: int = 5):
        """Search memories."""
        try:
            dki = get_dki_system()
            results = dki.search_memories(query, top_k=top_k)
            return {
                "query": query,
                "results": [r.to_dict() for r in results],
            }
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/stats")
    async def get_stats():
        """Get system statistics."""
        try:
            dki = get_dki_system()
            rag = get_rag_system()
            return {
                "dki": dki.get_stats(),
                "rag": rag.get_stats(),
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/experiment/generate-data")
    async def generate_experiment_data():
        """Generate experiment data."""
        try:
            generator = ExperimentDataGenerator("./data")
            generator.generate_all()
            generator.generate_alpha_sensitivity_data()
            return {"status": "success", "message": "Experiment data generated"}
        except Exception as e:
            logger.error(f"Generate data error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/experiment/run")
    async def run_experiment(request: ExperimentRequest):
        """Run an experiment."""
        try:
            runner = ExperimentRunner()
            config = ExperimentConfig(
                name=request.name,
                description=request.description,
                modes=request.modes,
                datasets=request.datasets,
                max_samples=request.max_samples,
            )
            results = runner.run_experiment(config)
            return results
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/experiments")
    async def list_experiments():
        """List all experiments."""
        try:
            db_manager = DatabaseManager()
            with db_manager.session_scope() as db:
                exp_repo = ExperimentRepository(db)
                experiments = exp_repo.list_all()
                return {
                    "experiments": [e.to_dict() for e in experiments],
                }
        except Exception as e:
            logger.error(f"List experiments error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Web UI Route
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main web UI."""
        return get_index_html()
    
    return app


def get_index_html() -> str:
    """Return the main HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DKI System - Dynamic KV Injection</title>
    <style>
        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-tertiary: #252540;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 2fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 20px;
        }
        
        .panel-title {
            font-size: 1.2rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--accent-primary);
            border-radius: 2px;
        }
        
        /* Mode Selector */
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .mode-btn {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .mode-btn:hover {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }
        
        .mode-btn.active {
            border-color: var(--accent-primary);
            background: var(--accent-primary);
            color: white;
        }
        
        .mode-btn.dki.active { background: var(--accent-primary); }
        .mode-btn.rag.active { background: var(--accent-success); }
        .mode-btn.baseline.active { background: var(--accent-warning); }
        
        /* Chat Area */
        .chat-container {
            height: 500px;
            overflow-y: auto;
            margin-bottom: 15px;
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
        }
        
        .message.user {
            background: var(--accent-primary);
            margin-left: auto;
        }
        
        .message.assistant {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
        }
        
        .message-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 8px;
            display: flex;
            gap: 15px;
        }
        
        .meta-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* Input Area */
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
        }
        
        .input-area input:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        
        .input-area button {
            padding: 12px 24px;
            background: var(--accent-primary);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1rem;
            transition: background 0.3s;
        }
        
        .input-area button:hover {
            background: var(--accent-secondary);
        }
        
        /* Memory Panel */
        .memory-input {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .memory-input textarea {
            padding: 12px;
            border: 2px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 8px;
            font-family: inherit;
            resize: vertical;
            min-height: 80px;
        }
        
        .memory-input textarea:focus {
            outline: none;
            border-color: var(--accent-primary);
        }
        
        .memory-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .memory-item {
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }
        
        /* Stats Panel */
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-card {
            padding: 15px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-primary);
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        /* Alpha Slider */
        .alpha-control {
            margin: 15px 0;
        }
        
        .alpha-control label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-secondary);
        }
        
        .alpha-control input[type="range"] {
            width: 100%;
            -webkit-appearance: none;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
        }
        
        .alpha-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--accent-primary);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .alpha-value {
            text-align: center;
            font-size: 1.2rem;
            color: var(--accent-primary);
            margin-top: 5px;
        }
        
        /* Experiment Panel */
        .experiment-btn {
            width: 100%;
            padding: 12px;
            margin-bottom: 10px;
            background: var(--bg-tertiary);
            border: 2px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        .experiment-btn:hover {
            border-color: var(--accent-primary);
            background: var(--accent-primary);
        }
        
        /* Loading */
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid var(--bg-tertiary);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 DKI System</h1>
            <p>Dynamic KV Injection - Attention-Level Memory Augmentation</p>
        </header>
        
        <div class="main-grid">
            <!-- Memory Panel -->
            <div class="panel">
                <div class="panel-title">Memory Store</div>
                
                <div class="memory-input">
                    <textarea id="memoryInput" placeholder="Enter memory content..."></textarea>
                    <button class="experiment-btn" onclick="addMemory()">Add Memory</button>
                </div>
                
                <div class="panel-title" style="margin-top: 20px;">Memories</div>
                <div class="memory-list" id="memoryList">
                    <div class="memory-item">No memories yet. Add some above!</div>
                </div>
                
                <div class="alpha-control">
                    <label>Force Alpha (DKI mode):</label>
                    <input type="range" id="alphaSlider" min="0" max="100" value="50" oninput="updateAlpha()">
                    <div class="alpha-value" id="alphaValue">α = 0.50</div>
                </div>
            </div>
            
            <!-- Chat Panel -->
            <div class="panel">
                <div class="panel-title">Conversation</div>
                
                <div class="mode-selector">
                    <button class="mode-btn dki active" onclick="setMode('dki')">DKI Mode</button>
                    <button class="mode-btn rag" onclick="setMode('rag')">RAG Mode</button>
                    <button class="mode-btn baseline" onclick="setMode('baseline')">Baseline</button>
                </div>
                
                <div class="chat-container" id="chatContainer">
                    <div class="message assistant">
                        <div>Welcome! I'm the DKI system. Add some memories and start chatting!</div>
                        <div class="message-meta">
                            <span class="meta-item">Mode: DKI</span>
                        </div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px; color: var(--text-secondary);">Generating response...</p>
                </div>
                
                <div class="input-area">
                    <input type="text" id="queryInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <!-- Stats & Experiments Panel -->
            <div class="panel">
                <div class="panel-title">Statistics</div>
                
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="statLatency">0</div>
                        <div class="stat-label">Avg Latency (ms)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statMemories">0</div>
                        <div class="stat-label">Memories Used</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statCacheHit">0%</div>
                        <div class="stat-label">Cache Hit Rate</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="statAlpha">-</div>
                        <div class="stat-label">Avg Alpha</div>
                    </div>
                </div>
                
                <div class="panel-title" style="margin-top: 20px;">Experiments</div>
                
                <button class="experiment-btn" onclick="generateData()">Generate Test Data</button>
                <button class="experiment-btn" onclick="runExperiment()">Run Comparison</button>
                <button class="experiment-btn" onclick="runAlphaSensitivity()">α Sensitivity Test</button>
                
                <div class="panel-title" style="margin-top: 20px;">Session</div>
                <div style="font-size: 0.85rem; color: var(--text-secondary);">
                    Session ID: <span id="sessionId">-</span>
                </div>
                <button class="experiment-btn" style="margin-top: 10px;" onclick="newSession()">New Session</button>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentMode = 'dki';
        let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        let useForceAlpha = false;
        let forceAlphaValue = 0.5;
        let stats = { latencies: [], memoriesUsed: 0, cacheHits: 0, totalQueries: 0, alphas: [] };
        
        document.getElementById('sessionId').textContent = sessionId;
        
        // Mode selection
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.mode-btn.' + mode).classList.add('active');
        }
        
        // Alpha control
        function updateAlpha() {
            const slider = document.getElementById('alphaSlider');
            forceAlphaValue = slider.value / 100;
            document.getElementById('alphaValue').textContent = 'α = ' + forceAlphaValue.toFixed(2);
        }
        
        // Add memory
        async function addMemory() {
            const content = document.getElementById('memoryInput').value.trim();
            if (!content) return;
            
            try {
                const response = await fetch('/api/memory', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, content: content })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    addMemoryToList(data.content);
                    document.getElementById('memoryInput').value = '';
                }
            } catch (error) {
                console.error('Add memory error:', error);
            }
        }
        
        function addMemoryToList(content) {
            const list = document.getElementById('memoryList');
            if (list.children[0].textContent.includes('No memories')) {
                list.innerHTML = '';
            }
            const item = document.createElement('div');
            item.className = 'memory-item';
            item.textContent = content;
            list.appendChild(item);
        }
        
        // Send message
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;
            
            // Add user message
            addMessage('user', query);
            input.value = '';
            
            // Show loading
            document.getElementById('loading').classList.add('active');
            
            try {
                const requestBody = {
                    query: query,
                    session_id: sessionId,
                    mode: currentMode,
                    max_new_tokens: 256,
                    temperature: 0.7
                };
                
                // Add force alpha for DKI mode if checkbox would be checked
                if (currentMode === 'dki') {
                    requestBody.force_alpha = forceAlphaValue;
                }
                
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                
                // Add assistant message
                addMessage('assistant', data.response, data);
                
                // Update stats
                updateStats(data);
                
            } catch (error) {
                console.error('Chat error:', error);
                addMessage('assistant', 'Error: ' + error.message, {});
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }
        
        function addMessage(role, content, meta = {}) {
            const container = document.getElementById('chatContainer');
            const msg = document.createElement('div');
            msg.className = 'message ' + role;
            
            let metaHtml = '';
            if (role === 'assistant' && meta.mode) {
                metaHtml = '<div class="message-meta">';
                metaHtml += '<span class="meta-item">Mode: ' + meta.mode + '</span>';
                if (meta.latency_ms) {
                    metaHtml += '<span class="meta-item">Latency: ' + meta.latency_ms.toFixed(0) + 'ms</span>';
                }
                if (meta.alpha !== undefined && meta.alpha !== null) {
                    metaHtml += '<span class="meta-item">α: ' + meta.alpha.toFixed(2) + '</span>';
                }
                if (meta.cache_hit) {
                    metaHtml += '<span class="meta-item">Cache Hit ✓</span>';
                }
                if (meta.memories_used && meta.memories_used.length > 0) {
                    metaHtml += '<span class="meta-item">Memories: ' + meta.memories_used.length + '</span>';
                }
                metaHtml += '</div>';
            }
            
            msg.innerHTML = '<div>' + content + '</div>' + metaHtml;
            container.appendChild(msg);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateStats(data) {
            stats.totalQueries++;
            if (data.latency_ms) {
                stats.latencies.push(data.latency_ms);
            }
            if (data.memories_used) {
                stats.memoriesUsed += data.memories_used.length;
            }
            if (data.cache_hit) {
                stats.cacheHits++;
            }
            if (data.alpha !== undefined && data.alpha !== null) {
                stats.alphas.push(data.alpha);
            }
            
            // Update display
            const avgLatency = stats.latencies.length > 0 
                ? stats.latencies.reduce((a, b) => a + b) / stats.latencies.length 
                : 0;
            document.getElementById('statLatency').textContent = avgLatency.toFixed(0);
            document.getElementById('statMemories').textContent = stats.memoriesUsed;
            document.getElementById('statCacheHit').textContent = 
                stats.totalQueries > 0 ? ((stats.cacheHits / stats.totalQueries) * 100).toFixed(0) + '%' : '0%';
            document.getElementById('statAlpha').textContent = 
                stats.alphas.length > 0 
                    ? (stats.alphas.reduce((a, b) => a + b) / stats.alphas.length).toFixed(2) 
                    : '-';
        }
        
        // Session management
        function newSession() {
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            document.getElementById('sessionId').textContent = sessionId;
            document.getElementById('chatContainer').innerHTML = '';
            document.getElementById('memoryList').innerHTML = '<div class="memory-item">No memories yet. Add some above!</div>';
            stats = { latencies: [], memoriesUsed: 0, cacheHits: 0, totalQueries: 0, alphas: [] };
            updateStats({});
        }
        
        // Experiments
        async function generateData() {
            try {
                const response = await fetch('/api/experiment/generate-data', { method: 'POST' });
                const data = await response.json();
                alert('Data generated successfully!');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function runExperiment() {
            alert('Starting experiment... This may take a while. Check console for progress.');
            try {
                const response = await fetch('/api/experiment/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: 'Web UI Experiment',
                        modes: ['dki', 'rag', 'baseline'],
                        datasets: ['persona_chat'],
                        max_samples: 10
                    })
                });
                const data = await response.json();
                console.log('Experiment results:', data);
                alert('Experiment completed! Check console for results.');
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        async function runAlphaSensitivity() {
            alert('Running α sensitivity analysis...');
            // This would typically call the alpha sensitivity endpoint
            alert('Feature coming soon!');
        }
    </script>
</body>
</html>'''


# Run server
def run_server():
    """Run the web server."""
    import uvicorn
    
    config = ConfigLoader().config
    app = create_app()
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


if __name__ == "__main__":
    run_server()
