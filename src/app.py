"""HTTP API service using AgentScope AgentApp."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import CustomerServiceAgent, get_agent, load_agent_config, reload_agent, save_agent_config
from .skills import get_skill_manager, reload_skill_manager

load_dotenv()


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    session_id: Optional[str] = None


class MessageInput(BaseModel):
    """Message input model for AgentApp."""

    role: str
    content: str


class ProcessRequest(BaseModel):
    """Process request model for AgentApp."""

    input: List[MessageInput]


class AgentConfigUpdate(BaseModel):
    """Model for updating agent configuration."""

    agent_name: Optional[str] = None
    system_prompt: Optional[str] = None
    llm_config: Optional[Dict[str, Any]] = None


# Intent Configuration
INTENT_CONFIG_PATH = Path("config/intent_routing.json")

def load_intent_config() -> List[Dict]:
    """Load intent routing configuration from file."""
    if not INTENT_CONFIG_PATH.exists():
        return []
    try:
        with open(INTENT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_intent_config(intents: List[Dict]) -> None:
    """Save intent routing configuration to file."""
    INTENT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTENT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(intents, f, indent=2, ensure_ascii=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Starting Customer Service Agent...")
    agent = get_agent()
    print(f"📝 Agent initialized with system prompt: {agent.system_prompt[:50]}...")
    yield
    print("🛑 Shutting down Customer Service Agent...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=os.getenv("APP_NAME", "CustomerServiceAgent"),
        description=os.getenv("APP_DESCRIPTION", "AI-powered customer service agent"),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "CustomerServiceAgent"}

    @app.get("/readiness")
    async def readiness_check():
        """Readiness check endpoint."""
        return {"status": "ready"}

    @app.get("/liveness")
    async def liveness_check():
        """Liveness check endpoint."""
        return {"status": "alive"}

    # Chat endpoint
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Handle chat requests."""
        agent = get_agent()
        response = await agent.chat(request.message)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
        )

    # Chat endpoint compatible with deep-chat component format
    @app.post("/chat/stream")
    async def chat_stream(request: dict):
        """Handle chat requests from deep-chat component.
        
        Accepts: {"messages": [{"role": "user", "content": "..."}]}
        Returns: {"text": "response content"}
        """
        agent = get_agent()
        messages = request.get("messages", [])
        
        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return {"text": "No message provided"}
        
        response = await agent.chat(user_message)
        return {"text": response}

    # Process endpoint (AgentApp style)
    @app.post("/process")
    async def process(request: ProcessRequest):
        """Handle process requests in AgentApp format."""
        agent = get_agent()
        messages = [msg.content for msg in request.input if msg.role == "user"]
        if not messages:
            raise HTTPException(status_code=400, detail="No user message found")
        response = await agent.chat(messages[-1])
        return {
            "output": response,
            "status": "completed",
        }

    # Agent Configuration Endpoints
    @app.get("/config")
    async def get_config():
        """Get current agent configuration."""
        return load_agent_config()

    @app.put("/config")
    async def update_config(config: AgentConfigUpdate):
        """Update agent configuration."""
        current_config = load_agent_config()
        
        # Update fields if provided
        if config.agent_name is not None:
            current_config["agent_name"] = config.agent_name
        if config.system_prompt is not None:
            current_config["system_prompt"] = config.system_prompt
        if config.llm_config is not None:
            current_config["model_config"] = config.llm_config
        
        save_agent_config(current_config)
        return {"status": "success", "message": "Configuration updated", "config": current_config}

    @app.post("/config/reload")
    async def reload_agent_endpoint():
        """Reload the agent with new configuration."""
        try:
            agent = reload_agent()
            return {"status": "success", "message": "Agent reloaded", "agent_name": agent.agent_name}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reload agent: {str(e)}")

    @app.post("/config/reset")
    async def reset_conversation():
        """Reset agent conversation history."""
        agent = get_agent()
        agent.reset_history()
        return {"status": "success", "message": "Conversation history cleared"}

    # Intent Routing Endpoints
    @app.get("/intents")
    async def get_intents():
        """Get all intent routing rules."""
        return load_intent_config()

    @app.post("/intents")
    async def add_intent(intent: dict):
        """Add a new intent routing rule."""
        intents = load_intent_config()
        intents.append(intent)
        save_intent_config(intents)
        return {"status": "success", "message": "Intent added"}

    @app.get("/intents/{index}")
    async def get_intent(index: int):
        """Get a specific intent by index."""
        intents = load_intent_config()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        return intents[index]

    @app.put("/intents/{index}")
    async def update_intent(index: int, intent: dict):
        """Update an intent routing rule."""
        intents = load_intent_config()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        intents[index] = intent
        save_intent_config(intents)
        return {"status": "success", "message": "Intent updated"}

    @app.delete("/intents/{index}")
    async def delete_intent(index: int):
        """Delete an intent routing rule."""
        intents = load_intent_config()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        intents.pop(index)
        save_intent_config(intents)
        return {"status": "success", "message": "Intent deleted"}

    @app.post("/intents/detect")
    async def detect_intent(request: dict):
        """Detect intent from user message."""
        message = request.get("message", "")
        intents = load_intent_config()
        
        # Sort by priority (higher first)
        sorted_intents = sorted(intents, key=lambda x: x.get("priority", 10), reverse=True)
        
        for intent in sorted_intents:
            keywords = intent.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in message.lower():
                    return {
                        "detected": True,
                        "intent": intent.get("name"),
                        "handler": intent.get("handler"),
                        "matched_keyword": keyword
                    }
        
        return {"detected": False, "intent": None, "handler": None}

    # Skills Management Endpoints
    @app.get("/skills")
    async def get_skills():
        """Get all registered skills."""
        skill_manager = get_skill_manager()
        return {
            "skills": skill_manager.get_skills_summary(),
        }

    @app.get("/skills/{skill_name}")
    async def get_skill(skill_name: str):
        """Get details of a specific skill."""
        skill_manager = get_skill_manager()
        skill = skill_manager.get_skill(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")
        return skill.to_dict()

    @app.post("/skills/detect")
    async def detect_skill(request: dict):
        """Detect which skill should handle the message."""
        message = request.get("message", "")
        skill_manager = get_skill_manager()
        result = skill_manager.detect_intent(message)
        return result

    # Admin UI endpoint
    @app.get("/admin", response_class=HTMLResponse)
    async def admin_ui():
        """Serve the admin configuration page."""
        return ADMIN_HTML_PAGE

    # Welcome page
    @app.get("/")
    async def root():
        """Root endpoint - welcome page."""
        return {
            "name": os.getenv("APP_NAME", "CustomerServiceAgent"),
            "description": "AI-powered customer service agent powered by AgentScope and DeepSeek",
            "endpoints": {
                "chat": "/chat (POST) - Send a chat message",
                "process": "/process (POST) - Process requests in AgentApp format",
                "health": "/health (GET) - Health check",
                "config": "/config (GET/PUT) - Get/Update agent configuration",
                "admin": "/admin (GET) - Admin UI for configuration management",
            },
        }

    return app


# Create the app instance
app = create_app()


# Admin HTML Page
ADMIN_HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OD Assistant - Agent Configuration</title>
    <script type="module" src="https://unpkg.com/deep-chat@1.4.11/dist/deepChat.bundle.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .card.chat-card {
            padding: 0;
            overflow: hidden;
        }
        .card.chat-card h2 {
            padding: 20px 30px;
            margin-bottom: 0;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
        }
        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .form-group input:focus,
        .form-group textarea:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-group textarea {
            min-height: 200px;
            font-family: 'Monaco', 'Menlo', monospace;
            resize: vertical;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn-success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4);
        }
        .btn-danger {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
        }
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(235, 51, 73, 0.4);
        }
        .message {
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            align-items: center;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-dot.active {
            background: #38ef7d;
            box-shadow: 0 0 10px #38ef7d;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .model-config {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .model-config h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        /* Tab Navigation */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 0;
            border-bottom: none;
        }
        .tab-btn {
            padding: 12px 24px;
            border: none;
            border-radius: 12px 12px 0 0;
            background: rgba(255,255,255,0.3);
            color: white;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab-btn:hover {
            background: rgba(255,255,255,0.5);
        }
        .tab-btn.active {
            background: white;
            color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        /* Intent Routing */
        .intent-list {
            margin-top: 15px;
        }
        .intent-item {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .intent-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .intent-item-header h4 {
            color: #333;
            font-size: 16px;
        }
        .intent-keywords {
            color: #666;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .intent-keywords span {
            display: inline-block;
            background: #e0e0e0;
            padding: 4px 10px;
            border-radius: 20px;
            margin: 2px;
            font-size: 12px;
        }
        .intent-actions {
            display: flex;
            gap: 8px;
        }
        .btn-sm {
            padding: 8px 16px;
            font-size: 13px;
        }
        .btn-edit {
            background: #667eea;
            color: white;
        }
        .btn-delete {
            background: #eb3349;
            color: white;
        }
        .add-intent-form {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .add-intent-form h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .keyword-input-group {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }
        .keyword-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
        }
        .keyword-tag button {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        }
        .intent-preview {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .intent-preview h4 {
            color: #856404;
            margin-bottom: 10px;
            font-size: 14px;
        }
        /* Skills */
        .skill-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .skill-card h4 {
            margin: 0 0 10px 0;
            font-size: 18px;
        }
        .skill-card p {
            margin: 0 0 15px 0;
            opacity: 0.9;
            font-size: 14px;
        }
        .skill-meta {
            display: flex;
            gap: 15px;
            font-size: 12px;
            opacity: 0.9;
        }
        .skill-meta span {
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 20px;
        }
        .skill-badge {
            display: inline-block;
            background: #38ef7d;
            color: #155724;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 10px;
        }
        .skill-test-area {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .skill-test-area h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .test-result {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid #e0e0e0;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 13px;
            max-height: 300px;
            overflow-y: auto;
        }
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .status-indicator .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #38ef7d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 OD Assistant Admin</h1>
            <p>Configure and manage your customer service agent</p>
        </div>

        <!-- Tab Navigation -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('config')">⚙️ Configuration</button>
            <button class="tab-btn" onclick="switchTab('intents')">🎯 Intent Routing</button>
            <button class="tab-btn" onclick="switchTab('skills')">🧠 Skills</button>
        </div>

        <!-- Tab: Configuration -->
        <div id="tab-config" class="tab-content active">
            <div class="status-item">
                <div class="status-dot active"></div>
                <span>Agent Active</span>
            </div>
            <div class="status-item">
                <span id="agentName">Loading...</span>
            </div>
        </div>

        <div class="card">
            <h2>📝 Agent Configuration</h2>
            <div class="form-group">
                <label for="agentName">Agent Name</label>
                <input type="text" id="agentNameInput" placeholder="Enter agent name">
            </div>
            <div class="form-group">
                <label for="systemPrompt">System Prompt</label>
                <textarea id="systemPrompt" placeholder="Enter system prompt"></textarea>
            </div>
            <div id="saveMessage" class="message"></div>
            <div style="margin-top: 20px;">
                <button class="btn btn-primary" onclick="saveConfig()">💾 Save Configuration</button>
                <button class="btn btn-success" onclick="reloadAgent()">🔄 Reload Agent</button>
                <button class="btn btn-danger" onclick="resetHistory()">🗑️ Reset History</button>
            </div>
        </div>

        <div class="card">
            <h2>⚙️ Model Configuration</h2>
            <div class="model-config">
                <div class="grid">
                    <div class="form-group">
                        <label for="modelName">Model Name</label>
                        <input type="text" id="modelName" placeholder="deepseek-chat">
                    </div>
                    <div class="form-group">
                        <label for="baseUrl">Base URL</label>
                        <input type="text" id="baseUrl" placeholder="https://api.deepseek.com/v1">
                    </div>
                </div>
                <div class="grid">
                    <div class="form-group">
                        <label for="temperature">Temperature (0-1)</label>
                        <input type="number" id="temperature" min="0" max="2" step="0.1" value="0.7">
                    </div>
                    <div class="form-group">
                        <label for="maxTokens">Max Tokens</label>
                        <input type="number" id="maxTokens" min="100" max="8000" value="2000">
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📋 Quick Actions</h2>
            <button class="btn btn-primary" onclick="loadConfig()">🔃 Refresh Configuration</button>
            <button class="btn btn-success" onclick="initChat()">🔄 Initialize Chat</button>
        </div>

        <!-- Tab: Intent Routing -->
        <div id="tab-intents" class="tab-content">
            <div class="card">
                <h2>🎯 Intent Routing Rules</h2>
                <p style="color: #666; margin-bottom: 15px;">Configure keyword-based routing to direct users to specialized agents or handlers.</p>
                
                <div class="intent-list" id="intentList">
                    <!-- Intent items will be loaded here -->
                </div>

                <div class="add-intent-form">
                    <h3>+ Add New Intent Rule</h3>
                    <div class="grid">
                        <div class="form-group">
                            <label for="intentName">Intent Name</label>
                            <input type="text" id="intentName" placeholder="e.g., Billing Inquiry">
                        </div>
                        <div class="form-group">
                            <label for="intentHandler">Handler / Agent</label>
                            <input type="text" id="intentHandler" placeholder="e.g., billing_agent or /api/billing">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Keywords (press Enter to add)</label>
                        <input type="text" id="keywordInput" placeholder="Type keyword and press Enter..." onkeypress="if(event.key==='Enter'){addKeyword();event.preventDefault();}">
                        <div id="keywordTags" style="margin-top: 10px; min-height: 30px;"></div>
                    </div>
                    <div class="form-group">
                        <label for="intentDescription">Description (optional)</label>
                        <input type="text" id="intentDescription" placeholder="Brief description of this intent">
                    </div>
                    <div class="form-group">
                        <label for="intentPriority">Priority (higher = checked first)</label>
                        <input type="number" id="intentPriority" value="10" min="1" max="100">
                    </div>
                    <button class="btn btn-success" onclick="addIntent()">✅ Add Intent</button>
                </div>

                <div id="intentMessage" class="message" style="margin-top: 15px;"></div>
            </div>
        </div>

        <!-- Tab: Skills -->
        <div id="tab-skills" class="tab-content">
            <div class="card">
                <h2>🧠 Registered Skills</h2>
                <p style="color: #666; margin-bottom: 15px;">Skills are automatically triggered based on intent detection.</p>
                
                <div class="skill-list" id="skillList">
                    <!-- Skill cards will be loaded here -->
                </div>

                <div class="skill-test-area">
                    <h3>🧪 Skill Detection Test</h3>
                    <div class="form-group">
                        <label for="testMessage">Test Message</label>
                        <input type="text" id="testMessage" placeholder="输入一条测试消息来检测技能触发...">
                    </div>
                    <button class="btn btn-success" onclick="testSkillDetection()">🔍 Test Detection</button>
                    
                    <div id="testResult" class="test-result" style="display: none;"></div>
                </div>
            </div>
        </div>

        <!-- Fixed Chat Widget -->
        <deep-chat
            id="testChat"
            chat-title="OD Assistant"
            user-role="user"
            ai-role="assistant"
            intro-message='{"text": "👋 Hello! How can I help you?"}'
            drop-image-enabled="false"
            history-section-position="top"
            show-reload-button="true"
            ai-avatar='{"initial": "A"}'
            user-avatar='{"initial": "U"}'
            styles='{
                "chatContainer": {"position": "fixed", "bottom": "0", "left": "0", "right": "0", "height": "450px", "z-index": "1000", "borderRadius": "16px 16px 0 0", "boxShadow": "0 -4px 20px rgba(0,0,0,0.15)"},
                "title": {"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "color": "white"},
                "toggleChatButton": {"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"}
            }'
            attach-input-area-init='{"minimise": true}'
        ></deep-chat>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', async function() {
            await loadConfig();
            initChat();
        });

        async function loadConfig() {
            try {
                const response = await fetch('/config');
                const config = await response.json();
                
                document.getElementById('agentName').textContent = 'Agent: ' + (config.agent_name || 'OD_Assistant');
                document.getElementById('agentNameInput').value = config.agent_name || 'OD_Assistant';
                document.getElementById('systemPrompt').value = config.system_prompt || '';
                
                if (config.model_config) {
                    const llmConfig = config.model_config;
                    document.getElementById('modelName').value = llmConfig.model_name || 'deepseek-chat';
                    document.getElementById('baseUrl').value = llmConfig.base_url || 'https://api.deepseek.com/v1';
                    document.getElementById('temperature').value = llmConfig.temperature || 0.7;
                    document.getElementById('maxTokens').value = llmConfig.max_tokens || 2000;
                }
            } catch (error) {
                showMessage('Failed to load configuration: ' + error.message, 'error');
            }
        }

        async function saveConfig() {
            const config = {
                agent_name: document.getElementById('agentNameInput').value,
                system_prompt: document.getElementById('systemPrompt').value,
                llm_config: {
                    model_name: document.getElementById('modelName').value,
                    base_url: document.getElementById('baseUrl').value,
                    temperature: parseFloat(document.getElementById('temperature').value),
                    max_tokens: parseInt(document.getElementById('maxTokens').value)
                }
            };

            try {
                const response = await fetch('/config', {
                    method: 'PUT',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                showMessage('Configuration saved successfully!', 'success');
            } catch (error) {
                showMessage('Failed to save configuration: ' + error.message, 'error');
            }
        }

        async function reloadAgent() {
            try {
                const response = await fetch('/config/reload', {method: 'POST'});
                const result = await response.json();
                showMessage('Agent reloaded successfully!', 'success');
                loadConfig();
            } catch (error) {
                showMessage('Failed to reload agent: ' + error.message, 'error');
            }
        }

        async function resetHistory() {
            if (!confirm('Are you sure you want to reset the conversation history?')) return;
            
            try {
                const response = await fetch('/config/reset', {method: 'POST'});
                const result = await response.json();
                showMessage('Conversation history cleared!', 'success');
            } catch (error) {
                showMessage('Failed to reset history: ' + error.message, 'error');
            }
        }

        function initChat() {
            const chatElement = document.getElementById('testChat');
            if (!chatElement) return;

            // Configure the chat to connect to our /chat/stream endpoint
            chatElement.request = {
                url: '/chat',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            // Custom response handler for our API format
            chatElement.responseHandler = function(response) {
                return new Promise(function(resolve, reject) {
                    if (response.status === 200) {
                        response.json().then(function(data) {
                            resolve({ text: data.text || 'No response' });
                        }).catch(reject);
                    } else {
                        response.json().then(function(data) {
                            reject(new Error(data.detail || 'Request failed'));
                        }).catch(function() {
                            reject(new Error('Request failed with status ' + response.status));
                        });
                    }
                });
            };
        }

        function showMessage(text, type) {
            const messageEl = document.getElementById('saveMessage');
            if (!messageEl) return;
            messageEl.textContent = text;
            messageEl.className = 'message ' + type;
            messageEl.style.display = 'block';
            setTimeout(() => messageEl.style.display = 'none', 3000);
        }

        function showIntentMessage(text, type) {
            const messageEl = document.getElementById('intentMessage');
            if (!messageEl) return;
            messageEl.textContent = text;
            messageEl.className = 'message ' + type;
            messageEl.style.display = 'block';
            setTimeout(() => messageEl.style.display = 'none', 3000);
        }

        // Tab Navigation
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${tabId}')"]`).classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
            
            if (tabId === 'intents') {
                loadIntents();
            } else if (tabId === 'skills') {
                loadSkills();
            }
        }

        // Intent Management
        let currentKeywords = [];

        function addKeyword() {
            const input = document.getElementById('keywordInput');
            const keyword = input.value.trim();
            if (keyword && !currentKeywords.includes(keyword)) {
                currentKeywords.push(keyword);
                renderKeywordTags();
                input.value = '';
            }
        }

        function removeKeyword(keyword) {
            currentKeywords = currentKeywords.filter(k => k !== keyword);
            renderKeywordTags();
        }

        function renderKeywordTags() {
            const container = document.getElementById('keywordTags');
            container.innerHTML = currentKeywords.map(k =>
                `<span class="keyword-tag"><span>${k}</span><button onclick="removeKeyword('${k}')">×</button></span>`
            ).join('');
        }

        async function loadIntents() {
            try {
                const response = await fetch('/intents');
                const intents = await response.json();
                renderIntentList(intents);
            } catch (error) {
                showIntentMessage('Failed to load intents: ' + error.message, 'error');
            }
        }

        function renderIntentList(intents) {
            const container = document.getElementById('intentList');
            if (!intents || intents.length === 0) {
                container.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No intent rules configured yet. Add your first rule below.</p>';
                return;
            }
            container.innerHTML = intents.map((intent, index) => `
                <div class="intent-item">
                    <div class="intent-item-header">
                        <h4>${intent.name || 'Unnamed Intent'}</h4>
                        <span style="color: #999; font-size: 12px;">Priority: ${intent.priority || 10}</span>
                    </div>
                    <div class="intent-keywords">
                        ${(intent.keywords || []).map(k => `<span>${k}</span>`).join('')}
                    </div>
                    <p style="color: #666; font-size: 13px; margin-bottom: 10px;">${intent.description || 'No description'}</p>
                    <p style="color: #667eea; font-size: 13px; font-weight: 600;">Handler: ${intent.handler || 'default'}</p>
                    <div class="intent-actions" style="margin-top: 10px;">
                        <button class="btn btn-sm btn-edit" onclick="editIntent(${index})">✏️ Edit</button>
                        <button class="btn btn-sm btn-delete" onclick="deleteIntent(${index})">🗑️ Delete</button>
                    </div>
                </div>
            `).join('');
        }

        async function addIntent() {
            const name = document.getElementById('intentName').value.trim();
            const handler = document.getElementById('intentHandler').value.trim();
            const description = document.getElementById('intentDescription').value.trim();
            const priority = parseInt(document.getElementById('intentPriority').value) || 10;

            if (!name || !handler) {
                showIntentMessage('Please fill in intent name and handler.', 'error');
                return;
            }

            const intent = {
                name,
                handler,
                keywords: currentKeywords,
                description,
                priority
            };

            try {
                const response = await fetch('/intents', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(intent)
                });
                const result = await response.json();
                showIntentMessage('Intent added successfully!', 'success');
                
                // Reset form
                document.getElementById('intentName').value = '';
                document.getElementById('intentHandler').value = '';
                document.getElementById('intentDescription').value = '';
                document.getElementById('intentPriority').value = '10';
                currentKeywords = [];
                renderKeywordTags();
                
                loadIntents();
            } catch (error) {
                showIntentMessage('Failed to add intent: ' + error.message, 'error');
            }
        }

        async function deleteIntent(index) {
            if (!confirm('Are you sure you want to delete this intent?')) return;
            
            try {
                const response = await fetch(`/intents/${index}`, {method: 'DELETE'});
                const result = await response.json();
                showIntentMessage('Intent deleted successfully!', 'success');
                loadIntents();
            } catch (error) {
                showIntentMessage('Failed to delete intent: ' + error.message, 'error');
            }
        }

        async function editIntent(index) {
            try {
                const response = await fetch('/intents/' + index);
                const intent = await response.json();
                
                document.getElementById('intentName').value = intent.name || '';
                document.getElementById('intentHandler').value = intent.handler || '';
                document.getElementById('intentDescription').value = intent.description || '';
                document.getElementById('intentPriority').value = intent.priority || 10;
                currentKeywords = intent.keywords || [];
                renderKeywordTags();
                
                showIntentMessage('Edit mode: modify the fields and click Add Intent to update.', 'success');
            } catch (error) {
                showIntentMessage('Failed to load intent: ' + error.message, 'error');
            }
        }

        // Skills Management
        async function loadSkills() {
            try {
                const response = await fetch('/skills');
                const data = await response.json();
                renderSkillList(data.skills || []);
            } catch (error) {
                console.error('Failed to load skills:', error);
            }
        }

        function renderSkillList(skills) {
            const container = document.getElementById('skillList');
            if (!skills || skills.length === 0) {
                container.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No skills registered.</p>';
                return;
            }
            container.innerHTML = skills.map(skill => `
                <div class="skill-card">
                    <h4>${skill.name}</h4>
                    <p>${skill.description}</p>
                    <div class="skill-meta">
                        <span>Priority: ${skill.priority}</span>
                        <span>Keywords: ${skill.keywords_count}</span>
                    </div>
                    <span class="skill-badge">✓ Active</span>
                </div>
            `).join('');
        }

        async function testSkillDetection() {
            const message = document.getElementById('testMessage').value.trim();
            if (!message) {
                alert('请输入测试消息');
                return;
            }

            const resultDiv = document.getElementById('testResult');
            resultDiv.style.display = 'block';
            resultDiv.textContent = '检测中...';

            try {
                const response = await fetch('/skills/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                const result = await response.json();

                if (result.matched) {
                    resultDiv.innerHTML = `✅ 检测到技能：<strong>${result.intent}</strong>\n` +
                        `Handler: ${result.handler}\n` +
                        `匹配关键词: ${result.matched_keyword}`;
                    resultDiv.style.color = '#155724';
                } else {
                    resultDiv.textContent = '❌ 未检测到匹配的技能';
                    resultDiv.style.color = '#721c24';
                }
            } catch (error) {
                resultDiv.textContent = '检测失败: ' + error.message;
                resultDiv.style.color = '#721c24';
            }
        }
    </script>
</body>
</html>
"""


def main():
    """Run the application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
