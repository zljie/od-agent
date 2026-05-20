"""HTTP API service using AgentScope AgentApp."""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import CustomerServiceAgent, get_agent, load_agent_config, reload_agent, save_agent_config

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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 OD Assistant Admin</h1>
            <p>Configure and manage your customer service agent</p>
        </div>

        <div class="status-bar">
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
            <button class="btn btn-success" onclick="testChat()">🧪 Test Chat</button>
        </div>
    </div>

    <script>
        // Load configuration on page load
        document.addEventListener('DOMContentLoaded', loadConfig);

        async function loadConfig() {
            try {
                const response = await fetch('/config');
                const config = await response.json();
                
                document.getElementById('agentName').textContent = 'Agent: ' + (config.agent_name || 'OD_Assistant');
                document.getElementById('agentNameInput').value = config.agent_name || 'OD_Assistant';
                document.getElementById('systemPrompt').value = config.system_prompt || '';
                
                if (config.model_config) {
                    // Use llm_config for API, but display as model_config
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

        async function testChat() {
            const message = prompt('Enter a test message:');
            if (!message) return;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                });
                const result = await response.json();
                alert('Agent Response:\\n\\n' + result.response);
            } catch (error) {
                showMessage('Test failed: ' + error.message, 'error');
            }
        }

        function showMessage(text, type) {
            const messageEl = document.getElementById('saveMessage');
            messageEl.textContent = text;
            messageEl.className = 'message ' + type;
            messageEl.style.display = 'block';
            setTimeout(() => messageEl.style.display = 'none', 3000);
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
