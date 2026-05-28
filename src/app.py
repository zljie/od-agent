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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from starlette.requests import Request

from .agent import CustomerServiceAgent, get_agent, load_agent_config, reload_agent, save_agent_config
from .models import ModelConfig, get_model_config
from .skills import get_skill_manager, reload_skill_manager
from .llm_providers import provider_catalog

load_dotenv()


class ChatRequest(BaseModel):
    """Chat request model.

    Supports two formats:
    - Simple: {"message": "...", "stream": false}
    - deep-chat SSE: {"messages": [{"role": "user", "content": "..."}], "stream": true}
    """

    message: Optional[str] = None
    messages: Optional[List[Any]] = None
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


class IntentFormData(BaseModel):
    """Model for intent form data submitted via HTMX."""

    name: str
    handler: str
    keywords: List[str] = []
    description: str = ""
    priority: int = 10


# Jinja2 templates - initialized at module level
templates = Jinja2Templates(directory="templates")


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

    # Mount static files for admin UI
    app.mount("/static", StaticFiles(directory="static"), name="static")

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

    # Chat endpoint — unified, supports both blocking and SSE streaming
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Handle chat requests.

        Unified endpoint supporting two modes:
        - stream=false (default): waits for full response, returns {response, session_id}
        - stream=true: returns SSE stream of text chunks

        Accepts both formats:
        - Simple: {"message": "...", "stream": true}
        - deep-chat: {"messages": [{"role": "user", "content": "..."}]}
        """
        agent = get_agent()

        # Support both simple message and deep-chat messages array
        user_message: Optional[str] = None
        if request.message:
            user_message = request.message
        else:
            for msg in reversed(getattr(request, 'messages', [])):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
                if hasattr(msg, 'role') and msg.role == "user":
                    user_message = msg.content
                    break

        if not user_message:
            return ChatResponse(response="No message provided", session_id=request.session_id)

        if request.stream:
            async def event_generator():
                async for chunk in agent.chat_stream(user_message):
                    if chunk:
                        yield {"event": "message", "data": json.dumps({"content": chunk}, ensure_ascii=False)}
                yield {"event": "message", "data": "[DONE]"}

            return EventSourceResponse(event_generator())

        response = await agent.chat(user_message)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
        )

    # Backward-compatible alias: POST /chat/stream
    # deep-chat sends {"messages": [...]} — normalize to ChatRequest
    @app.post("/chat/stream")
    async def chat_stream_alias(request: dict):
        """Alias for POST /chat with stream=true.

        Accepts deep-chat format: {"messages": [{"role": "user", "content": "..."}]}
        Internally delegates to the unified /chat endpoint.
        """
        # Extract last user message from messages array
        user_message = None
        for msg in reversed(request.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return {"text": "No message provided"}

        agent = get_agent()

        async def event_generator():
            async for chunk in agent.chat_stream(user_message):
                if chunk:
                    yield {"event": "message", "data": json.dumps({"content": chunk}, ensure_ascii=False)}
            yield {"event": "message", "data": "[DONE]"}

        return EventSourceResponse(event_generator())

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
        """Get current agent configuration with full provider catalog for dropdowns."""
        config = load_agent_config()
        model_cfg = config.get("model_config", {})
        mc = get_model_config()
        mc.provider_id = model_cfg.get("provider_id", mc.provider_id)
        mc.model_name = model_cfg.get("model_name", mc.model_name)
        mc.base_url = model_cfg.get("base_url", mc.base_url)
        mc.temperature = model_cfg.get("temperature", mc.temperature)
        mc.max_tokens = model_cfg.get("max_tokens", mc.max_tokens)
        mc.top_p = model_cfg.get("top_p", mc.top_p)
        mc.top_k = model_cfg.get("top_k", mc.top_k)
        mc.presence_penalty = model_cfg.get("presence_penalty", mc.presence_penalty)
        mc.frequency_penalty = model_cfg.get("frequency_penalty", mc.frequency_penalty)
        mc.seed = model_cfg.get("seed", mc.seed)
        mc.thinking = model_cfg.get("thinking", mc.thinking)
        mc.thinking_budget = model_cfg.get("thinking_budget", mc.thinking_budget)
        catalog = mc.to_catalog_dict()
        catalog["agent_name"] = config.get("agent_name", "OD_Assistant")
        catalog["system_prompt"] = config.get("system_prompt", "")
        return catalog

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

    # Model Management Endpoints
    @app.get("/models")
    async def get_models():
        """Get LLM provider catalog with all vendors and their models."""
        return {"providers": provider_catalog()}

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
    async def admin_ui(request: Request):
        """Serve the admin configuration page."""
        return templates.TemplateResponse("admin/base.html", {"request": request})

    # HTMX partial routes for intent management
    @app.get("/admin/intents/list", response_class=HTMLResponse)
    async def intents_list_partial(request: Request):
        """Return intent list as HTML fragment for HTMX refresh."""
        intents = load_intent_config()
        return templates.TemplateResponse("admin/partials/intent_list.html", {
            "request": request,
            "intents": intents,
        })

    @app.post("/admin/intents/add", response_class=HTMLResponse)
    async def add_intent_partial(request: Request, intent: IntentFormData):
        """Add a new intent and return updated list fragment."""
        intents = load_intent_config()
        intents.append(intent.model_dump())
        save_intent_config(intents)
        return templates.TemplateResponse("admin/partials/intent_list.html", {
            "request": request,
            "intents": intents,
        })

    @app.delete("/admin/intents/{index}", response_class=HTMLResponse)
    async def delete_intent_partial(request: Request, index: int):
        """Delete an intent by index and return updated list fragment."""
        intents = load_intent_config()
        if 0 <= index < len(intents):
            intents.pop(index)
            save_intent_config(intents)
        return templates.TemplateResponse("admin/partials/intent_list.html", {
            "request": request,
            "intents": intents,
        })

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


def main():
    """Run the application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
