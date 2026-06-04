"""HTTP API service using AgentScope AgentApp."""

import json
import os
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from starlette.requests import Request

# Suppress httpx/httpcore event loop warnings (Python 3.13 compatibility issue)
import warnings
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

# Patch httpcore to suppress "Event loop is closed" exceptions globally.
# This is a known Python 3.13 issue where AsyncConnectionPool._close_connections
# raises RuntimeError when called after the event loop is already closed.
# The patch wraps all awaitable _close_connections calls to catch and ignore this error.
def _install_httpcore_loop_closed_patch():
    try:
        import asyncio
        import httpcore._async.connection_pool as _pool_mod

        _orig_close = _pool_mod.AsyncConnectionPool._close_connections

        async def _safe_close(self, closing_connections=None):
            try:
                await _orig_close(self, closing_connections)
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    raise
            except Exception:
                pass

        _pool_mod.AsyncConnectionPool._close_connections = _safe_close

        # Also patch the base _Connection.aclose method
        import httpcore._async.connection as _conn_mod
        _orig_aclose = _conn_mod.AsyncConnection.aclose

        async def _safe_aclose(self):
            try:
                await _orig_aclose(self)
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    raise
            except Exception:
                pass

        _conn_mod.AsyncConnection.aclose = _safe_aclose
    except Exception:
        pass

_install_httpcore_loop_closed_patch()

from .agent import CustomerServiceAgent, get_agent, load_agent_config, reload_agent, save_agent_config
from .diagnostics import DiagnosticsCollector
from .models import ModelConfig, get_model_config
from .skills import get_skill_manager, reload_skill_manager
from .llm_providers import provider_catalog

SEMANTIC_CONFIG_PATH = Path("config/semantic_config.json")

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
    # New fields for 5-step:
    enable_five_step: bool = True
    confirmation_result: Optional[Dict[str, Any]] = None
    capabilities: List[str] = []
    user_id: Optional[str] = None


# Track pending confirmations for 5-step pipeline
_pending_confirmations: Dict[str, Dict[str, Any]] = {}


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


# Templates directory — always resolved relative to project root, not cwd
import jinja2

_templates_dir = Path(__file__).parent.parent / "templates"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_templates_dir)),
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)


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


def load_semantic_config() -> Dict[str, Any]:
    """Load semantic backend configuration from file."""
    if not SEMANTIC_CONFIG_PATH.exists():
        return {"yaml_path": "", "graphql_endpoint": "", "use_demo": True}
    try:
        with open(SEMANTIC_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"yaml_path": "", "graphql_endpoint": "", "use_demo": True}


def save_semantic_config(config: Dict[str, Any]) -> None:
    """Save semantic backend configuration to file."""
    SEMANTIC_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


class SemanticConfigUpdate(BaseModel):
    """Model for updating semantic backend configuration."""

    yaml_path: Optional[str] = None
    graphql_endpoint: Optional[str] = None
    use_demo: Optional[bool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} 🚀 Starting Customer Service Agent...")
    agent = get_agent()
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} 📝 Agent initialized with system prompt: {agent.system_prompt[:50]}...")
    yield
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} 🛑 Shutting down Customer Service Agent...")
    # Suppress httpx event loop closed errors during shutdown.
    try:
        _install_httpcore_loop_closed_patch()
    except Exception:
        pass


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
        - stream=true: returns SSE stream per docs/SSE流式响应规范.md

        Accepts both formats:
        - Simple: {"message": "...", "stream": true}
        - deep-chat: {"messages": [{"role": "user", "content": "..."}]}
        """
        agent = get_agent()

        # Get model config for logging
        config = load_agent_config()
        model_cfg = config.get("model_config", {})
        mc = get_model_config()
        mc.provider_id = model_cfg.get("provider_id", mc.provider_id)
        mc.model_name = model_cfg.get("model_name", mc.model_name)
        mc.base_url = model_cfg.get("base_url", mc.base_url)
        mc.temperature = model_cfg.get("temperature", mc.temperature)
        mc.max_tokens = model_cfg.get("max_tokens", mc.max_tokens)
        mc.thinking = model_cfg.get("thinking", mc.thinking)
        mc.thinking_budget = model_cfg.get("thinking_budget", mc.thinking_budget)

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

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [CHAT REQUEST] stream={request.stream}, session_id={request.session_id}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [MODEL] provider={mc.provider_id}, model={mc.model_name}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [THINKING] enabled={mc.thinking}, budget={mc.thinking_budget}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [SYSTEM PROMPT] {mc.provider_id}/{mc.model_name} -> {agent.system_prompt[:100]}...")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [USER MESSAGE] {user_message}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} {'='*60}\n")

        if request.stream:
            return EventSourceResponse(agent.chat_stream(user_message))

        # Use 5-step pipeline if enabled
        if request.enable_five_step:
            from .five_step.pipeline import FiveStepPipeline
            pipeline = FiveStepPipeline(agent=agent, session_id=request.session_id)
            response_text = ""
            async for event in pipeline.run(user_message, session_id=request.session_id):
                # event is a dict with "event" and "data" (data may be a JSON string)
                event_type = event.get("event")
                if event_type == "content":
                    data = event.get("data", {})
                    if isinstance(data, str):
                        import json
                        data = json.loads(data)
                    response_text = data.get("content", "")
                elif event_type == "error":
                    data = event.get("data", {})
                    if isinstance(data, str):
                        import json
                        data = json.loads(data)
                    error_msg = data.get("message", "处理出错")
                    return ChatResponse(response=f"处理出错：{error_msg}", session_id=request.session_id)
                elif event_type == "done":
                    break
            if not response_text:
                response_text = "处理完成"
            return ChatResponse(response=response_text, session_id=request.session_id)

        response = await agent.chat(user_message)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [CHAT RESPONSE] (non-streaming) length={len(response)}, response={response[:500]}...")
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
        Returns SSE stream per docs/SSE流式响应规范.md.
        """
        user_message = None
        for msg in reversed(request.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return {"text": "No message provided"}

        agent = get_agent()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [CHAT/STREAM ALIAS] user_message={user_message}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT/STREAM ALIAS] model={agent.model.model}, thinking_enable={agent.model.parameters.thinking_enable}, reasoning_effort={agent.model.parameters.reasoning_effort}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [CHAT/STREAM ALIAS] system_prompt={agent.system_prompt[:100]}...")
        return EventSourceResponse(agent.chat_stream(user_message))

    # ── Five-Step Pipeline Endpoints ───────────────────────────────────────────

    @app.post("/chat/confirm")
    async def chat_confirm(request: ChatRequest):
        """Handle user confirmation for 5-step pipeline.

        Receives confirmation result and resumes the pipeline from the waiting step.
        """
        from .five_step.pipeline import FiveStepPipeline
        from .sse_stream import error_event, done as done_event
        from sse_starlette.sse import EventSourceResponse

        confirmation = request.confirmation_result
        if not confirmation:
            return {"error": "confirmation_result is required"}

        raw_task_id = confirmation.get("task_id", "")
        task_id = raw_task_id.removeprefix("slot-fill-")
        action = confirmation.get("action", "")  # confirm / cancel

        if action == "cancel":
            return EventSourceResponse(iter([
                error_event(
                    code="CANCELLED",
                    message="用户取消操作",
                    recoverable=True,
                ),
                done_event(),
            ]))

        # Store confirmation for the pipeline to pick up
        _pending_confirmations[task_id] = confirmation

        # Create pipeline and run
        agent = get_agent()
        pipeline = FiveStepPipeline(
            agent=agent,
            session_id=request.session_id
        )

        # Run with confirmation context
        return EventSourceResponse(
            pipeline.run_with_confirmation(
                user_input=confirmation.get("user_input", ""),
                task_id=task_id,
                confirmation=confirmation,
                session_id=request.session_id,
            )
        )

    @app.get("/five-step/status")
    async def five_step_status():
        """Get the 5-step pipeline status and capabilities."""
        config = load_agent_config()
        return {
            "enabled": config.get("enable_five_step", False),
            "supported_events": [
                "step_update",
                "content",
                "tool_call",
                "tool_result",
                "think",
                "think_done",
                "confirm_request",
                "error",
                "done"
            ],
            "capabilities": config.get("capabilities", ["step_lifecycle"]),
        }

    # Process endpoint (AgentApp style)
    @app.post("/process")
    async def process(request: ProcessRequest):
        """Handle process requests in AgentApp format."""
        agent = get_agent()
        messages = [msg.content for msg in request.input if msg.role == "user"]
        if not messages:
            raise HTTPException(status_code=400, detail="No user message found")
        user_msg = messages[-1]
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{ts} [PROCESS] user_message={user_msg}")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [PROCESS] model={agent.model.model}, thinking_enable={agent.model.parameters.thinking_enable}, reasoning_effort={agent.model.parameters.reasoning_effort}")
        response = await agent.chat(user_msg)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"{ts} [PROCESS] response={response[:500] if response else '(empty)'}...")
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
        await agent.reset_history()
        return {"status": "success", "message": "Conversation history cleared"}

    # Model Management Endpoints
    @app.get("/models")
    async def get_models():
        """Get LLM provider catalog with all vendors and their models."""
        return {"providers": provider_catalog()}

    # ── Intent Routing Service (New Comprehensive API) ────────────────────────

    from .intent_routing_service import (
        load_intents,
        create_intent,
        get_intent,
        update_intent,
        delete_intent,
        get_intent_stats,
        test_routing,
        run_diagnostics,
        load_diagnostics,
        add_example,
        remove_example,
        add_slot,
        update_slot,
        remove_slot,
        add_test_case,
        remove_test_case,
        get_publish_check,
    )

    # Intent Routing Endpoints (New Comprehensive API)
    @app.get("/intent-routing/intents")
    async def get_intents_routing():
        """Get all intents with full configuration."""
        intents = load_intents()
        return {"intents": intents, "stats": get_intent_stats()}

    @app.post("/intent-routing/intents")
    async def create_intent_routing(request: Request):
        """Create a new intent."""
        body = await request.json()
        intent = create_intent(body)
        return {"status": "success", "intent": intent}

    @app.get("/intent-routing/intents/{intent_id}")
    async def get_intent_routing(intent_id: str):
        """Get a specific intent by ID."""
        intent = get_intent(intent_id)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")
        return intent

    @app.put("/intent-routing/intents/{intent_id}")
    async def update_intent_routing(intent_id: str, request: Request):
        """Update an existing intent."""
        body = await request.json()
        intent = update_intent(intent_id, body)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "intent": intent}

    @app.delete("/intent-routing/intents/{intent_id}")
    async def delete_intent_routing(intent_id: str):
        """Delete an intent."""
        success = delete_intent(intent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "message": "Intent deleted"}

    @app.get("/intent-routing/stats")
    async def get_intent_routing_stats():
        """Get intent routing statistics."""
        return get_intent_stats()

    # Quick Routing Test
    @app.post("/intent-routing/test")
    async def test_routing_endpoint(request: Request):
        """Test routing for a user message."""
        body = await request.json()
        message = body.get("message", "")
        user_profile = body.get("user_profile", "normal_user")

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        result = test_routing(message, user_profile)
        return result

    # Diagnostics
    @app.get("/intent-routing/diagnose")
    async def diagnose_intents():
        """Run diagnostics on all intents."""
        return run_diagnostics()

    @app.get("/intent-routing/diagnostics")
    async def get_diagnostics():
        """Get latest diagnostics result."""
        return load_diagnostics()

    @app.post("/intent-routing/diagnose")
    async def run_diagnostics_endpoint():
        """Force run diagnostics."""
        return run_diagnostics()

    # Example Management
    @app.post("/intent-routing/intents/{intent_id}/examples")
    async def add_intent_example(intent_id: str, request: Request):
        """Add an example to an intent."""
        body = await request.json()
        example_type = body.get("type")
        text = body.get("text")

        if not example_type or not text:
            raise HTTPException(status_code=400, detail="type and text are required")

        intent = add_example(intent_id, example_type, text)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")

        return {"status": "success", "intent": intent}

    @app.delete("/intent-routing/intents/{intent_id}/examples")
    async def remove_intent_example(intent_id: str, request: Request):
        """Remove an example from an intent."""
        body = await request.json()
        example_type = body.get("type")
        text = body.get("text")

        if not example_type or not text:
            raise HTTPException(status_code=400, detail="type and text are required")

        intent = remove_example(intent_id, example_type, text)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")

        return {"status": "success", "intent": intent}

    # Slot Management
    @app.post("/intent-routing/intents/{intent_id}/slots")
    async def add_intent_slot(intent_id: str, request: Request):
        """Add a slot to an intent."""
        body = await request.json()
        intent = add_slot(intent_id, body)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "intent": intent}

    @app.put("/intent-routing/intents/{intent_id}/slots/{slot_id}")
    async def update_intent_slot(intent_id: str, slot_id: str, request: Request):
        """Update a slot in an intent."""
        body = await request.json()
        intent = update_slot(intent_id, slot_id, body)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "intent": intent}

    @app.delete("/intent-routing/intents/{intent_id}/slots/{slot_id}")
    async def remove_intent_slot(intent_id: str, slot_id: str):
        """Remove a slot from an intent."""
        intent = remove_slot(intent_id, slot_id)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent or slot not found")
        return {"status": "success", "intent": intent}

    # Test Case Management
    @app.post("/intent-routing/intents/{intent_id}/test-cases")
    async def add_intent_test_case(intent_id: str, request: Request):
        """Add a test case to an intent."""
        body = await request.json()
        intent = add_test_case(intent_id, body)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "intent": intent}

    @app.delete("/intent-routing/intents/{intent_id}/test-cases/{case_id}")
    async def remove_intent_test_case(intent_id: str, case_id: str):
        """Remove a test case from an intent."""
        intent = remove_test_case(intent_id, case_id)
        if not intent:
            raise HTTPException(status_code=404, detail="Intent or test case not found")
        return {"status": "success", "intent": intent}

    # Publish Check
    @app.get("/intent-routing/publish-check")
    async def get_intent_publish_check():
        """Run pre-publish checks on intents."""
        return get_publish_check()

    # Legacy Intent Routing Endpoints (Backward Compatible)
    @app.get("/intents")
    async def get_intents():
        """Get all intent routing rules (legacy)."""
        intents = load_intents()
        return intents

    @app.post("/intents")
    async def add_intent(intent: dict):
        """Add a new intent routing rule (legacy)."""
        intent = create_intent(intent)
        return {"status": "success", "message": "Intent added", "intent": intent}

    @app.get("/intents/{index}")
    async def get_intent_by_index(index: int):
        """Get a specific intent by index (legacy)."""
        intents = load_intents()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        return intents[index]

    @app.put("/intents/{index}")
    async def update_intent_by_index(index: int, intent: dict):
        """Update an intent routing rule by index (legacy)."""
        intents = load_intents()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        old_intent = intents[index]
        updated = update_intent(old_intent.get("id"), intent)
        if not updated:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "message": "Intent updated"}

    @app.delete("/intents/{index}")
    async def delete_intent_by_index(index: int):
        """Delete an intent routing rule by index (legacy)."""
        intents = load_intents()
        if index < 0 or index >= len(intents):
            raise HTTPException(status_code=404, detail="Intent not found")
        deleted = delete_intent(intents[index].get("id"))
        if not deleted:
            raise HTTPException(status_code=404, detail="Intent not found")
        return {"status": "success", "message": "Intent deleted"}

    @app.post("/intents/detect")
    async def detect_intent(request: dict):
        """Detect intent from user message (legacy)."""
        message = request.get("message", "")
        result = test_routing(message)
        if result.get("selected_intent"):
            return {
                "detected": True,
                "intent": result["selected_intent"],
                "handler": result.get("route_target"),
                "confidence": result.get("confidence"),
            }
        return {"detected": False, "intent": None, "handler": None}

    @app.post("/intents/test")
    async def test_intent_routing(request: dict):
        """Test intent routing (legacy)."""
        message = request.get("message", "")
        user_profile = request.get("user_profile", "normal_user")
        return test_routing(message, user_profile)

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

    # Semantic Backend Endpoints
    @app.get("/semantic/config")
    async def get_semantic_config():
        """Get current semantic backend configuration."""
        return load_semantic_config()

    @app.put("/semantic/config")
    async def update_semantic_config(config: SemanticConfigUpdate):
        """Update semantic backend configuration."""
        current = load_semantic_config()
        if config.yaml_path is not None:
            current["yaml_path"] = config.yaml_path
        if config.graphql_endpoint is not None:
            current["graphql_endpoint"] = config.graphql_endpoint
        if config.use_demo is not None:
            current["use_demo"] = config.use_demo
        save_semantic_config(current)
        return {"status": "success", "message": "Semantic config updated", "config": current}

    @app.get("/semantic/schema")
    async def get_semantic_schema():
        """Get the current GraphQL SDL schema."""
        try:
            from .skills import get_skill_manager
            sm = get_skill_manager()
            skill = sm.get_skill("Semantic Query")
            if skill and hasattr(skill, "get_schema"):
                return {"schema": skill.get_schema()}
            return {"schema": ""}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/semantic/tools")
    async def get_semantic_tools():
        """Get MCP tools from the semantic backend."""
        try:
            from .skills import get_skill_manager
            sm = get_skill_manager()
            skill = sm.get_skill("Semantic Query")
            if skill and hasattr(skill, "get_all_tools"):
                return {"tools": skill.get_all_tools()}
            return {"tools": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/semantic/search")
    async def semantic_search(request: dict):
        """Test semantic search with a natural language query."""
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        try:
            from .skills import get_skill_manager
            import asyncio
            sm = get_skill_manager()
            skill = sm.get_skill("Semantic Query")
            if not skill:
                raise HTTPException(status_code=404, detail="Semantic Query skill not found")
            result = await skill.execute({"message": query})
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/semantic/reload")
    async def reload_semantic_backend():
        """Reload the semantic backend with current config."""
        try:
            from .skills import get_skill_manager
            sm = get_skill_manager()
            config = load_semantic_config()
            skill = sm.get_skill("Semantic Query")
            if skill:
                skill._initialized = False
                skill._backend = None
                skill._yaml_path = config.get("yaml_path") or None
                skill._graphql_endpoint = config.get("graphql_endpoint") or None
                skill._use_demo_model = config.get("use_demo", True)
                skill._ensure_loaded()
            return {"status": "success", "message": "Semantic backend reloaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── New Admin Workspace (Pico CSS + HTMX) ──────────────────────────────────
    @app.get("/admin", response_class=HTMLResponse)
    async def admin_ui(request: Request):
        """Serve the new admin configuration page with three-column layout."""
        tmpl = _jinja_env.get_template("admin/layout.html")
        return HTMLResponse(tmpl.render(request=request))

    @app.get("/admin/agents/{agent_id}/sections/{section}", response_class=HTMLResponse)
    async def admin_section(request: Request, agent_id: str, section: str):
        """HTMX partial: return the section template for the given section name."""
        section_map = {
            "basic": "admin/sections/basic.html",
            "model": "admin/sections/model.html",
            "prompt": "admin/sections/prompt.html",
            "intent_routing": "admin/sections/intent_routing.html",
            "skills": "admin/sections/skills.html",
            "semantic": "admin/sections/semantic.html",
            "test_publish": "admin/sections/test_publish.html",
            "test_diagnostic": "admin/sections/test_diagnostic.html",
            "test_cases": "admin/sections/test_cases.html",
            "procurement": "admin/sections/procurement.html",
        }
        tmpl_path = section_map.get(section, "admin/sections/basic.html")
        tmpl = _jinja_env.get_template(tmpl_path)
        return HTMLResponse(tmpl.render(request=request))

    @app.post("/admin/agents/{agent_id}/basic")
    async def save_basic(request: Request, agent_id: str):
        """Save agent basic information."""
        body = await request.json()
        current_config = load_agent_config()
        current_config.setdefault("basic", {}).update(body)
        for key, value in body.items():
            if key in ("agent_name", "display_name", "description", "scenario",
                        "owner", "default_language", "timezone",
                        "response_style", "memory_policy"):
                current_config[key] = value
        save_agent_config(current_config)
        return {"status": "success", "message": "基础信息已保存"}

    @app.post("/admin/agents/{agent_id}/model")
    async def save_model(request: Request, agent_id: str):
        """Save model configuration and hot-reload the agent in development mode."""
        body = await request.json()
        current_config = load_agent_config()
        current_config.setdefault("model_config", {}).update(body)
        save_agent_config(current_config)
        
        # Hot-reload agent in development mode (not published)
        reload_result = {"status": "success"}
        if current_config.get("status") != "published":
            try:
                agent = reload_agent()
                reload_result = {
                    "status": "success",
                    "hot_reload": True,
                    "message": "模型配置已保存并应用（热更新）",
                }
                print(f"[HOT-RELOAD] Agent reloaded after model config change")
            except Exception as e:
                reload_result = {
                    "status": "partial",
                    "hot_reload": False,
                    "message": f"配置已保存但热更新失败: {str(e)}",
                }
        else:
            reload_result = {
                "status": "success",
                "hot_reload": False,
                "message": "模型配置已保存（生产模式，请发布后生效）",
            }
        
        return reload_result

    @app.post("/admin/agents/{agent_id}/prompt")
    async def save_prompt(request: Request, agent_id: str):
        """Save prompt configuration and hot-reload the agent in development mode."""
        body = await request.json()
        current_config = load_agent_config()
        current_config.setdefault("prompt_config", {}).update(body)
        if "system_prompt" in body:
            current_config["system_prompt"] = body["system_prompt"]
        save_agent_config(current_config)
        
        # Hot-reload agent in development mode (not published)
        reload_result = {"status": "success"}
        if current_config.get("status") != "published":
            try:
                agent = reload_agent()
                reload_result = {
                    "status": "success",
                    "hot_reload": True,
                    "message": "Prompt 配置已保存并应用（热更新）",
                }
                print(f"[HOT-RELOAD] Agent reloaded after prompt change, new prompt: {body.get('system_prompt', '')[:50]}...")
            except Exception as e:
                reload_result = {
                    "status": "partial",
                    "hot_reload": False,
                    "message": f"配置已保存但热更新失败: {str(e)}",
                }
        else:
            reload_result = {
                "status": "success",
                "hot_reload": False,
                "message": "Prompt 配置已保存（生产模式，请发布后生效）",
            }
        
        return reload_result

    @app.post("/admin/agents/{agent_id}/save-draft")
    async def save_draft(request: Request, agent_id: str):
        """Save full draft configuration and hot-reload the agent in development mode."""
        body = await request.json()
        current_config = load_agent_config()
        current_config.update(body)
        save_agent_config(current_config)
        
        # Hot-reload agent in development mode (not published)
        reload_result = {"status": "success"}
        if current_config.get("status") != "published":
            try:
                agent = reload_agent()
                reload_result = {
                    "status": "success",
                    "hot_reload": True,
                    "message": "草稿已保存并应用（热更新）",
                }
                print(f"[HOT-RELOAD] Agent reloaded after draft save")
            except Exception as e:
                reload_result = {
                    "status": "partial",
                    "hot_reload": False,
                    "message": f"草稿已保存但热更新失败: {str(e)}",
                }
        else:
            reload_result = {
                "status": "success",
                "hot_reload": False,
                "message": "草稿已保存（生产模式，请发布后生效）",
            }
        
        return reload_result

    @app.post("/admin/agents/{agent_id}/hot-reload")
    async def hot_reload_agent(request: Request, agent_id: str):
        """Manually trigger agent hot-reload to apply latest configuration.
        
        This endpoint is designed for development mode where we want immediate
        effect without the publish flow.
        """
        current_config = load_agent_config()
        try:
            agent = reload_agent()
            print(f"[HOT-RELOAD] Manual hot-reload triggered")
            return {
                "status": "success",
                "message": "Agent 已热更新，当前配置已生效",
                "system_prompt_preview": agent.system_prompt[:100] + "..." if len(agent.system_prompt) > 100 else agent.system_prompt,
                "model_name": agent.model.model if hasattr(agent, "model") else "unknown",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"热更新失败: {str(e)}")

    @app.post("/admin/agents/{agent_id}/publish")
    async def publish_agent(request: Request, agent_id: str):
        """Publish the current draft configuration."""
        body = await request.json() or {}
        change_summary = body.get("change_summary", "")
        current_config = load_agent_config()
        current_config["status"] = "published"
        current_config["last_published"] = datetime.now().isoformat()
        current_config["change_summary"] = change_summary
        save_agent_config(current_config)
        # TODO: save config version snapshot to agent_config_versions table
        return {
            "status": "success",
            "message": "发布成功",
            "version": "v1.1",
        }

    @app.post("/admin/agents/{agent_id}/test")
    async def test_agent(request: Request, agent_id: str):
        """Test the agent with a given message, return response text."""
        body = await request.json() or {}
        message = body.get("message", "")
        if not message:
            return {"status": "error", "message": "message is required"}
        agent = get_agent()
        response = await agent.chat(message)
        return {"status": "success", "response": response}

    # ── Diagnostics & Test Runs ──────────────────────────────────────────────

    @app.post("/admin/agents/{agent_id}/test-runs")
    async def create_test_run(request: Request, agent_id: str):
        """Run a test with full diagnostics collection.

        Returns the complete test run with:
        - Agent response
        - Intent classification trace
        - Task plan steps
        - Resource usage
        - Tool call traces
        - Evaluation result and suggestions
        """
        from .diagnostics import DiagnosticsCollector

        body = await request.json() or {}
        message = body.get("message", "")
        scenario_id = body.get("scenario_id", "")

        if not message:
            return {"status": "error", "message": "message is required"}

        agent = get_agent()
        diagnostics = DiagnosticsCollector(agent_id=agent_id, agent_version="draft")

        result = await agent.chat_with_diagnostics(
            user_input=message,
            scenario_id=scenario_id,
            diagnostics=diagnostics,
        )

        test_run = result["test_run"]

        return {
            "status": "success",
            "response": result["response"],
            "decision": result["decision"],
            "test_run": test_run.to_dict(),
        }

    @app.get("/admin/agents/{agent_id}/test-runs")
    async def list_test_runs(request: Request, agent_id: str, limit: int = 20, offset: int = 0):
        """Get history of test runs."""
        from .diagnostics import TestCaseManager

        tcm = TestCaseManager()
        runs = tcm.list_cases(agent_id)
        runs_data = [r.to_dict() for r in runs[offset:offset + limit]]

        return {
            "total": len(runs),
            "test_runs": runs_data,
        }

    @app.get("/admin/agents/{agent_id}/test-runs/{run_id}")
    async def get_test_run(request: Request, agent_id: str, run_id: str):
        """Get detailed information for a specific test run."""
        from pathlib import Path

        storage_dir = Path("data/test_runs")
        file_path = storage_dir / f"{run_id}.json"

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Test run not found")

        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    @app.post("/admin/agents/{agent_id}/test-runs/{run_id}/save-as-case")
    async def save_test_run_as_case(request: Request, agent_id: str, run_id: str):
        """Save a test run as a reusable test case."""
        from .diagnostics import TestCaseManager

        body = await request.json() or {}
        name = body.get("name", f"用例_{datetime.now().strftime('%m%d_%H%M')}")
        expected_intent = body.get("expected_intent", "")

        storage_dir = Path("data/test_runs")
        file_path = storage_dir / f"{run_id}.json"

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Test run not found")

        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tcm = TestCaseManager()
        test_case = tcm.create_from_test_run(
            agent_id=agent_id,
            test_run_id=run_id,
            input_text=data.get("input", ""),
            expected_intent=expected_intent,
            name=name,
        )

        return {
            "status": "success",
            "test_case": test_case.to_dict(),
        }

    @app.post("/admin/agents/{agent_id}/release-check")
    async def release_check(request: Request, agent_id: str):
        """Run pre-release checks on the agent configuration.

        Checks:
        - Basic info completeness
        - Model configuration validity
        - Prompt configuration
        - Intent routing setup
        - Skill availability
        - Risk/policy checks
        """
        from .skills import get_skill_manager

        config = load_agent_config()
        checks = []

        # Check 1: Basic info
        basic_checks = {
            "name": "基础信息完整",
            "description": "Agent Name、描述、场景是否完整",
            "status": "pass",
        }
        if not config.get("agent_name"):
            basic_checks["status"] = "fail"
        checks.append(basic_checks)

        # Check 2: Model config
        model_checks = {
            "name": "模型配置可用",
            "description": "Provider / Model / API 状态是否正常",
            "status": "pass",
        }
        model_cfg = config.get("model_config", {})
        if not model_cfg.get("model_name"):
            model_checks["status"] = "fail"
        checks.append(model_checks)

        # Check 3: Prompt
        prompt_checks = {
            "name": "Prompt 已配置",
            "description": "是否存在有效 System Prompt",
            "status": "pass",
        }
        if not config.get("system_prompt"):
            prompt_checks["status"] = "fail"
        checks.append(prompt_checks)

        # Check 4: Intent routing
        intent_checks = {
            "name": "意图路由有效",
            "description": "是否至少配置一个启用意图",
            "status": "pass",
        }
        intents = load_intent_config()
        if not intents:
            intent_checks["status"] = "warning"
        checks.append(intent_checks)

        # Check 5: Skills
        skill_checks = {
            "name": "Skill 可用",
            "description": "选中 Skill 是否通过测试",
            "status": "pass",
        }
        skill_manager = get_skill_manager()
        skills = skill_manager.get_all_skills()
        if not skills:
            skill_checks["status"] = "warning"
            skill_checks["description"] = "无可用 Skill"
        checks.append(skill_checks)

        # Check 6: Risk/policy
        risk_checks = {
            "name": "风险检查",
            "description": "是否存在敏感配置或高风险输出",
            "status": "pass",
        }
        checks.append(risk_checks)

        # Calculate overall status
        failed_checks = [c for c in checks if c["status"] == "fail"]
        warning_checks = [c for c in checks if c["status"] == "warning"]

        overall_status = "pass"
        if failed_checks:
            overall_status = "fail"
        elif warning_checks:
            overall_status = "warning"

        return {
            "status": overall_status,
            "checks": checks,
            "summary": {
                "total": len(checks),
                "passed": len([c for c in checks if c["status"] == "pass"]),
                "failed": len(failed_checks),
                "warnings": len(warning_checks),
            },
        }

    # ── Test Case Management ─────────────────────────────────────────────────

    @app.get("/admin/agents/{agent_id}/test-cases")
    async def list_test_cases(
        request: Request,
        agent_id: str,
        scenario_id: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """List all test cases for an agent."""
        from .diagnostics import TestCaseManager

        tcm = TestCaseManager()
        cases = tcm.list_cases(agent_id, scenario_id=scenario_id, status=status)

        return {
            "total": len(cases),
            "test_cases": [c.to_dict() for c in cases],
        }

    @app.post("/admin/agents/{agent_id}/test-cases")
    async def create_test_case(request: Request, agent_id: str):
        """Create a new test case."""
        from .diagnostics import TestCaseManager

        body = await request.json() or {}
        tcm = TestCaseManager()

        test_case = tcm.create_case(
            agent_id=agent_id,
            name=body.get("name", ""),
            input_text=body.get("input", ""),
            description=body.get("description", ""),
            scenario_id=body.get("scenario_id", ""),
            user_profile=body.get("user_profile", ""),
            expected_intent=body.get("expected_intent", ""),
            expected_skill=body.get("expected_skill", ""),
            expected_keywords=body.get("expected_keywords", []),
        )

        return {
            "status": "success",
            "test_case": test_case.to_dict(),
        }

    @app.get("/admin/agents/{agent_id}/test-cases/{case_id}")
    async def get_test_case(request: Request, agent_id: str, case_id: str):
        """Get a specific test case."""
        from .diagnostics import TestCaseManager

        tcm = TestCaseManager()
        test_case = tcm.get_case(agent_id, case_id)

        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")

        return test_case.to_dict()

    @app.put("/admin/agents/{agent_id}/test-cases/{case_id}")
    async def update_test_case(request: Request, agent_id: str, case_id: str):
        """Update a test case."""
        from .diagnostics import TestCaseManager

        body = await request.json() or {}
        tcm = TestCaseManager()
        test_case = tcm.update_case(agent_id, case_id, body)

        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")

        return {
            "status": "success",
            "test_case": test_case.to_dict(),
        }

    @app.delete("/admin/agents/{agent_id}/test-cases/{case_id}")
    async def delete_test_case(request: Request, agent_id: str, case_id: str):
        """Delete a test case."""
        from .diagnostics import TestCaseManager

        tcm = TestCaseManager()
        deleted = tcm.delete_case(agent_id, case_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Test case not found")

        return {"status": "success", "message": "Test case deleted"}

    @app.post("/admin/agents/{agent_id}/test-cases/{case_id}/run")
    async def run_test_case(request: Request, agent_id: str, case_id: str):
        """Run a single test case."""
        from .diagnostics import TestCaseManager, DiagnosticsCollector

        tcm = TestCaseManager()
        test_case = tcm.get_case(agent_id, case_id)

        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")

        agent = get_agent()
        diagnostics = DiagnosticsCollector(agent_id=agent_id)

        result = await agent.chat_with_diagnostics(
            user_input=test_case.input,
            diagnostics=diagnostics,
        )

        test_run = result["test_run"]

        return {
            "status": "success",
            "test_run": test_run.to_dict(),
            "response": result["response"],
        }

    # ── Batch Eval ───────────────────────────────────────────────────────────

    @app.post("/admin/agents/{agent_id}/eval/batch")
    async def run_batch_eval(request: Request, agent_id: str):
        """Run batch evaluation on multiple test cases."""
        from .diagnostics import AgentEval, TestCaseManager

        body = await request.json() or {}
        case_ids = body.get("case_ids", [])
        run_mode = body.get("run_mode", "sequential")

        if not case_ids:
            raise HTTPException(status_code=400, detail="case_ids is required")

        tcm = TestCaseManager()
        agent_eval = AgentEval(test_case_manager=tcm)

        async def executor(msg: str):
            agent = get_agent()
            return await agent.chat(msg)

        batch_result = await agent_eval.run_batch(
            agent_id=agent_id,
            case_ids=case_ids,
            run_mode=run_mode,
            agent_executor=executor,
        )

        return batch_result.to_dict()

    @app.get("/admin/agents/{agent_id}/eval/batch/{batch_id}")
    async def get_batch_result(request: Request, agent_id: str, batch_id: str):
        """Get batch evaluation result."""
        from .diagnostics import AgentEval

        agent_eval = AgentEval()
        result = agent_eval.get_batch_result(batch_id)

        if not result:
            raise HTTPException(status_code=404, detail="Batch result not found")

        return result.to_dict()

    @app.get("/admin/agents/{agent_id}/summary", response_class=HTMLResponse)
    async def agent_summary(request: Request, agent_id: str):
        """Return updated right-summary partial."""
        config = load_agent_config()
        tmpl = _jinja_env.get_template("admin/partials/right_summary.html")
        return HTMLResponse(tmpl.render(request=request, agent_config=config))

    # ── Legacy Admin routes (kept for backward compat) ──────────────────────
    @app.get("/admin/intents/list", response_class=HTMLResponse)
    async def intents_list_partial(request: Request):
        """Return intent list as HTML fragment for HTMX refresh."""
        intents = load_intent_config()
        tmpl = _jinja_env.get_template("admin/partials/intent_list.html")
        return HTMLResponse(tmpl.render(request=request, intents=intents))

    @app.post("/admin/intents/add", response_class=HTMLResponse)
    async def add_intent_partial(request: Request, intent: IntentFormData):
        """Add a new intent and return updated list fragment."""
        intents = load_intent_config()
        intents.append(intent.model_dump())
        save_intent_config(intents)
        tmpl = _jinja_env.get_template("admin/partials/intent_list.html")
        return HTMLResponse(tmpl.render(request=request, intents=intents))

    @app.delete("/admin/intents/{index}", response_class=HTMLResponse)
    async def delete_intent_partial(request: Request, index: int):
        """Delete an intent by index and return updated list fragment."""
        intents = load_intent_config()
        if 0 <= index < len(intents):
            intents.pop(index)
            save_intent_config(intents)
        tmpl = _jinja_env.get_template("admin/partials/intent_list.html")
        return HTMLResponse(tmpl.render(request=request, intents=intents))

    # ── Procurement Management ────────────────────────────────────────────────

    @app.get("/procurement/ontology")
    async def get_procurement_ontology():
        """Get procurement ontology summary."""
        try:
            from .procurement import get_procurement_ontology
            ontology = get_procurement_ontology()
            return ontology.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/procurement/intents")
    async def get_procurement_intents():
        """Get all procurement intents."""
        from .procurement import PROCUREMENT_INTENTS
        return {
            "intents": [
                {
                    "id": i.id,
                    "name": i.name,
                    "description": i.description,
                    "trigger_keywords": i.trigger_keywords,
                    "required_slots": i.required_slots,
                    "action": i.action
                } for i in PROCUREMENT_INTENTS
            ]
        }

    @app.post("/procurement/route")
    async def route_procurement_message(request: Request):
        """Route a procurement message to the appropriate intent."""
        body = await request.json()
        message = body.get("message", "")

        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        from .procurement import IntentRouter
        router = IntentRouter()
        result = router.handle_message(message)

        return {"response": result}

    @app.post("/procurement/execute")
    async def execute_procurement_action(request: Request):
        """Execute a procurement action directly."""
        body = await request.json()
        action = body.get("action", "")
        params = body.get("params", {})

        if not action:
            raise HTTPException(status_code=400, detail="action is required")

        from .procurement import get_procurement_connector
        connector = get_procurement_connector()
        result = connector.call(action, params)

        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
            "error": result.error
        }

    @app.get("/procurement/actions")
    async def get_procurement_actions():
        """Get all available procurement actions."""
        from .procurement import get_procurement_connector
        connector = get_procurement_connector()
        return {"actions": connector.get_available_actions()}

    @app.get("/procurement/scenarios")
    async def get_procurement_scenarios():
        """Get all procurement scenarios from ontology."""
        from .procurement import get_procurement_ontology
        ontology = get_procurement_ontology()
        return {
            "scenarios": [
                {
                    "name": s.name,
                    "description": s.description,
                    "mode": s.mode,
                    "key_functions": s.key_functions
                } for s in ontology.scenarios
            ]
        }

    # ── Agent Tool Manifest API (AI-Facing) ────────────────────────────────────
    from .agent_api import get_tool_registry
    from .agent_api.validate import validate_input
    from .agent_api.dry_run import dry_run
    from .agent_api.invoke import invoke
    from .agent_api.hitl import resume_hitl

    @app.get("/agent/manifest/tools")
    async def list_agent_tools():
        """Return all registered Agent tools (lightweight summary)."""
        registry = get_tool_registry()
        return {"tools": registry.list_tools()}

    @app.get("/agent/manifest/tools/{tool_name}")
    async def get_agent_tool(tool_name: str):
        """Return full manifest for a specific tool."""
        registry = get_tool_registry()
        manifest = registry.get_tool(tool_name)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        return manifest

    @app.get("/agent/manifest/tools/{tool_name}/schema/input")
    async def get_tool_input_schema(tool_name: str):
        """Return input JSON schema for a specific tool."""
        registry = get_tool_registry()
        schema = registry.get_input_schema(tool_name)
        if schema is None:
            raise HTTPException(status_code=404, detail=f"Input schema for '{tool_name}' not found")
        return schema

    @app.post("/agent/tools/{tool_name}/validate")
    async def validate_tool_params(tool_name: str, request: Request):
        """Validate input parameters against a tool's input schema."""
        body = await request.json()
        input_params = body.get("input", {})
        session_id = body.get("session_id")
        result = validate_input(tool_name, input_params, session_id)
        return result.to_dict()

    @app.post("/agent/tools/{tool_name}/dry-run")
    async def dry_run_tool(tool_name: str, request: Request):
        """Dry-run a tool: simulate execution and return a confirmation card."""
        body = await request.json()
        input_params = body.get("input", {})
        session_id = body.get("session_id")
        result = dry_run(tool_name, input_params, session_id)
        if not result.get("success") and "not found" in str(result.get("error", "")):
            raise HTTPException(status_code=404, detail=result.get("error"))
        return result

    @app.post("/agent/tools/{tool_name}/invoke")
    async def invoke_tool(tool_name: str, request: Request):
        """Formally invoke a tool with validated parameters."""
        body = await request.json()
        input_params = body.get("input", {})
        dry_run_id = body.get("dry_run_id")
        confirmed = body.get("confirmed", True)
        session_id = body.get("session_id")
        result = invoke(tool_name, input_params, dry_run_id, confirmed, session_id)
        if not result.get("success") and "not found" in str(result.get("error", "")):
            raise HTTPException(status_code=404, detail=result.get("error"))
        return result

    @app.post("/hitl/resume")
    async def hitl_resume(request: Request):
        """Resume a pending HITL task (confirm / cancel / edit / supplement)."""
        body = await request.json()
        task_id = body.get("task_id")
        action = body.get("action")
        payload = body.get("payload", {})
        session_id = body.get("session_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id is required")
        if not action:
            raise HTTPException(status_code=400, detail="action is required")
        result = resume_hitl(task_id, action, payload, session_id)
        return result

    # ── Procurement Management ────────────────────────────────────────────────

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
                "procurement": "/procurement/* - Procurement management APIs",
            },
        }

    return app


# Create the app instance
app = create_app()


def main():
    """Run the application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    loop = os.getenv("UVICORN_LOOP", "asyncio")
    http = os.getenv("UVICORN_HTTP", "h11")
    uvicorn.run(app, host=host, port=port, loop=loop, http=http)


if __name__ == "__main__":
    main()
