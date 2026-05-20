"""HTTP API service using AgentScope AgentApp."""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import CustomerServiceAgent, get_agent

load_dotenv()

# Global agent instance
_agent: Optional[CustomerServiceAgent] = None


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


def get_agent() -> CustomerServiceAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = CustomerServiceAgent()
    return _agent


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
