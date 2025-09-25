from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Form
import logging

from app.services.agentic_graph import run_agentic_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agentic/chat")
async def agentic_chat(
    input: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    prefer: Optional[str] = Form(None),  # optional hint: "constitution" | "cases" | "web"
) -> Dict[str, Any]:
    """Agentic chat endpoint backed by app.services.agentic_graph.run_agentic_chat."""
    question = (input or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="input is required")
    result = await run_agentic_chat(question, prefer=prefer, conversation_id=conversation_id)
    return result
