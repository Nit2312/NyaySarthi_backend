from typing import Optional
from fastapi import APIRouter, Form, Body, HTTPException
from fastapi.responses import JSONResponse
from app.services.rag import chat_handler

router = APIRouter()

@router.post("/chat")
async def chat(
    input: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    payload: Optional[dict] = Body(None),
):
    # Allow JSON payloads too
    if input is None and payload:
        input = payload.get("input") or payload.get("message")
        conversation_id = conversation_id or payload.get("conversation_id")

    if not input or not str(input).strip():
        raise HTTPException(status_code=400, detail="'input' is required")

    try:
        result = await chat_handler(str(input), conversation_id)
        return result
    except HTTPException as he:
        # Pass through FastAPI HTTPExceptions
        raise he
    except Exception as e:
        # Normalize error shape for frontend (expects 'detail')
        return JSONResponse(status_code=500, content={"detail": str(e) or "Internal Server Error"})
