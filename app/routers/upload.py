from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os
import io
import logging

# Optional heavy imports guarded
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# LLM
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    if not fitz:
        raise RuntimeError("PyMuPDF is not installed. Please add 'pymupdf' to requirements.txt")
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n\n".join([t.strip() for t in text_parts if t and t.strip()])


def _extract_text_from_docx(file_bytes: bytes) -> str:
    if not docx:
        raise RuntimeError("python-docx is not installed. Please add 'python-docx' to requirements.txt")
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)


def _summarize_and_advise(text: str, user_goal: Optional[str]) -> Dict[str, Any]:
    """Use the LLM to produce a concise legal-focused critique and improvements."""
    if not ChatGroq:
        raise RuntimeError("LangChain Groq integration not available")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in backend environment")

    llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model)

    # Keep the text at a reasonable length for the model context
    snippet = text[:6000]

    prompt = (
        "You are a legal assistant. Analyze the following user-provided legal document/text and provide: \n"
        "1) A brief summary (3-4 lines).\n"
        "2) Potential legal issues or risks (bullet points).\n"
        "3) Suggested improvements (bullet points).\n"
        "4) If applicable, missing clauses or data points.\n"
        "Keep the advice general and educational; do not provide definitive legal conclusions. "
        "Be concise and practical.\n\n"
        f"USER GOAL (optional): {user_goal or 'N/A'}\n\n"
        f"DOCUMENT/TEXT:\n{snippet}\n\n"
        "Respond in structured JSON with keys: summary, issues, improvements, missing."
    )

    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.error(f"LLM error while advising document: {e}")
        raise

    # Best-effort parsing; if not valid JSON, wrap into fields
    advice: Dict[str, Any]
    try:
        import json
        advice = json.loads(content)
        if not isinstance(advice, dict):
            raise ValueError("not a dict")
    except Exception:
        advice = {
            "summary": content[:800],
            "issues": [],
            "improvements": [],
            "missing": [],
        }

    return advice


@router.post("/api/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
) -> JSONResponse:
    """Upload a document (PDF/DOCX/TXT) and receive legal-styled advice for improvements.
    Returns: { success, filename, content_type, extracted_chars, advice: {summary, issues, improvements, missing} }
    """
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="File is required")

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        filename = file.filename
        ctype = (file.content_type or "").lower()

        # Extract text depending on type
        extracted = ""
        try:
            if filename.lower().endswith(".pdf") or "pdf" in ctype:
                extracted = _extract_text_from_pdf(raw)
            elif filename.lower().endswith(".docx") or "word" in ctype:
                extracted = _extract_text_from_docx(raw)
            else:
                # Treat as plain text
                try:
                    extracted = raw.decode("utf-8", errors="ignore")
                except Exception:
                    extracted = ""
        except Exception as e:
            logger.warning(f"Extraction failed for {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

        if not extracted or len(extracted.strip()) < 20:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the document")

        advice = _summarize_and_advise(extracted, description)

        return JSONResponse(
            content={
                "success": True,
                "filename": filename,
                "content_type": file.content_type,
                "extracted_chars": len(extracted),
                "advice": advice,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/analyze-document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analyze failed: {str(e)}")
