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

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# LLM
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()

ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}


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


def _extract_text_from_image(file_bytes: bytes) -> str:
    if not (pytesseract and Image):
        raise RuntimeError("OCR not available. Please add 'pytesseract' and 'Pillow' to requirements.txt")
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)


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
        "Respond strictly in JSON with keys: summary (string), issues (array of strings), improvements (array of strings), missing (array of strings), disclaimer (string)."
        " The disclaimer must say that these are suggestions for educational purposes and not legal advice."
    )

    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        logger.error(f"LLM error while advising document: {e}")
        raise

    # Best-effort parsing; extract JSON even if wrapped in fences or prose
    advice: Dict[str, Any]
    import json, re
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        pass
    if not isinstance(parsed, dict):
        try:
            fenced = re.sub(r"^```json\s*|^```\s*|```\s*$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)
            parsed = json.loads(fenced)
        except Exception:
            parsed = None
    if not isinstance(parsed, dict):
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = content[start:end+1]
                parsed = json.loads(candidate)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        advice = parsed  # type: ignore
    else:
        advice = {
            "summary": content,
            "issues": [],
            "improvements": [],
            "missing": [],
            "disclaimer": "These insights are suggestions for educational purposes and are not legal advice. Do not rely on them.",
        }

    # Normalize and enforce schema
    # Map common alias keys produced by LLMs
    alias_map = {
        "issues": ["risks", "legal_issues", "concerns", "red_flags"],
        "improvements": ["suggestions", "recommendations", "actions", "fixes"],
        "missing": ["missing_clauses", "gaps", "omissions", "missing_items"],
        "summary": ["summary_text", "overview", "brief"],
        "disclaimer": ["note", "notes", "disclaimer_text"],
    }
    for target, aliases in alias_map.items():
        if advice.get(target) is None:
            for a in aliases:
                if a in advice:
                    advice[target] = advice[a]
                    break

    if not isinstance(advice.get("summary"), str):
        advice["summary"] = str(advice.get("summary", ""))
    for key in ("issues", "improvements", "missing"):
        val = advice.get(key)
        if isinstance(val, str):
            # try to parse stringified list, else split lines
            try:
                arr = json.loads(val)
                val = arr if isinstance(arr, list) else [val]
            except Exception:
                parts = [p.strip("- •\t ") for p in val.splitlines() if p.strip()]
                val = parts
        if not isinstance(val, list):
            val = []
        # keep only strings
        advice[key] = [str(x) for x in val]
    if not isinstance(advice.get("disclaimer"), str):
        advice["disclaimer"] = "These insights are suggestions for educational purposes and are not legal advice. Do not rely on them."

    return advice


@router.post("/api/analyze-document")
async def analyze_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
) -> JSONResponse:
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
            elif (
                filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"))
                or ctype.startswith("image/")
            ):
                extracted = _extract_text_from_image(raw)
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

        from uuid import uuid4
        analysis_id = str(uuid4())
        ANALYSIS_STORE[analysis_id] = {
            "filename": filename,
            "content_type": file.content_type,
            "extracted_chars": len(extracted),
            "advice": advice,
        }

        redirect_url = f"/insights/{analysis_id}"

        response = JSONResponse(
            content={
                "success": True,
                "filename": filename,
                "content_type": file.content_type,
                "extracted_chars": len(extracted),
                "advice": advice,
                "analysis_id": analysis_id,
                "redirect_url": redirect_url,
            }
        )
        response.headers["Location"] = redirect_url
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/analyze-document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analyze failed: {str(e)}")


@router.get("/api/analyze-document/{analysis_id}")
async def get_analysis(analysis_id: str) -> JSONResponse:
    try:
        data = ANALYSIS_STORE.get(analysis_id)
        if not data:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return JSONResponse(content={"success": True, "analysis_id": analysis_id, **data})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/analyze-document/{{analysis_id}} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch analysis")
