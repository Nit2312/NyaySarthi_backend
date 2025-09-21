from fastapi import APIRouter, Form, HTTPException
from typing import Optional, Dict, Any
import time
import hashlib
import math

from app.services.cases import search_indian_kanoon_async, get_case_details_async
from app.services.rag import EMBEDDINGS, GROQ_API_KEY, GROQ_MODEL
try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:
    ChatGroq = None  # type: ignore

router = APIRouter()

@router.post("/cases/search")
async def cases_search(input: str = Form(...), limit: int = Form(5)):
    """Search Indian Kanoon and return a response compatible with the frontend.
    Frontend expects: { success: bool, cases: [...], ik_error?: str, message?: str }
    """
    query = (input or "").strip()
    if not query:
        # Frontend handles message on non-2xx, but still provide detail
        raise HTTPException(status_code=400, detail="Input query cannot be empty")

    limit_val = max(1, min(10, int(limit or 5)))

    try:
        docs, ik_error = await search_indian_kanoon_async(query, limit_val)
        # Normalize response
        if docs:
            return {
                "success": True,
                "cases": [d.dict() for d in docs],
                "ik_error": ik_error,
            }
        else:
            # No results is not an error; return success with empty cases and any ik_error code
            return {
                "success": True,
                "cases": [],
                "ik_error": ik_error,
                "message": "no_results" if not ik_error else ik_error,
            }
    except Exception as e:
        # Surface a friendly error that the frontend can display
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/api/case-details")
async def case_details(doc_id: str = Form(...), description: Optional[str] = Form(None)):
    """Return case details for a given Indian Kanoon doc_id.
    Response is normalized to include success boolean and common fields.
    """
    if not doc_id or not doc_id.strip():
        raise HTTPException(status_code=400, detail="'doc_id' is required")
    result = await get_case_details_async(doc_id.strip(), description)
    # If not found, preserve 404 for the proxy to surface a clear message
    if not result.get("success") and result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Case not found")
    return result


# Simple in-memory cache for case analysis to avoid recomputation
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
_ANALYSIS_TTL_SEC = 600  # 10 minutes


def _cache_key_for_analysis(doc_id: str, full_text: str, description: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update((doc_id or "").encode("utf-8", errors="ignore"))
    # Text can be huge; hash it to keep key small
    h.update((str(len(full_text)) + ":" + full_text[:5000]).encode("utf-8", errors="ignore"))
    h.update((description or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _cosine(a, b) -> float:
    try:
        import numpy as np
        from numpy.linalg import norm
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        na = norm(a) or 1.0
        nb = norm(b) or 1.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def _truncate_for_embedding(text: str, max_chars: int = 4000) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    # Try to cut at paragraph boundary
    cut = t.rfind("\n\n", 0, max_chars)
    return t[: cut if cut > 1000 else max_chars]


def _split_sentences(text: str) -> list[str]:
    import re
    # Basic sentence splitter; keeps periods in abbreviations better than naive split
    text = (text or "").strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split on punctuation followed by space+capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    # Clean and filter
    out = []
    for p in parts:
        s = p.strip()
        if len(s.split()) >= 6:  # avoid tiny fragments
            out.append(s)
    return out[:2000]


def _select_key_points(text: str, description: Optional[str]) -> list[str]:
    """Select up to 10 meaningful, self-contained key sentences.
    Uses embeddings if available, combining similarity to description and centrality.
    """
    try:
        sents = _split_sentences(text)
        if not sents:
            return []
        # Compute embeddings
        if EMBEDDINGS is None:
            # Fallback: pick first 10 informative sentences (with legal cues)
            cues = ["issue", "held", "holding", "finding", "conclusion", "therefore", "court", "facts", "question", "law", "ratio", "summons", "jurisdiction", "maintainable"]
            scored = []
            for idx, s in enumerate(sents):
                score = 0.0
                low = s.lower()
                for c in cues:
                    if c in low:
                        score += 1.0
                # position bonus (earlier and later paragraphs often have summaries/conclusions)
                score += 0.25 * (1.0 - idx / max(1, len(sents)-1))
                if len(s.split()) > 30:
                    score -= 0.2  # penalize very long
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            uniq = []
            seen = set()
            for _, s in scored:
                key = s[:80]
                if key in seen:
                    continue
                seen.add(key)
                # Ensure sentence ends with punctuation
                if not s.endswith(('.', '!', '?')):
                    s = s + '.'
                uniq.append(s)
                if len(uniq) >= 10:
                    break
            return uniq

        # With embeddings: similarity to description and to centroid
        import numpy as np
        sent_embs = [EMBEDDINGS.embed_query(s) for s in sents]
        M = np.array(sent_embs, dtype=float)
        centroid = M.mean(axis=0)
        def cos(a, b):
            from numpy.linalg import norm
            na = norm(a) or 1.0
            nb = norm(b) or 1.0
            return float(np.dot(a, b) / (na * nb))
        desc_emb = EMBEDDINGS.embed_query(description) if description else None
        scored = []
        for idx, s in enumerate(sents):
            ce = cos(M[idx], centroid)
            ds = cos(M[idx], desc_emb) if desc_emb is not None else 0.0
            # combine with small position prior
            score = 0.7 * ce + 0.3 * ds + 0.1 * (1.0 - idx / max(1, len(sents)-1))
            # penalize too short/too long
            w = len(s.split())
            if w < 8:
                score -= 0.5
            if w > 35:
                score -= 0.2
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate by starting substring
        uniq = []
        seen = set()
        for _, s in scored:
            key = s[:80]
            if key in seen:
                continue
            seen.add(key)
            if not s.endswith(('.', '!', '?')):
                s = s + '.'
            # Capitalize start
            s = s[0].upper() + s[1:]
            uniq.append(s)
            if len(uniq) >= 10:
                break
        return uniq
    except Exception:
        return []


@router.post("/cases/analyze")
async def analyze_case(
    doc_id: str = Form(...),
    title: Optional[str] = Form(None),
    full_text: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Analyze a case: return AI summary, similarity to user's description, and top 10 key points.
    Input via application/x-www-form-urlencoded. Returns JSON with fields:
      { success, summary, similarity_score (0..1), similarity_percent, key_points[], cached }
    """
    if not full_text or not full_text.strip():
        raise HTTPException(status_code=400, detail="'full_text' is required for analysis")

    # Check cache
    key = _cache_key_for_analysis(doc_id, full_text, description)
    now = time.time()
    cached = _ANALYSIS_CACHE.get(key)
    if cached and now - cached.get("_ts", 0) < _ANALYSIS_TTL_SEC:
        return {**cached, "cached": True}

    # Compute similarity if description provided
    similarity = None
    similarity_percent = None
    try:
        if description and EMBEDDINGS is not None:
            q = description.strip()
            d = _truncate_for_embedding(full_text, 4000)
            qe = EMBEDDINGS.embed_query(q)
            de = EMBEDDINGS.embed_query(d)
            sim = _cosine(qe, de)
            similarity = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # map [-1,1] -> [0,1]
            similarity_percent = int(round(similarity * 100))
    except Exception:
        similarity = None
        similarity_percent = None

    # Build LLM summary and key points (with safe fallbacks)
    summary = None
    key_points = []
    used_llm = False
    try:
        if GROQ_API_KEY and ChatGroq is not None:
            used_llm = True
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
            prompt = (
                "You are an expert legal analyst. Analyze the Indian judgment text and respond in the EXACT format below.\n\n"
                "FORMAT:\n"
                "Summary:\n"
                "- 6 to 10 complete sentences covering facts, issues, holding, and reasoning.\n\n"
                "Key Points:\n"
                "1. <short, self-contained point in <= 25 words>\n"
                "2. <short, self-contained point in <= 25 words>\n"
                "3. ... up to 10 points\n\n"
                "Rules:\n"
                "- Use only information present in the text.\n"
                "- Each key point must be a complete sentence that stands alone (no fragments).\n"
                "- Avoid quoting raw dates/SLP numbers unless necessary for meaning.\n"
                "- Prefer substantive legal findings, reasoning, and outcomes over procedural metadata.\n\n"
                f"Judgment Title: {title or doc_id}\n"
                f"User Description (optional): {description or '—'}\n\n"
                "Judgment Text (may be truncated for analysis):\n" + _truncate_for_embedding(full_text, 8000)
            )
            resp = await llm.ainvoke(prompt) if hasattr(llm, 'ainvoke') else llm.invoke(prompt)
            content = resp.content if hasattr(resp, 'content') else str(resp)
            # Parse sections
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            # Extract summary between 'Summary:' and 'Key Points:'
            try:
                si = lines.index('Summary:') if 'Summary:' in lines else -1
                ki = lines.index('Key Points:') if 'Key Points:' in lines else -1
            except ValueError:
                si, ki = -1, -1
            if si != -1 and ki != -1 and ki > si:
                summary_text = " ".join(lines[si+1:ki])
            else:
                summary_text = " ".join(lines[:15])
            import re as _re
            sentences = _re.split(r"(?<=[.!?])\s+", summary_text)
            summary = " ".join([s for s in sentences if s][:8]).strip()[:2000]

            # Extract numbered key points
            kp: list[str] = []
            in_kp = False
            for ln in lines:
                if ln == 'Key Points:':
                    in_kp = True
                    continue
                if in_kp:
                    if ln[:2].isdigit() or ln.startswith('-') or ln.startswith('•'):
                        # remove leading markers like '1. ', '- ', '• '
                        cleaned = _re.sub(r"^(\d+\.|[-•])\s*", "", ln).strip()
                        if cleaned:
                            # enforce sentence-like shape
                            if not cleaned.endswith(('.', '!', '?')):
                                cleaned += '.'
                            cleaned = cleaned[0].upper() + cleaned[1:]
                            kp.append(cleaned)
                    elif kp and len(kp) >= 10:
                        break
            if kp:
                key_points = kp[:10]
    except Exception:
        used_llm = False

    # Fallback heuristics if LLM not available or content empty
    if not summary:
        # Use first ~1200 chars as pseudo-summary with some trimming
        text = full_text.strip()
        summary = _truncate_for_embedding(text, 1200)
    if not key_points:
        # Robust heuristic key points
        text = _truncate_for_embedding(full_text, 8000)
        key_points = _select_key_points(text, description)

    result = {
        "success": True,
        "summary": summary,
        "similarity_score": similarity,
        "similarity_percent": similarity_percent,
        "key_points": key_points,
        "used_llm": used_llm,
        "cached": False,
    }
    _ANALYSIS_CACHE[key] = {**result, "_ts": now}
    return result
