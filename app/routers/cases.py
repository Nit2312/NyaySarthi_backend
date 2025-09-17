from fastapi import APIRouter, Form, HTTPException
from typing import Optional

from app.services.cases import search_indian_kanoon_async, get_case_details_async

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
