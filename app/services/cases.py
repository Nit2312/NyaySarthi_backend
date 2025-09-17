import os
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import httpx
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HTTP_CLIENT: Optional[httpx.AsyncClient] = None
IK_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")
IK_EMAIL = os.getenv("INDIAN_KANOON_EMAIL", "")

class CaseDoc(BaseModel):
    id: str
    title: str
    court: Optional[str] = None
    date: Optional[str] = None
    citation: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None

async def init_cases_service():
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            headers={"User-Agent": "NyaySarthi/1.0 Legal Research Assistant"},
        )
        logger.info("[CASES] HTTP client initialized")

async def shutdown_cases_service():
    global HTTP_CLIENT
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None
        logger.info("[CASES] HTTP client closed")

async def scrape_indian_kanoon_search_async(query: str, limit: int = 5) -> List[CaseDoc]:
    """Scrape Indian Kanoon search results and return real case entries.
    Parses result links to /doc/<id>/ and extracts titles.
    """
    await init_cases_service()
    query = (query or "").strip()
    if not query:
        return []

    base_url = "https://indiankanoon.org/search/"
    params = {
        "formInput": f'"{query}"',
        "pagenum": 0,
        "sortby": "mostrecent",
        "type": "judgments",
        "fromdate": "01-01-1950",
        "from": "01-01-1950",
        "to": datetime.now().strftime("%d-%m-%Y"),
    }
    try:
        resp = await HTTP_CLIENT.get(base_url, params=params, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, 'html.parser')

        # Find anchors that link to individual judgments: /doc/<id>/
        import re
        results: List[CaseDoc] = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            m = re.match(r"^/doc/(\d+)/?", href)
            if not m:
                continue
            doc_id = m.group(1)
            title = a.get_text(strip=True) or f"Case {doc_id}"
            # Build absolute URL
            url = f"https://indiankanoon.org/doc/{doc_id}/"
            # Avoid duplicates
            if any(r.id == doc_id for r in results):
                continue
            results.append(CaseDoc(id=doc_id, title=title, url=url))
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.warning(f"[CASES] Scrape failed: {e}")
        return []

async def search_indian_kanoon_async(query: str, limit: int = 5) -> Tuple[List[CaseDoc], Optional[str]]:
    query = (query or "").strip()
    if not query:
        return [], "empty_query"

    # Try scraping first (no credentials needed)
    scraped = await scrape_indian_kanoon_search_async(query, min(limit, 5))
    if scraped:
        return scraped, None

    # Fallback to API if configured
    if not IK_EMAIL or not IK_API_KEY:
        return [], "no_credentials"

    try:
        params = {
            "formInput": query,
            "pagenum": 0,
            "sort_by": "relevance",
            "type": "judgments",
            "from": "01-01-1950",
            "to": datetime.now().strftime("%d-%m-%Y"),
            "format": "json",
        }
        if not HTTP_CLIENT:
            await init_cases_service()
        resp = await HTTP_CLIENT.get(
            "https://api.indiankanoon.org/search/",
            params=params,
            auth=(IK_EMAIL, IK_API_KEY),
        )
        if resp.status_code == 200:
            data = resp.json()
            docs: List[CaseDoc] = []
            for d in data.get("docs", [])[:limit]:
                doc_id = str(d.get("id", "")).strip()
                if not doc_id:
                    continue
                docs.append(
                    CaseDoc(
                        id=doc_id,
                        title=(d.get("title") or "").strip() or f"Case {doc_id}",
                        court=(d.get("court") or d.get("docsource") or None),
                        date=d.get("date") or None,
                        citation=d.get("citation") or None,
                        url=f"https://indiankanoon.org/doc/{doc_id}/",
                    )
                )
            if docs:
                return docs, None
            return [], "no_results"
        elif resp.status_code == 401:
            return [], "auth_failed"
        elif resp.status_code == 429:
            return [], "rate_limit"
        return [], f"api_error_{resp.status_code}"
    except Exception as e:
        logger.error(f"[CASES] API search failed: {e}", exc_info=True)
        return [], "unexpected_error"


async def get_case_details_async(doc_id: str, description: Optional[str] = None) -> Dict[str, Any]:
    """Fetch details for a specific Indian Kanoon case by doc_id.
    Returns a normalized response compatible with the frontend proxy expectations.
    """
    import re
    doc_id = (doc_id or "").strip()
    # Accept full URLs like https://indiankanoon.org/doc/1234567/ and extract the numeric id
    m = re.search(r"indiankanoon\.org/Doc/(\d+)|indiankanoon\.org/doc/(\d+)", doc_id, re.IGNORECASE)
    if m:
        doc_id = m.group(1) or m.group(2)
    if not doc_id:
        return {"success": False, "error": "missing_doc_id"}

    if not HTTP_CLIENT:
        await init_cases_service()

    url = f"https://indiankanoon.org/doc/{doc_id}/"
    try:
        resp = await HTTP_CLIENT.get(url, follow_redirects=True)
        if resp.status_code == 404:
            return {"success": False, "error": "not_found", "message": "Case not found", "url": url}
        resp.raise_for_status()

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # Best-effort extraction
        title = (soup.find("h1") or soup.find("title") or {}).get_text(strip=True) if soup else None
        # Meta fields are not consistent; extract common labels if present
        court = None
        date = None
        citation = None

        # Attempt to find common info sections
        info_div = soup.find(id="info") or soup.find("div", class_="info")
        if info_div:
            text = info_div.get_text("\n", strip=True)
            # crude parsing
            for line in text.splitlines():
                low = line.lower()
                if not court and ("court" in low or "bench" in low):
                    court = line.split(":", 1)[-1].strip() if ":" in line else line
                if not date and ("date" in low or "judgment" in low):
                    date = line.split(":", 1)[-1].strip() if ":" in line else line
                if not citation and ("citation" in low or "reported" in low):
                    citation = line.split(":", 1)[-1].strip() if ":" in line else line

        return {
            "success": True,
            "doc_id": doc_id,
            "title": title or f"Case {doc_id}",
            "court": court,
            "date": date,
            "citation": citation,
            "url": url,
            "description": description,
        }
    except Exception as e:
        logger.error(f"[CASES] get_case_details_async failed: {e}", exc_info=True)
        return {"success": False, "error": "exception", "message": str(e), "url": url}
