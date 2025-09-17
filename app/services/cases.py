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
            headers={
                "User-Agent": "NyaySarthi/1.0 Legal Research Assistant",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9",
            },
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
    Enhancements:
    - Try multiple query variants (exact phrase and broad)
    - Paginate through pages until `limit` is satisfied (up to 3 pages)
    - Robust title extraction from result blocks
    """
    await init_cases_service()
    query = (query or "").strip()
    if not query:
        return []

    base_url = "https://indiankanoon.org/search/"

    # Two query strategies: exact phrase, then broad
    query_variants = [f'"{query}"', query]

    import re

    def extract_results(soup: BeautifulSoup, acc: List[CaseDoc], seen: set, needed: int) -> None:
        # Prefer structured result titles if present
        # Many results are within elements with class result_title containing anchors to /doc/<id>/
        for blk in soup.select('.result_title a[href^="/doc/"]'):
            href = blk.get('href', '')
            m = re.match(r"^/doc/(\d+)/?", href)
            if not m:
                continue
            doc_id = m.group(1)
            if doc_id in seen:
                continue
            # Prefer the visible title text in the result_title container rather than the anchor ('Full Document')
            parent_title_el = blk.find_parent(class_='result_title')
            parent_text = parent_title_el.get_text(" ", strip=True) if parent_title_el else ""
            # Remove common link label
            import re as _re
            cleaned_parent = _re.sub(r"\bFull\s+Document\b", "", parent_text, flags=_re.IGNORECASE).strip()
            title = cleaned_parent
            # If still empty or generic, search for a better anchor text within the result block
            if not title or title.lower() == '':
                container = blk.find_parent(class_='result') or blk.find_parent('li') or blk.parent
                if container:
                    best = None
                    for aa in container.find_all('a', href=True):
                        text = aa.get_text(strip=True)
                        href = aa['href']
                        if not text or len(text) < 6:
                            continue
                        # Skip the full document link and purely navigational/meta links
                        if _re.search(r"full\s*document|citation|cited by|download|pdf|order|more", text, _re.IGNORECASE):
                            continue
                        if href.startswith('/doc/'):
                            # Likely the full doc; skip if text looks generic
                            if _re.search(r"full\s*document", text, _re.IGNORECASE):
                                continue
                        # Pick the first reasonably descriptive text
                        best = text
                        break
                    if best:
                        title = best
            if not title:
                title = blk.get_text(strip=True) or f"Case {doc_id}"
            # Try to collect nearby meta info (court/date/citation)
            court = None
            date = None
            citation = None
            result_container = blk.find_parent(class_='result') or blk.find_parent('li') or blk.parent
            if result_container:
                meta_text = result_container.get_text('\n', strip=True)
                import re as _re
                m1 = _re.search(r'(?:Court|Bench):\s*([^\n]+)', meta_text, _re.IGNORECASE)
                if m1:
                    court = m1.group(1).strip()
                m2 = _re.search(r'(?:Date|Judgment Date|On):\s*([0-9]{1,2}[-/][A-Za-z]{3,}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}|[A-Za-z]{3,}\s+\d{1,2},\s*\d{2,4})', meta_text, _re.IGNORECASE)
                if m2:
                    date = m2.group(1).strip()
                m3 = _re.search(r'(?:Citation|Citations?):\s*([^\n]+)', meta_text, _re.IGNORECASE)
                if m3:
                    citation = m3.group(1).strip()
            acc.append(CaseDoc(id=doc_id, title=title, url=f"https://indiankanoon.org/doc/{doc_id}/", court=court, date=date, citation=citation))
            seen.add(doc_id)
            if len(acc) >= needed:
                return

        # Fallback: scan all anchors
        for a in soup.find_all('a', href=True):
            href = a['href']
            m = re.match(r"^/doc/(\d+)/?", href)
            if not m:
                continue
            doc_id = m.group(1)
            if doc_id in seen:
                continue
            # Try to get a meaningful title: prefer enclosing result title container
            title = a.get_text(strip=True)
            if not title:
                parent_title = a.find_parent(class_='result_title')
                if parent_title:
                    title = parent_title.get_text(" ", strip=True)
            title = title or f"Case {doc_id}"
            acc.append(CaseDoc(id=doc_id, title=title, url=f"https://indiankanoon.org/doc/{doc_id}/"))
            seen.add(doc_id)
            if len(acc) >= needed:
                return

    results: List[CaseDoc] = []
    seen_ids: set = set()

    try:
        for qv in query_variants:
            if len(results) >= limit:
                break
            # Try first 3 pages or until we have enough
            for page in range(0, 3):
                if len(results) >= limit:
                    break
                params = {
                    "formInput": qv,
                    "pagenum": page,
                    "sortby": "mostrecent" if page == 0 else "relevance",
                    "type": "judgments",
                    "fromdate": "01-01-1950",
                    "from": "01-01-1950",
                    "to": datetime.now().strftime("%d-%m-%Y"),
                }
                resp = await HTTP_CLIENT.get(base_url, params=params, follow_redirects=True)
                # If a page fetch fails, try next strategy/page rather than bailing out
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                extract_results(soup, results, seen_ids, limit)

        return results[:limit]
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
        # Enrich by visiting the case page to extract precise title/court/date/citation
        try:
            sem = asyncio.Semaphore(5)
            async def enrich(c: CaseDoc) -> CaseDoc:
                async with sem:
                    details = await get_case_details_async(c.id)
                    if details.get("success"):
                        # Only overwrite when missing or generic
                        title = (details.get("title") or "").strip()
                        if title and (not c.title or c.title.lower().startswith("full document") or c.title.lower().startswith("case ")):
                            c.title = title
                        c.court = c.court or details.get("court")
                        c.date = c.date or details.get("date")
                        c.citation = c.citation or details.get("citation")
                return c
            scraped = await asyncio.gather(*(enrich(c) for c in scraped))
        except Exception as e:
            logger.warning(f"[CASES] Enrichment failed: {e}")
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

        # Title extraction: try known selectors first
        title_el = soup.select_one('h1.doctitle') or soup.select_one('h1.title') or soup.find('title') or soup.find('h1')
        title = title_el.get_text(strip=True) if title_el else None
        # Clean suffix " - Indian Kanoon"
        if title:
            import re as _re
            title = _re.sub(r"\s*-\s*Indian\s*Kanoon\s*$", "", title, flags=_re.IGNORECASE)

        # Meta fields
        court = None
        date = None
        citation = None

        # Preferred meta container on IK
        doc_meta = soup.find('div', class_='doc_meta')
        if doc_meta:
            meta_text = doc_meta.get_text('\n', strip=True)
            import re as _re
            m = _re.search(r'(?:Bench|Court):\s*([^\n]+)', meta_text, _re.IGNORECASE)
            if m:
                court = m.group(1).strip()
            m = _re.search(r'(?:Judgment Date|On):\s*([^\n]+)', meta_text, _re.IGNORECASE)
            if m:
                date = m.group(1).strip()
            m = _re.search(r'Citation:\s*([^\n]+)', meta_text, _re.IGNORECASE)
            if m:
                citation = m.group(1).strip()
        else:
            # Fallback to generic info sections
            info_div = soup.find(id="info") or soup.find("div", class_="info")
            if info_div:
                text = info_div.get_text("\n", strip=True)
                for line in text.splitlines():
                    low = line.lower()
                    if not court and ("court" in low or "bench" in low):
                        court = line.split(":", 1)[-1].strip() if ":" in line else line
                    if not date and ("date" in low or "judgment" in low):
                        date = line.split(":", 1)[-1].strip() if ":" in line else line
                    if not citation and ("citation" in low or "reported" in low):
                        citation = line.split(":", 1)[-1].strip() if ":" in line else line

        # Extract main content
        full_text = None
        full_text_html = None
        try:
            # 1) Prefer pre#pre_1 or any element with id='pre_1'
            pre_main = soup.select_one('pre#pre_1') or soup.select_one('#pre_1')
            if pre_main:
                full_text_html = str(pre_main)
                full_text = pre_main.get_text('\n', strip=True)
            else:
                pre_tags = soup.find_all('pre')
                if pre_tags:
                    # Concatenate ALL <pre> blocks in DOM order
                    texts = []
                    htmls = []
                    for el in pre_tags:
                        texts.append(el.get_text('\n', strip=True))
                        htmls.append(str(el))
                    full_text = '\n\n'.join([t for t in texts if t])
                    full_text_html = '\n'.join(htmls) if htmls else None
                else:
                    # 2) Fallback: specific containers first (avoid entire body which brings boilerplate)
                    content_selectors = [
                        '.judgments', '.judgment', '.doc_content', '.doc', '#content', '.content'
                    ]
                    content_el = None
                    for sel in content_selectors:
                        content_el = soup.select_one(sel)
                        if content_el:
                            break
                    if content_el:
                        for el in content_el.select('script, style, noscript, .noprint, .ad, .ad-container, .disclaimer, .hidden-print, header, footer, nav'):
                            el.decompose()
                        full_text_html = str(content_el)
                        paragraphs = []
                        for p in content_el.find_all(['p', 'div', 'span', 'li'], recursive=True):
                            txt = p.get_text(' ', strip=True)
                            if txt and len(txt) > 10:
                                paragraphs.append(txt)
                        if not paragraphs:
                            paragraphs = [content_el.get_text(' ', strip=True)]
                        full_text = '\n\n'.join(paragraphs)
        except Exception as ex:
            logger.warning(f"[CASES] Content extraction warning for {doc_id}: {ex}")

        # Judges (best-effort)
        judges = None
        try:
            meta_block = soup.find('div', class_='doc_meta') or soup
            import re as _re
            jt = meta_block.get_text('\n', strip=True) if meta_block else ''
            jm = _re.search(r'(?:Before|Bench|Coram)\s*[:\-]?\s*([^\n]+)', jt, _re.IGNORECASE)
            if jm:
                judges = jm.group(1).strip()
        except Exception:
            pass

        # Always try print view (often contains the complete single-page text)
        try:
            print_url = f"https://indiankanoon.org/doc/{doc_id}/?type=print"
            resp2 = await HTTP_CLIENT.get(print_url, follow_redirects=True)
            if resp2.status_code == 200:
                soup2 = BeautifulSoup(resp2.text, 'html.parser')
                pre_main2 = soup2.select_one('pre#pre_1') or soup2.select_one('#pre_1')
                ft_text = None
                ft_html = None
                if pre_main2:
                    ft_html = str(pre_main2)
                    ft_text = pre_main2.get_text('\n', strip=True)
                else:
                    pre_tags2 = soup2.find_all('pre')
                    if pre_tags2:
                        texts2 = []
                        htmls2 = []
                        for el in pre_tags2:
                            texts2.append(el.get_text('\n', strip=True))
                            htmls2.append(str(el))
                        ft_text = '\n\n'.join([t for t in texts2 if t])
                        ft_html = '\n'.join(htmls2) if htmls2 else None
                # Prefer print view if it yields equal or more content
                if ft_text and (not full_text or len(ft_text) >= len(full_text)):
                    full_text = ft_text
                    full_text_html = ft_html or full_text_html
            logger.info(f"[CASES] Content lengths -> normal: {len(full_text) if full_text else 0}, print: {len(ft_text) if 'ft_text' in locals() and ft_text else 0}")
        except Exception as ex:
            logger.warning(f"[CASES] Print-view extraction warning for {doc_id}: {ex}")

        return {
            "success": True,
            "doc_id": doc_id,
            "title": title or f"Case {doc_id}",
            "court": court,
            "date": date,
            "citation": citation,
            "url": url,
            "full_text": full_text,
            "full_text_html": full_text_html,
            "judges": judges,
            "description": description,
        }
    except Exception as e:
        logger.error(f"[CASES] get_case_details_async failed: {e}", exc_info=True)
        # Soft-fail: return a minimal but successful payload so the UI can still render
        return {
            "success": True,
            "doc_id": doc_id or "unknown",
            "title": f"Case {doc_id}" if doc_id else "Case Details",
            "url": url,
            "full_text": None,
            "full_text_html": None,
            "message": f"partial_parse_error: {str(e)}",
        }
