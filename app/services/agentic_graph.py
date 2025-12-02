import os
import logging
import json
import time
from typing import Any, Dict, List, Optional

from app.services.rag import GROQ_API_KEY, GROQ_MODEL, chat_handler as constitution_answer
from app.services.cases import search_indian_kanoon_async, get_case_details_async

logger = logging.getLogger(__name__)

# In-memory cache for case searches (30 minute TTL)
CASE_CACHE = {}
CACHE_TTL = 30 * 60  # 30 minutes

LANGGRAPH_AVAILABLE = True
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_groq import ChatGroq
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
except Exception as e:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False
    logger.warning("[AGENTIC] LangGraph/LangChain not fully available: %s", e)


# ---------------- Tools ----------------

if LANGGRAPH_AVAILABLE:
    @tool("constitution_answer")
    async def constitution_tool(question: str, conversation_id: Optional[str] = None) -> str:
        """Answer constitutional questions about Indian Constitution articles, fundamental rights, directive principles, etc."""
        try:
            # Enhance question with constitutional context
            enhanced_question = f"Based on the Indian Constitution: {question}"
            resp = await constitution_answer(enhanced_question, conversation_id)
            if isinstance(resp, dict):
                answer = str(resp.get("response") or resp.get("detail") or "")
                if answer and len(answer.strip()) > 10:
                    return f"CONSTITUTIONAL REFERENCE:\n{answer}\n\nSource: Indian Constitution via RAG system"
                else:
                    return "No relevant constitutional provisions found for this query."
            return str(resp)
        except Exception as e:
            logger.error("[AGENTIC] constitution_tool error: %s", e, exc_info=True)
            return "I could not retrieve constitutional information. Please rephrase your question about specific articles or constitutional provisions."

    @tool("find_cases_and_answer")
    async def cases_tool(question: str, description: Optional[str] = None) -> str:
        """Find top 5 relevant Indian legal cases from Indian Kanoon and provide detailed analysis with caching."""
        try:
            # Check cache first
            cache_key = f"cases_{hash(question + (description or ''))}"
            current_time = time.time()
            
            if cache_key in CASE_CACHE:
                cached_data, timestamp = CASE_CACHE[cache_key]
                if current_time - timestamp < CACHE_TTL:
                    logger.info(f"[AGENTIC] Using cached cases for: {question[:50]}...")
                    return cached_data
                else:
                    # Remove expired cache
                    del CASE_CACHE[cache_key]
            
            # Build a focused search query
            import re as _re
            q0 = question if not description else f"{question}\n{description}"
            
            # Enhanced case name detection
            case_patterns = [
                r"([A-Z][A-Za-z .'-]+\s+v\.?s?\.?\s+[A-Z][A-Za-z .'-]+)",
                r"([A-Z][A-Za-z .'-]+\s+vs?\.?\s+[A-Z][A-Za-z .'-]+)"
            ]
            
            docs = []
            seen = set()
            
            # Try specific case name patterns first
            for pattern in case_patterns:
                m = _re.search(pattern, q0)
                if m:
                    case_name = m.group(1).strip()
                    variants = [f'"{case_name}"', case_name]
                    
                    # Add constitutional context if relevant
                    if any(art in q0.lower() for art in ['article', 'fundamental right', 'constitution']):
                        art_match = _re.search(r'article\s+(\d+)', q0.lower())
                        if art_match:
                            variants.insert(0, f'"{case_name}" "Article {art_match.group(1)}"')
                    
                    for v in variants:
                        if len(docs) >= 5:
                            break
                        part = await search_indian_kanoon_async(v, limit=5)
                        for d in part:
                            if d.id not in seen and len(docs) < 5:
                                docs.append(d)
                                seen.add(d.id)
                    break
            
            # If no specific case found, do general search for top 5
            if not docs:
                docs = await search_indian_kanoon_async(q0, limit=5)
            
            if not docs:
                result = "No relevant cases were found in Indian Kanoon database. Please try rephrasing your query with specific case names, legal concepts, or constitutional articles."
                CASE_CACHE[cache_key] = (result, current_time)
                return result
            
            # Get details for multiple cases (top 3 for performance)
            case_details = []
            for d in docs[:3]:
                try:
                    tmp = await get_case_details_async(d.id)
                    if tmp and tmp.get("success"):
                        case_details.append((d, tmp))
                except Exception as e:
                    logger.warning(f"[AGENTIC] Failed to get details for case {d.id}: {e}")
                    continue
            
            if not case_details:
                result = "Found case references but could not retrieve detailed information. The cases may be:"
                for i, d in enumerate(docs[:5], 1):
                    result += f"\n{i}. {getattr(d, 'title', 'Unknown case')} (ID: {d.id})"
                result += "\n\nPlease try a more specific search or check Indian Kanoon directly."
                CASE_CACHE[cache_key] = (result, current_time)
                return result

            # Process the main case (first with details)
            main_case, main_details = case_details[0]
            title = main_details.get("title") or main_details.get("doc_id") or "Unknown Case"
            url = main_details.get("url", "")
            court = main_details.get("court") or ""
            date = main_details.get("date") or ""
            citation = main_details.get("citation") or ""
            judges = main_details.get("judges") or ""
            full_text = main_details.get("full_text") or ""

            # Prepare as-much-as-possible judgment text for LLM (prioritize completeness)
            def _prepare_llm_text(txt: str) -> str:
                t = (txt or "").strip()
                MAX = 24000  # generous cap to keep within reasonable token limits for instant models
                if len(t) <= MAX:
                    return t
                # When too long, pick: intro, sections with keywords, and conclusion
                import re as _re
                parts: list[str] = []
                # 1) Intro (first 6k)
                parts.append(t[:6000])
                # 2) Keyword windows (Article 21, holding, conclusion, ratio)
                keys = [r"article\s*21", r"holding", r"conclusion", r"ratio", r"issue", r"reasoning"]
                for k in keys:
                    for m in _re.finditer(k, t, flags=_re.IGNORECASE):
                        start = max(0, m.start() - 2000)
                        end = min(len(t), m.end() + 2000)
                        parts.append(t[start:end])
                        if sum(len(p) for p in parts) > MAX - 6000:
                            break
                    if sum(len(p) for p in parts) > MAX - 4000:
                        break
                # 3) Ending (last 6k)
                parts.append(t[-6000:])
                # Deduplicate overlaps
                joined = "\n\n---\n\n".join(parts)
                return joined[:MAX]

            llm_text = _prepare_llm_text(full_text)

            if GROQ_API_KEY and full_text:
                try:
                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
                    
                    # Build comprehensive case analysis including related cases
                    related_cases_info = ""
                    if len(case_details) > 1:
                        related_cases_info = "\n\nRELATED CASES FOUND:\n"
                        for i, (related_case, related_detail) in enumerate(case_details[1:], 2):
                            related_title = related_detail.get("title", "Unknown")
                            related_court = related_detail.get("court", "")
                            related_date = related_detail.get("date", "")
                            related_citation = related_detail.get("citation", "")
                            related_cases_info += f"{i}. {related_title}\n   Court: {related_court} | Date: {related_date} | Citation: {related_citation}\n"
                    
                    # Additional cases from search results
                    if len(docs) > len(case_details):
                        related_cases_info += "\nADDITIONAL RELEVANT CASES:\n"
                        for i, d in enumerate(docs[len(case_details):], len(case_details) + 1):
                            case_title = getattr(d, 'title', 'Unknown case')
                            related_cases_info += f"{i}. {case_title} (ID: {d.id})\n"
                    
                    prompt = (
                        "You are an expert Indian legal case analyst. Using ONLY the judgment text provided, create a comprehensive case brief.\n"
                        "STRICT REQUIREMENTS:\n"
                        "- Base your analysis ONLY on the provided judgment text\n"
                        "- Do not add information not present in the text\n"
                        "- If information is missing, state 'Not specified in judgment'\n"
                        "- Always include source URL for verification\n\n"
                        "MAIN CASE ANALYSIS:\n"
                        f"Title: {title}\n"
                        f"Court: {court}\n"
                        f"Date: {date}\n"
                        f"Citation: {citation}\n"
                        f"Source URL: {url}\n"
                        f"Bench/Judges: {judges}\n\n"
                        "CASE BRIEF:\n"
                        "1) FACTS (4-6 sentences from judgment)\n"
                        "2) PROCEDURAL HISTORY (1-3 sentences)\n"
                        "3) LEGAL ISSUES (bullet points, 2-5 items)\n"
                        "4) ARGUMENTS (petitioner and respondent positions)\n"
                        "5) COURT'S REASONING (5-8 sentences with paragraph references)\n"
                        "6) HOLDING/DECISION (clear ruling, 1-3 sentences)\n"
                        "7) RATIO DECIDENDI (legal principle established)\n"
                        "8) KEY TAKEAWAYS (3-5 practical points)\n\n"
                        f"{related_cases_info}\n"
                        "JUDGMENT TEXT (Primary Source):\n" + llm_text
                    )
                    resp = await llm.ainvoke(prompt) if hasattr(llm, 'ainvoke') else llm.invoke(prompt)
                    result = resp.content if hasattr(resp, 'content') else str(resp)
                    
                    # Cache the successful result
                    CASE_CACHE[cache_key] = (result, current_time)
                    return result
                    
                except Exception as e:
                    logger.warning("[AGENTIC] LLM case-brief failed, using heuristic: %s", e)

            # Enhanced fallback with multiple cases
            result = f"MAIN CASE: {title}\n"
            result += f"Court: {court} | Date: {date} | Citation: {citation}\n"
            result += f"Source: {url}\n"
            result += f"Judges: {judges}\n\n"
            
            excerpt = full_text[:2000]
            result += "CASE EXCERPT:\n" + excerpt
            
            if len(case_details) > 1:
                result += "\n\nRELATED CASES:\n"
                for i, (related_case, related_detail) in enumerate(case_details[1:], 2):
                    rel_title = related_detail.get("title", "Unknown")
                    rel_court = related_detail.get("court", "")
                    rel_date = related_detail.get("date", "")
                    result += f"{i}. {rel_title} - {rel_court} ({rel_date})\n"
            
            if len(docs) > len(case_details):
                result += "\nADDITIONAL MATCHES:\n"
                for i, d in enumerate(docs[len(case_details):5], len(case_details) + 1):
                    result += f"{i}. {getattr(d, 'title', 'Case')} (ID: {d.id})\n"
            
            result += "\n(Full case details available at source URLs)"
            
            # Cache the fallback result
            CASE_CACHE[cache_key] = (result, current_time)
            return result
        except Exception as e:
            logger.error("[AGENTIC] cases_tool error: %s", e, exc_info=True)
            return "I encountered an error while searching Indian Kanoon database. Please try rephrasing your query or specify exact case names, constitutional articles, or legal concepts."

    @tool("tavily_web_search")
    async def tavily_tool(query: str) -> str:
        """Use Tavily to fetch recent web results when latest/current info is needed."""
        try:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if not tavily_key:
                return "Web search is unavailable (TAVILY_API_KEY not configured)."
            search = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            results = search.results(query, max_results=5)
            lines = []
            for r in (results or []):
                title = r.get("title")
                url = r.get("url")
                content = (r.get("content") or "")[:300]
                lines.append(f"- {title} ({url})\n  {content}")
            return "Latest web results:\n" + ("\n".join(lines) if lines else "No results.")
        except Exception as e:
            logger.error("[AGENTIC] tavily_tool error: %s", e, exc_info=True)
            return "I could not perform web search due to an internal error."


def build_agent_app():
    """Compile and return the LangGraph app."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph / LangChain are not available in this environment")

    tools = [constitution_tool, cases_tool, tavily_tool]

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured; cannot initialize tool-calling LLM")
    model = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    model_with_tools = model.bind_tools(tools)

    async def tool_calling_llm(state: Dict[str, Any]):
        msgs = state.get("messages", [])
        response = await model_with_tools.ainvoke(msgs) if hasattr(model_with_tools, 'ainvoke') else model_with_tools.invoke(msgs)
        return {"messages": msgs + [response]}

    graph = StateGraph(state_schema=dict)
    graph.add_node("tool_calling_llm", tool_calling_llm)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("tool_calling_llm")
    graph.add_conditional_edges("tool_calling_llm", tools_condition, {"tools": "tools", "end": END})
    graph.add_edge("tools", "tool_calling_llm")

    return graph.compile()


async def run_agentic_chat(input_text: str, prefer: Optional[str] = None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """Run the agent graph (or fallbacks) and return response + metadata."""
    question = (input_text or "").strip()
    if not question:
        return {"response": "", "route": "", "graph": False}

    # Heuristic pre-routing: if the user asks to "describe the case" or mentions case-like patterns, prefer case tool
    def _looks_like_case_query(txt: str) -> bool:
        import re
        t = (txt or "").lower()
        cues = [" v. ", " vs ", " vs. ", "describe the case", "facts of the case", "ratio", "judgment", "citation", "holding", "case details"]
        if any(c in t for c in cues):
            return True
        # Patterns like SCC/AIR citations or SLP numbers
        if re.search(r"\b\d{4}\b.*\bSCC\b", txt):
            return True
        if re.search(r"\bAIR\b\s*\d{4}", txt, re.IGNORECASE):
            return True
        return False

    prefer_cases = _looks_like_case_query(question)

    # Lightweight normalization for common misspellings and noisy inputs
    def _normalize_query(txt: str) -> str:
        t = (txt or "").strip()
        low = t.lower()
        replacements = {
            # Maneka Gandhi common misspellings
            "maheka gnadhi": "maneka gandhi",
            "maheka gandhi": "maneka gandhi",
            "meheka gandhi": "maneka gandhi",
            "maneka ghandi": "maneka gandhi",
            "menaka gandhi": "maneka gandhi",
            # v./vs normalization spacing
            " v s ": " v. ",
            " vs ": " v. ",
        }
        for k, v in replacements.items():
            if k in low:
                low = low.replace(k, v)
        # Restore basic capitalization for prominent case name if present
        if "maneka gandhi" in low and "union of india" not in low:
            low += " v. Union of India"
        # Try to standardize 'article' token spacing
        low = low.replace("art. ", "article ")
        return low

    question = _normalize_query(question)

    # Deprecated: deterministic direct pipeline. We keep it as an internal helper but
    # we will NOT short-circuit to it when LangGraph is available. Tool outputs must
    # be routed back into the LLM which composes the final answer.
    async def _run_cases_pipeline(q: str) -> Dict[str, Any]:
        try:
            import re as _re
            q0 = q
            m = _re.search(r"([A-Z][A-Za-z .'-]+\s+v\.?s?\.?\s+[A-Z][A-Za-z .'-]+)", q0)
            if m:
                case_name = m.group(1).strip()
                variants = [f'"{case_name}"', case_name]
                if 'article 21' in q0.lower():
                    variants.insert(0, f'"{case_name}" "Article 21"')
                docs = []
                seen = set()
                for v in variants:
                    if len(docs) >= 3:
                        break
                    part = await search_indian_kanoon_async(v, limit=3)
                    for d in part:
                        if d.id not in seen:
                            docs.append(d); seen.add(d.id)
                if not docs:
                    docs = await search_indian_kanoon_async(q0, limit=3)
            else:
                docs = await search_indian_kanoon_async(q0, limit=3)
            if not docs:
                return {"response": "No relevant cases were found for this query.", "route": "cases", "graph": False}
            det = None
            for d in docs[:2]:
                tmp = await get_case_details_async(d.id)
                if tmp and tmp.get("success"):
                    det = tmp
                    break
            if not det:
                return {"response": "I found candidate cases but could not retrieve their details.", "route": "cases", "graph": False}
            title = det.get("title") or det.get("doc_id")
            url = det.get("url", "")
            court = det.get("court") or ""
            date = det.get("date") or ""
            citation = det.get("citation") or ""
            judges = det.get("judges") or ""
            full_text = det.get("full_text") or ""
            # Reuse the LLM text preparation
            def _prepare_llm_text(txt: str) -> str:
                t = (txt or "").strip()
                MAX = 24000
                if len(t) <= MAX:
                    return t
                import re as _re
                parts: list[str] = []
                parts.append(t[:6000])
                keys = [r"article\s*21", r"holding", r"conclusion", r"ratio", r"issue", r"reasoning"]
                for k in keys:
                    for m2 in _re.finditer(k, t, flags=_re.IGNORECASE):
                        start = max(0, m2.start() - 2000)
                        end = min(len(t), m2.end() + 2000)
                        parts.append(t[start:end])
                        if sum(len(p) for p in parts) > MAX - 6000:
                            break
                    if sum(len(p) for p in parts) > MAX - 4000:
                        break
                parts.append(t[-6000:])
                joined = "\n\n---\n\n".join(parts)
                return joined[:MAX]
            llm_text = _prepare_llm_text(full_text)
            if GROQ_API_KEY and full_text:
                try:
                    from langchain_groq import ChatGroq
                    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
                    prompt = (
                        "You are a Supreme Court case explainer. Using ONLY the judgment text below, produce a thorough but concise, self-contained case brief.\n"
                        "Do not say 'not in context' or similar; rely strictly on the text provided.\n"
                        "Cite the URL in the header and avoid speculation.\n\n"
                        "OUTPUT FORMAT:\n"
                        f"Title: {title}\n"
                        f"Court: {court}\n"
                        f"Date: {date}\n"
                        f"Citation: {citation}\n"
                        f"URL: {url}\n"
                        f"Bench/Judges: {judges}\n\n"
                        "1) Facts (4-6 sentences)\n"
                        "2) Procedural History (1-3 sentences)\n"
                        "3) Issues (bullet points, 2-5 items)\n"
                        "4) Arguments (briefly for both sides)\n"
                        "5) Reasoning (5-8 sentences with key paragraphs if any)\n"
                        "6) Holding / Final Order (clear, 1-3 sentences; explicitly state the holding under Article 21 if applicable)\n"
                        "7) Ratio Decidendi (1-3 sentences)\n"
                        "8) Key Takeaways (3-5 bullets)\n\n"
                        "Judgment Text (may be truncated):\n" + llm_text
                    )
                    resp = await llm.ainvoke(prompt) if hasattr(llm, 'ainvoke') else llm.invoke(prompt)
                    text = resp.content if hasattr(resp, 'content') else str(resp)
                    return {"response": text, "route": "cases", "graph": False}
                except Exception:
                    pass
            excerpt = full_text[:1800]
            meta = f"{title} — {court} ({date}) {citation}\nSource: {url}\nJudges: {judges}\n"
            body = "Facts/Excerpt:\n" + excerpt + ("\n\n(For full reasoning, open the source URL.)" if url else "")
            return {"response": meta + "\n" + body, "route": "cases", "graph": False}
        except Exception as e:
            logger.error("[AGENTIC] _run_cases_pipeline error: %s", e, exc_info=True)
            return {"response": "Agent is unavailable", "route": "cases", "graph": False}

    # Do not bypass the graph. We will pass a hint to the tool-calling LLM instead.
    prefer_hint = None
    if prefer:
        prefer_hint = prefer.strip().lower()
    elif prefer_cases:
        prefer_hint = 'cases'

    # Fallback path if graph not available
    if not LANGGRAPH_AVAILABLE:
        try:
            base = await constitution_answer(question, conversation_id)
            if isinstance(base, dict) and base.get("response"):
                return {"response": base.get("response"), "route": "constitution", "graph": False}
        except Exception:
            pass
        try:
            # simple reuse of the tool function in fallback style
            async def _cases(question: str) -> str:
                docs = await search_indian_kanoon_async(question, limit=3)
                if not docs:
                    return "No relevant cases were found for this query."
                det = await get_case_details_async(docs[0].id)
                if not det or not det.get("success"):
                    return "Could not fetch case details."
                title = det.get("title") or det.get("doc_id")
                url = det.get("url", "")
                excerpt = (det.get("full_text") or "")[:1000]
                return f"Candidate: {title} ({url})\n\n{excerpt}"
            resp_text = await _cases(question)
            return {"response": resp_text, "route": "cases", "graph": False}
        except Exception as e:
            logger.error("[AGENTIC] fallback error: %s", e, exc_info=True)
            return {"response": "Agent is unavailable", "route": "", "graph": False}

    try:
        app = build_agent_app()
        sys_instructions = (
            "You are a precise Indian legal assistant with access to Constitution and case law databases.\n\n"
            "CRITICAL ACCURACY RULES:\n"
            "• NEVER make up facts, case names, or legal provisions\n"
            "• ALWAYS use tools for factual information - never rely on training data\n"
            "• If tools return no results, say so clearly - don't guess or fabricate\n"
            "• Cite exact sources: Article numbers, case names, URLs when provided\n"
            "• State limitations clearly when confidence is low\n\n"
            "TOOL USAGE REQUIREMENTS:\n"
            "1. CONSTITUTIONAL QUERIES: Use 'constitution_answer' for:\n"
            "   - Article references (Article 21, Article 14, etc.)\n"
            "   - Fundamental rights questions\n"
            "   - Constitutional interpretation\n"
            "   - Directive principles\n\n"
            "2. CASE LAW QUERIES: Use 'find_cases_and_answer' for:\n"
            "   - Specific case names (X v. Y format)\n"
            "   - Legal precedents\n"
            "   - Court judgments\n"
            "   - Ratio decidendi\n"
            "   - Case facts, holdings, reasoning\n\n"
            "3. COMBINED QUERIES: Use BOTH tools when question involves constitutional articles AND case law\n\n"
            "4. RECENT EVENTS: Use 'tavily_web_search' for current legal news\n\n"
            "RESPONSE STRUCTURE:\n"
            "• Acknowledge the user's question briefly\n"
            "• Present factual information from tools\n"
            "• Cite all sources (Articles, case names, URLs)\n"
            "• End with practical takeaway\n"
            "• Include disclaimer for legal advice\n\n"
            "ANTI-HALLUCINATION MEASURES:\n"
            "• If tools return empty/unclear results, state this explicitly\n"
            "• Never supplement with 'general knowledge' - stick to tool outputs\n"
            "• Use qualifying language: 'According to the case law database...'\n"
            "• Correct obvious spelling errors in user input before using tools\n"
        )
        messages: List[Any] = [SystemMessage(content=sys_instructions), HumanMessage(content=question)]
        # Prefer hints (non-binding). If query contains both article and case cues, hint to combine.
        import re as _re
        has_article = bool(_re.search(r"\barticle\s*\d+\b|\bअनुच्छेद\s*\d+\b", question, _re.IGNORECASE))
        has_case = bool(_re.search(r"\bv\.?s?\.?\b| vs |citation|holding|ratio|judgment", question, _re.IGNORECASE))
        if has_article and has_case:
            messages.insert(1, HumanMessage(content="HINT: Prefer tools constitution + cases (combine)"))
        elif prefer_hint:
            messages.insert(1, HumanMessage(content=f"HINT: Prefer tool {prefer_hint}"))
        result = await app.ainvoke({"messages": messages}) if hasattr(app, 'ainvoke') else app.invoke({"messages": messages})
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        final_text = None
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                final_text = m.content
                break
        # If the model returned empty or trivial, provide a graceful message rather than raw tool output
        if not final_text or len(final_text.strip()) < 10:
            final_text = "I couldn't generate a complete answer at the moment. Please try rephrasing your question or ask for specific constitutional provisions or case details."
        return {"response": final_text or "", "route": "agentic", "graph": True}
    except Exception as e:
        logger.error("[AGENTIC] graph error: %s", e, exc_info=True)
        return {"response": "Failed to run agent graph", "route": "", "graph": False}
