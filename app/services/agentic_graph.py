import os
import logging
from typing import Any, Dict, List, Optional

from app.services.rag import GROQ_API_KEY, GROQ_MODEL, chat_handler as constitution_answer
from app.services.cases import search_indian_kanoon_async, get_case_details_async

logger = logging.getLogger(__name__)

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
        try:
            resp = await constitution_answer(question, conversation_id)
            if isinstance(resp, dict):
                return str(resp.get("response") or resp.get("detail") or "")
            return str(resp)
        except Exception as e:
            logger.error("[AGENTIC] constitution_tool error: %s", e, exc_info=True)
            return "I could not retrieve a constitution-based answer due to an internal error."

    @tool("find_cases_and_answer")
    async def cases_tool(question: str, description: Optional[str] = None) -> str:
        try:
            # Build a focused search query if a case name pattern is present
            import re as _re
            q0 = question if not description else f"{question}\n{description}"
            m = _re.search(r"([A-Z][A-Za-z .'-]+\s+v\.?s?\.?\s+[A-Z][A-Za-z .'-]+)", q0)
            if m:
                case_name = m.group(1).strip()
                # Focused variants: exact case name, plus article mention if present
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
                return "No relevant cases were found for this query."
            # Prefer to give a full, structured description of the top case
            det_main = None
            for d in docs[:2]:
                tmp = await get_case_details_async(d.id)
                if tmp and tmp.get("success"):
                    det_main = tmp
                    break
            if not det_main:
                return "I found candidate cases but could not retrieve their details."

            title = det_main.get("title") or det_main.get("doc_id")
            url = det_main.get("url", "")
            court = det_main.get("court") or ""
            date = det_main.get("date") or ""
            citation = det_main.get("citation") or ""
            judges = det_main.get("judges") or ""
            full_text = det_main.get("full_text") or ""

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
                    return resp.content if hasattr(resp, 'content') else str(resp)
                except Exception as e:
                    logger.warning("[AGENTIC] LLM case-brief failed, using heuristic: %s", e)

            # Heuristic structured fallback
            excerpt = full_text[:1800]
            meta = f"{title} — {court} ({date}) {citation}\nSource: {url}\nJudges: {judges}\n"
            body = (
                "Facts/Excerpt:\n" + excerpt + ("\n\n(For full reasoning, open the source URL.)" if url else "")
            )
            return meta + "\n" + body
        except Exception as e:
            logger.error("[AGENTIC] cases_tool error: %s", e, exc_info=True)
            return "I could not retrieve case-based information due to an internal error."

    @tool("tavily_web_search")
    async def tavily_tool(query: str) -> str:
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

    # If user preference is 'cases' or heuristic strongly indicates a case query,
    # run the deterministic cases pipeline directly to guarantee full judgment usage.
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

    if (prefer and prefer.strip().lower() == 'cases') or prefer_cases:
        return await _run_cases_pipeline(question)

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
            "You are a legal agent with tools.\n"
            "- If the user asks about a specific CASE (e.g., contains 'v.'/'vs', asks to describe a case, mentions 'facts', 'holding', 'ratio', or gives a citation), you MUST call the 'find_cases_and_answer' tool first.\n"
            "- If the user asks about CONSTITUTIONAL provisions generally, use 'constitution_answer'.\n"
            "- If the user asks for latest/current events or recent updates, use 'tavily_web_search'.\n"
        )
        messages: List[Any] = [SystemMessage(content=sys_instructions), HumanMessage(content=question)]
        # Prefer hints
        if prefer:
            messages.insert(1, HumanMessage(content=f"HINT: Prefer tool {prefer.strip().lower()}"))
        elif prefer_cases:
            messages.insert(1, HumanMessage(content="HINT: Prefer tool cases"))
        result = await app.ainvoke({"messages": messages}) if hasattr(app, 'ainvoke') else app.invoke({"messages": messages})
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        final_text = None
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                final_text = m.content
                break
        # If the model returned empty or trivial, fall back to cases path when heuristic says so
        if (not final_text or len(final_text.strip()) < 10) and prefer_cases:
            # Minimal direct case flow
            docs = await search_indian_kanoon_async(question, limit=2)
            if docs:
                det = await get_case_details_async(docs[0].id)
                if det and det.get("success"):
                    title = det.get("title") or det.get("doc_id")
                    url = det.get("url", "")
                    court = det.get("court") or ""
                    date = det.get("date") or ""
                    citation = det.get("citation") or ""
                    excerpt = (det.get("full_text") or "")[:1500]
                    final_text = f"{title} — {court} ({date}) {citation}\nSource: {url}\n\n{excerpt}"
        return {"response": final_text or "", "route": "agentic", "graph": True}
    except Exception as e:
        logger.error("[AGENTIC] graph error: %s", e, exc_info=True)
        return {"response": "Failed to run agent graph", "route": "", "graph": False}
