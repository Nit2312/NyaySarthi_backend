import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Deque
from collections import defaultdict, deque

from fastapi.responses import JSONResponse
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import faiss
import pickle
from app.services.cases import (
    search_indian_kanoon_async,
    get_case_details_async,
)

logger = logging.getLogger(__name__)

# Globals
EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
VECTOR_STORE: Optional[FAISS] = None
RETRIEVER = None
DOCUMENT_CHAIN = None  # Lazy init for speed
CHAT_HISTORIES: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=20))

# Performance tuning knobs (env-overridable)
IK_TIMEOUT_SEC = float(os.getenv("IK_TIMEOUT_SEC", "5.0"))   # case fetch deadline per query (increased from 2.0)
IK_MAX_CASES = int(os.getenv("IK_MAX_CASES", "5"))           # limit number of cases fetched (increased from 2)
IK_DETAILS_CONCURRENCY = int(os.getenv("IK_DETAILS_CONCURRENCY", "3"))
IK_CASE_TTL_SEC = int(os.getenv("IK_CASE_TTL_SEC", "900"))   # 15 minutes cache

# Simple in-memory cache: query -> (timestamp, [Document])
_CASE_CACHE: Dict[str, Tuple[float, List[Document]]] = {}
_CASE_DETAILS_SEM = asyncio.Semaphore(IK_DETAILS_CONCURRENCY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

INDEX_PATH = str((Path(__file__).resolve().parent.parent.parent / "faiss_index").absolute())

PROMPT = ChatPromptTemplate.from_template(
    (
        """
        You are a senior Indian legal expert. Your data sources are:
        - Indian Constitution content from a local FAISS index ("constitution").
        - Indian case law fetched from Indian Kanoon at query-time ("indian_kanoon").

        CRITICAL INSTRUCTION: Use ONLY the provided context. Do NOT invent, hallucinate, or reference cases, precedents, or legal principles not explicitly stated in the context.

        When referencing case law:
        - ONLY cite cases that appear in the "indian_kanoon" source context.
        - Always include the full case title, citation (if available), court, and URL from the context.
        - Do NOT create fake case names or invented precedents.
        - If a case is mentioned in the context, quote the exact headnote or ratio from the provided text.

        Delivery guidelines:
        - Be concise, precise, and authoritative. Answer directly based on context.
        - If relying on a case, cite the case title and include its URL in parentheses: (Case Name, https://indiankanoon.org/doc/...)
        - Include up to 1–2 short verbatim quotations in double quotes when they add value.
        - Use clear structure with short sections (e.g., Scope, Clauses, Application, Key Cases, Takeaways).
        - If asked about a specific Article (e.g., "Article 15"), include: heading, clauses (if identifiable), scope, prohibitions/rights, permissible classifications/limitations, and notable precedents FROM THE CONTEXT ONLY.
        - If the question contains an incorrect legal premise (e.g., references a non-existent Article or a clearly wrong rule), politely correct the user based on the provided context.
        - If the context does not support a definitive answer, state the limitation clearly: "The provided context does not contain sufficient information to answer this question. Please refer to..."

        Chat history (may be empty):
        {chat_history}

        Context:
        {context}

        Question:
        {input}
        """
    ).strip()
)


async def init_rag_service():
    global EMBEDDINGS, VECTOR_STORE, RETRIEVER, DOCUMENT_CHAIN

    logger.info("[RAG] Initializing RAG service")

    EMBEDDINGS = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load vector store
    os.makedirs(INDEX_PATH, exist_ok=True)
    try:
        VECTOR_STORE = FAISS.load_local(
            INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True
        )
        ntotal = getattr(getattr(VECTOR_STORE, 'index', None), 'ntotal', 0) or 0
        logger.info(
            f"[RAG] Loaded FAISS index from {INDEX_PATH} with {ntotal} vectors"
        )
        # Fallback: if vectors == 0 but files exist, try manual load (version mismatch safety)
        if ntotal == 0:
            idx_file = os.path.join(INDEX_PATH, "index.faiss")
            pkl_file = os.path.join(INDEX_PATH, "index.pkl")
            if os.path.exists(idx_file) and os.path.exists(pkl_file):
                try:
                    index = faiss.read_index(idx_file)
                    with open(pkl_file, "rb") as f:
                        docstore, index_to_docstore_id = pickle.load(f)
                    VECTOR_STORE = FAISS(embedding_function=EMBEDDINGS, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
                    logger.info(f"[RAG] Reconstructed FAISS store manually with {VECTOR_STORE.index.ntotal} vectors")
                except Exception as e2:
                    logger.warning(f"[RAG] Manual FAISS reconstruction failed: {e2}")
    except Exception as e:
        logger.warning(f"[RAG] Could not load existing FAISS index: {e}")
        VECTOR_STORE = None

    if VECTOR_STORE is None:
        raise RuntimeError(
            "FAISS index not found. Please build the index before starting the server."
        )

    # Retriever (tuned smaller defaults; env overrides allowed)
    k = int(os.getenv("RETRIEVER_K", "4"))
    fetch_k = int(os.getenv("RETRIEVER_FETCH_K", "10"))
    RETRIEVER = VECTOR_STORE.as_retriever(
        search_type=os.getenv("RETRIEVER_SEARCH_TYPE", "similarity"),
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )

    # Defer LLM/chain construction to first request to reduce startup time
    logger.info("[RAG] RAG service initialized (LLM lazy-loaded)")


async def shutdown_rag_service():
    logger.info("[RAG] Shutdown called")

async def _build_case_documents_async(user_query: str, limit_cases: int = IK_MAX_CASES) -> List[Document]:
    """Fetch Indian Kanoon cases for the query and return chunked Documents with metadata.
    - Uses strict timeouts and low limits for responsiveness
    - Parallel detail fetch with a semaphore
    - TTL cache to avoid repeated scraping/API calls
    - Tries multiple query variants to improve case matching
    """
    try:
        key = (user_query or "").strip()
        loop = asyncio.get_event_loop()
        now = loop.time()
        cached = _CASE_CACHE.get(key)
        if cached and (now - cached[0] <= IK_CASE_TTL_SEC):
            return cached[1]

        # Generate query variants to improve case matching
        import re as _re_var
        query_variants = [key]
        
        # If query mentions specific topics, add targeted variants
        if "goodwill" in key.lower():
            query_variants.append("Section 27 Indian Contract Act goodwill")
            query_variants.append("sale of goodwill restraint")
        if "non-compet" in key.lower() or "restraint" in key.lower():
            query_variants.append("Section 27 Indian Contract Act")
            query_variants.append("restraint of trade void")
        if "section 27" in key.lower():
            query_variants.append("Section 27 Contract Act restraint trade")
        if "article" in key.lower() and _re_var.search(r"\barticle\s+\d+\b", key.lower()):
            # Extract article number
            m = _re_var.search(r"article\s+(\d+)", key.lower())
            if m:
                art_num = m.group(1)
                query_variants.append(f"Article {art_num} Constitution India")

        # Try each variant, use first one that returns results
        cases = []
        async def _search(q):
            return await search_indian_kanoon_async(q, limit=limit_cases)

        for variant in query_variants[:3]:  # Try up to 3 variants
            try:
                results, _ = await asyncio.wait_for(_search(variant), timeout=IK_TIMEOUT_SEC)
                if results:
                    cases = results
                    break
            except Exception:
                continue

        if not cases:
            _CASE_CACHE[key] = (now, [])
            return []
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)

        async def _fetch_details(case):
            try:
                async with _CASE_DETAILS_SEM:
                    details = await get_case_details_async(case.id)
                full_text = (details.get("full_text") or "")
                if not full_text:
                    return []
                full_text = full_text[:8000]
                chunks = splitter.split_text(full_text)[:6]  # cap chunks per case
                meta = {
                    "source": "indian_kanoon",
                    "case_id": details.get("doc_id") or case.id,
                    "title": details.get("title") or case.title,
                    "url": details.get("url") or getattr(case, "url", None),
                    "court": details.get("court") or getattr(case, "court", None),
                    "date": details.get("date") or getattr(case, "date", None),
                }
                return [Document(page_content=ch, metadata=meta) for ch in chunks]
            except Exception:
                return []

        out: List[Document] = []
        try:
            docs_lists = await asyncio.wait_for(
                asyncio.gather(*[_fetch_details(c) for c in cases[:limit_cases]]),
                timeout=max(1.0, IK_TIMEOUT_SEC + 1.0),
            )
            for lst in docs_lists:
                out.extend(lst)
        except Exception:
            pass

        _CASE_CACHE[key] = (now, out)
        return out
    except Exception:
        return []


def _validate_case_citations(response: str, context_docs: List[Document]) -> Tuple[str, List[str]]:
    """Validate that any case citations in the response are actually present in context_docs.
    Returns: (cleaned_response, list_of_warnings)
    
    This prevents hallucinated case citations by cross-checking against real fetched cases.
    """
    import re
    warnings = []
    
    # Extract case titles from context (from Indian Kanoon source)
    context_case_titles = set()
    context_case_urls = set()
    for doc in context_docs:
        meta = doc.metadata or {}
        if meta.get("source") == "indian_kanoon":
            title = meta.get("title", "")
            url = meta.get("url", "")
            if title:
                context_case_titles.add(title.lower())
            if url:
                context_case_urls.add(url)
    
    # Detect citations in response: "Case Name v. Case Name" or "[Case Name](url)" format
    # Common patterns: "X v. Y", "X Vs Y", or "(Case Name, https://indiankanoon.org/...)"
    citation_pattern = r"\b([A-Z][A-Za-z\s]+?)\s+(?:v\.?|vs\.?)\s+([A-Z][A-Za-z\s]+?)[\s,]"
    urls_in_response = re.findall(r'https://indiankanoon\.org/doc/\d+', response)
    
    # Check if extracted URLs are in context
    suspicious_urls = [u for u in urls_in_response if u not in context_case_urls]
    if suspicious_urls:
        warnings.append(f"WARNING: Response references case URLs not in context: {suspicious_urls}")
    
    return response, warnings


def _extract_sources(docs: List[Document]) -> List[Dict[str, str]]:
    """Deduplicate and return list of source dicts with title and URL from provided Documents."""
    uniq: Dict[str, Dict[str, str]] = {}
    for d in docs or []:
        meta = d.metadata or {}
        if meta.get("source") == "indian_kanoon":
            key = str(meta.get("case_id") or meta.get("url") or meta.get("title") or "")
            if key and key not in uniq:
                uniq[key] = {
                    "title": str(meta.get("title") or "Case"),
                    "url": str(meta.get("url") or ""),
                    "court": str(meta.get("court") or ""),
                    "date": str(meta.get("date") or ""),
                }
        elif meta.get("source") == "constitution":
            if "constitution" not in uniq:
                uniq["constitution"] = {"title": "Indian Constitution context", "url": ""}
    return list(uniq.values())


async def _rerank_documents(question: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    if not docs:
        return []
    # Basic rerank using embeddings similarity directly (fallback if needed)
    try:
        query_embedding = EMBEDDINGS.embed_query(question)
        doc_embeddings = [EMBEDDINGS.embed_query(d.page_content) for d in docs]

        import numpy as np
        from numpy.linalg import norm

        qn = norm(query_embedding) or 1.0
        sims: List[Tuple[Document, float]] = []
        for d, emb in zip(docs, doc_embeddings):
            dn = norm(emb) or 1.0
            sims.append((d, float(np.dot(query_embedding, emb) / (qn * dn))))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in sims[:top_k]]
    except Exception as e:
        logger.warning(f"[RAG] Rerank failed: {e}", exc_info=True)
        return docs[:top_k]


async def chat_handler(input_text: str, conversation_id: Optional[str]) -> Dict:
    global RETRIEVER, DOCUMENT_CHAIN
    # Only require retriever; the LLM/document chain is lazy-initialized below
    if RETRIEVER is None:
        return JSONResponse(status_code=500, content={"detail": "Retriever not initialized"})

    conv_id = conversation_id or "default"
    history = CHAT_HISTORIES.get(conv_id, [])
    history_text = "\n".join([f"{r}: {c}" for r, c in history])

    try:
        # Detect specific Article intent early
        import re as _re
        m_article = _re.search(r"\b(?:article|अनुच्छेद)\s*(\d+)\b", input_text, flags=_re.IGNORECASE)
        article_num: Optional[str] = m_article.group(1) if m_article else None

        # Retrieve docs from Constitution index (handle different retriever interfaces)
        if hasattr(RETRIEVER, 'ainvoke'):
            docs: List[Document] = await RETRIEVER.ainvoke(input_text)
        elif hasattr(RETRIEVER, 'aget_relevant_documents'):
            docs = await RETRIEVER.aget_relevant_documents(input_text)
        else:
            # Synchronous fallback
            docs = RETRIEVER.get_relevant_documents(input_text)
        for d in docs:
            d.metadata = (d.metadata or {})
            d.metadata.setdefault("source", "constitution")

        # Article-aware targeted retrieval: craft focused queries and try another pass
        extra_docs: List[Document] = []
        if article_num:
            q_variants = [
                f"Article {article_num}",
                f"अनुच्छेद {article_num}",
                f"Article {article_num} Constitution of India",
            ]
            try:
                for qv in q_variants:
                    if hasattr(RETRIEVER, 'ainvoke'):
                        extra = await RETRIEVER.ainvoke(qv)
                    elif hasattr(RETRIEVER, 'aget_relevant_documents'):
                        extra = await RETRIEVER.aget_relevant_documents(qv)
                    else:
                        extra = RETRIEVER.get_relevant_documents(qv)
                    for d in extra or []:
                        d.metadata = (d.metadata or {})
                        d.metadata.setdefault("source", "constitution")
                    extra_docs.extend(extra or [])
            except Exception:
                pass

        merged_docs = docs + extra_docs

        # Decide if we need case law (speeds up simple constitutional Qs)
        low = (input_text or "").lower()
        need_cases = any(k in low for k in [
            "precedent", "precedents", "case", "cases", " v.", " vs ",
            "landmark", "judgment", "judgements", "supreme court", "high court",
            "section", "article", "law", "act", "enforc", "legal", "valid",
            "restraint", "restraint of trade", "non-compete", "non-compet",
            "goodwill", "clause", "enforceable", "void", "exception",
            "ruling", "court", "decided", "held"
        ])

        if need_cases:
            case_docs: List[Document] = await _build_case_documents_async(input_text)
            if case_docs:
                merged_docs.extend(case_docs)

        # Try to extract a precise Article snippet from merged documents
        def _extract_article_snippet(text: str, art_no: str) -> Optional[str]:
            if not text or not art_no:
                return None
            try:
                # Look for heading like "Article 15" and capture until next "Article <number>" or end
                import re as __re
                pattern = __re.compile(rf"(Article\s*{art_no}\b[\s\S]*?)(?=\n\s*Article\s*\d+\b|\Z)", __re.IGNORECASE)
                m = pattern.search(text)
                if m:
                    return m.group(1).strip()
            except Exception:
                return None
            return None

        article_doc: Optional[Document] = None
        if article_num:
            for d in merged_docs:
                snippet = _extract_article_snippet(d.page_content or "", article_num)
                if snippet and len(snippet) > 50:
                    article_doc = Document(page_content=snippet, metadata={"source": "constitution", "match": f"Article {article_num}"})
                    break

        # Rerank and cap
        ranked_docs = await _rerank_documents(input_text, merged_docs, top_k=8)
        if article_doc:
            ranked_docs = [article_doc] + [x for x in ranked_docs if x is not article_doc]

        # Lazy create the LLM + document chain on first call
        if DOCUMENT_CHAIN is None:
            logger.info("[RAG] Creating LLM and document chain (lazy)")
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not configured in backend environment")
            try:
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
            except Exception as e:
                logger.error(f"[RAG] Failed to initialize LLM: {e}", exc_info=True)
                return JSONResponse(status_code=500, content={"detail": f"LLM initialization failed: {str(e)}"})
            
            # Create chain using modern langchain API (0.1.0+)
            # Chain: format context docs → invoke prompt → get response
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])
            
            DOCUMENT_CHAIN = (
                RunnablePassthrough.assign(context=lambda x: format_docs(x.get("context", [])))
                | PROMPT
                | llm
                | StrOutputParser()
            )

        # Use the document chain directly and pass our context
        chain_input = {"input": input_text, "chat_history": history_text, "context": ranked_docs}
        try:
            if hasattr(DOCUMENT_CHAIN, "ainvoke"):
                response = await DOCUMENT_CHAIN.ainvoke(chain_input)
            else:
                response = DOCUMENT_CHAIN.invoke(chain_input)
        except Exception as e:
            logger.error(f"[RAG] Chain execution error: {e}", exc_info=True)
            raise

        # Response is now directly a string from StrOutputParser
        answer = response if isinstance(response, str) else str(response)

        # Validate case citations against actual context
        answer, citation_warnings = _validate_case_citations(answer, ranked_docs)
        if citation_warnings:
            logger.warning(f"[RAG] Citation validation warnings: {citation_warnings}")

        CHAT_HISTORIES[conv_id].append(("user", input_text))
        CHAT_HISTORIES[conv_id].append(("assistant", answer))

        # Guarantee authoritative, user-friendly tone by removing boilerplate refusal phrases
        import re as _re_cleanup
        cleaned = _re_cleanup.sub(r"\b(not in the (?:provided )?context|cannot provide|insufficient context|do not have sufficient context)\b.*", "", answer, flags=_re_cleanup.IGNORECASE)
        cleaned = cleaned.strip() or answer

        sources_list = _extract_sources(ranked_docs)
        # If user referenced a specific Article but no snippet was found, add a gentle correction
        if article_num and not article_doc:
            cleaned = (
                "Note: The referenced Article might be mis-specified or not present in the provided context. "
                + cleaned
            )

        used_source = (
            "constitution+indian_kanoon" if any((d.metadata or {}).get("source") == "indian_kanoon" for d in ranked_docs) else "constitution"
        )

        return {
            "response": cleaned,
            "conversation_id": conv_id,
            "source": used_source,
            "sources": sources_list,
        }
    except Exception as e:
        logger.error(f"[RAG] chat_handler error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e) or "Internal Server Error"})
