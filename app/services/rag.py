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
from langchain.chains.combine_documents import create_stuff_documents_chain
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
IK_TIMEOUT_SEC = float(os.getenv("IK_TIMEOUT_SEC", "2.0"))  # case fetch deadline per query
IK_MAX_CASES = int(os.getenv("IK_MAX_CASES", "2"))          # limit number of cases fetched
IK_DETAILS_CONCURRENCY = int(os.getenv("IK_DETAILS_CONCURRENCY", "3"))
IK_CASE_TTL_SEC = int(os.getenv("IK_CASE_TTL_SEC", "900"))  # 15 minutes cache

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

        Use ONLY the provided context. Deliver a complete, user-facing answer with the following:
        - Be concise, precise, and authoritative. Answer directly.
        - If relying on a case, cite the case title and include its URL in parentheses.
        - Include up to 1–2 short quotations in double quotes when they add value.
        - Use clear structure with short sections (e.g., Scope, Clauses, Application, Key Cases, Takeaways).
        - If asked about a specific Article (e.g., "Article 15"), include: heading, clauses (if identifiable), scope, prohibitions/rights, permissible classifications/limitations, and notable precedents.
        - If the question contains an incorrect legal premise (e.g., references a non-existent Article or a clearly wrong rule), politely correct the user and provide the correct framing.
        - If the context does not support a definitive answer, state the limitation and provide the best-supported view from the context.

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
    """
    try:
        key = (user_query or "").strip()
        loop = asyncio.get_event_loop()
        now = loop.time()
        cached = _CASE_CACHE.get(key)
        if cached and (now - cached[0] <= IK_CASE_TTL_SEC):
            return cached[1]

        async def _search():
            return await search_indian_kanoon_async(key, limit=limit_cases)

        try:
            cases, _ = await asyncio.wait_for(_search(), timeout=IK_TIMEOUT_SEC)
        except Exception:
            cases = []
        if not cases:
            _CASE_CACHE[key] = (now, [])
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
            "landmark", "judgment", "judgements", "supreme court", "high court"
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
            DOCUMENT_CHAIN = create_stuff_documents_chain(llm, PROMPT)

        # Use the document chain directly and pass our context
        chain_input = {"input": input_text, "chat_history": history_text, "context": ranked_docs}
        if hasattr(DOCUMENT_CHAIN, "ainvoke"):
            response = await DOCUMENT_CHAIN.ainvoke(chain_input)
        else:
            response = DOCUMENT_CHAIN.invoke(chain_input)

        if isinstance(response, dict):
            answer = response.get("answer", "") or response.get("text", "")
        elif hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)

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
