import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Deque
from collections import defaultdict, deque

from fastapi.responses import JSONResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import faiss
import pickle

logger = logging.getLogger(__name__)

# Globals
EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
VECTOR_STORE: Optional[FAISS] = None
RETRIEVER = None
DOCUMENT_CHAIN = None  # Lazy init for speed
CHAT_HISTORIES: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=20))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

INDEX_PATH = str((Path(__file__).resolve().parent.parent.parent / "faiss_index").absolute())

PROMPT = ChatPromptTemplate.from_template(
    (
        """
        You are a legal expert. Use ONLY the provided context from the Indian Constitution and Indian case law excerpts to answer.
        Requirements:
        - Be concise and precise. If something is not in the context, say so.
        - When you rely on a case, cite the case title and include its URL in parentheses.
        - Where helpful, include 1-2 short quotations (within double quotes) from the context.
        - Structure the answer with short paragraphs or bullet points when appropriate.

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
        # Retrieve docs (handle different retriever interfaces)
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

        ranked_docs = await _rerank_documents(input_text, docs, top_k=5)

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

        return {
            "response": answer,
            "conversation_id": conv_id,
            "source": "constitution",
            "sources": [],
        }
    except Exception as e:
        logger.error(f"[RAG] chat_handler error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e) or "Internal Server Error"})
