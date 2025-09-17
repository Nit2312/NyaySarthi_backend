import os
from fastapi import APIRouter
from app.services import rag as rag_service
from app.services.rag import GROQ_API_KEY, DOCUMENT_CHAIN, INDEX_PATH, EMBEDDINGS
from langchain_community.vectorstores import FAISS

router = APIRouter()

@router.get("/health")
async def health():
    # Attempt to self-heal FAISS loading if index exists but not loaded (common during reloads)
    if rag_service.VECTOR_STORE is None and os.path.isdir(INDEX_PATH):
        idx_file = os.path.join(INDEX_PATH, "index.faiss")
        pkl_file = os.path.join(INDEX_PATH, "index.pkl")
        if os.path.exists(idx_file) and os.path.exists(pkl_file) and EMBEDDINGS is not None:
            try:
                rag_service.VECTOR_STORE = FAISS.load_local(
                    INDEX_PATH, EMBEDDINGS, allow_dangerous_deserialization=True
                )
                # (Re)initialize retriever if needed
                if getattr(rag_service, "RETRIEVER", None) is None and rag_service.VECTOR_STORE is not None:
                    import os as _os
                    k = int(_os.getenv("RETRIEVER_K", "4"))
                    fetch_k = int(_os.getenv("RETRIEVER_FETCH_K", "10"))
                    rag_service.RETRIEVER = rag_service.VECTOR_STORE.as_retriever(
                        search_type=_os.getenv("RETRIEVER_SEARCH_TYPE", "similarity"),
                        search_kwargs={"k": k, "fetch_k": fetch_k},
                    )
            except Exception:
                pass

    vs = rag_service.VECTOR_STORE
    vectors = getattr(getattr(vs, 'index', None), 'ntotal', 0) if vs else 0
    # If loaded but zero vectors, try manual reconstruction as a fallback
    if vs is not None and vectors == 0 and os.path.isdir(INDEX_PATH):
        idx_file = os.path.join(INDEX_PATH, "index.faiss")
        pkl_file = os.path.join(INDEX_PATH, "index.pkl")
        if os.path.exists(idx_file) and os.path.exists(pkl_file) and EMBEDDINGS is not None:
            try:
                import faiss as _faiss, pickle as _pickle
                index = _faiss.read_index(idx_file)
                with open(pkl_file, "rb") as f:
                    docstore, index_to_docstore_id = _pickle.load(f)
                rag_service.VECTOR_STORE = FAISS(
                    embedding_function=EMBEDDINGS,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id,
                )
                # Recreate retriever
                import os as _os
                k = int(_os.getenv("RETRIEVER_K", "4"))
                fetch_k = int(_os.getenv("RETRIEVER_FETCH_K", "10"))
                rag_service.RETRIEVER = rag_service.VECTOR_STORE.as_retriever(
                    search_type=_os.getenv("RETRIEVER_SEARCH_TYPE", "similarity"),
                    search_kwargs={"k": k, "fetch_k": fetch_k},
                )
                vs = rag_service.VECTOR_STORE
                vectors = getattr(getattr(vs, 'index', None), 'ntotal', 0) if vs else 0
            except Exception:
                pass
    return {
        "status": "ok",
        "faiss": {"loaded": rag_service.VECTOR_STORE is not None, "vectors": vectors, "path": INDEX_PATH},
        "llm": {"api_key_present": bool(GROQ_API_KEY), "initialized": DOCUMENT_CHAIN is not None},
    }

@router.get("/metrics")
async def metrics():
    return {"uptime": "unknown", "status": "ok"}
