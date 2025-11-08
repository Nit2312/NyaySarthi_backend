import os
import time
import json
import hashlib
import logging
import re
import html
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Deque
from collections import defaultdict, deque

# FastAPI imports
from fastapi import FastAPI, Form, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request as FastAPIRequest
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from uuid import uuid4

# Lazy imports for heavy libraries
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    from bs4 import BeautifulSoup
    from pydantic import BaseModel
    from dotenv import load_dotenv
    import requests
    import httpx
    import asyncio
    import faiss
    import numpy as np
except ImportError as e:
    logging.warning(f"Optional dependency not found: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    def __init__(self, app, max_requests=100, window=60):
        self.app = app
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        client_ip = scope.get("client")[0] if scope.get("client") else "unknown"
        now = time.time()
        
        # Clean up old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            response = JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."},
                status_code=429
            )
            await response(scope, receive, send)
            return
            
        self.requests[client_ip].append(now)
        await self.app(scope, receive, send)

app = FastAPI()

# Rate limiting
app.add_middleware(RateLimitMiddleware, max_requests=100, window=60)

# CORS configuration
origins_env = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").strip()
if origins_env == "*":
    cors_allow_origins = ["*"]
    cors_allow_credentials = False
else:
    cors_allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    cors_allow_credentials = True

# Ensure Vercel frontend origin is included
vercel_origin = "https://nyay-sarthi-frontend.vercel.app"
if cors_allow_origins != ["*"] and vercel_origin not in cors_allow_origins:
    cors_allow_origins.append(vercel_origin)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "*",
        "Content-Type",
        "Authorization",
        "Cache-Control",
        "Pragma",
        "X-Requested-With",
        "x-request-id",  # <-- add this line
    ],
    expose_headers=["*"],
    max_age=600,
)

# Add middleware to handle OPTIONS requests and CORS headers
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    # Get the origin from the request
    origin = request.headers.get("origin")
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://nyay-sarthi-frontend.vercel.app",
    ]
    
    # Default headers
    cors_headers = {
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE, PUT",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, X-Request-ID, Accept",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Max-Age": "600",  # 10 minutes
    }
    
    # If origin is in allowed origins, use it, otherwise use the first allowed origin
    if origin in allowed_origins:
        cors_headers["Access-Control-Allow-Origin"] = origin
    elif allowed_origins:
        cors_headers["Access-Control-Allow-Origin"] = allowed_origins[0]
    
    # Handle preflight requests
    if request.method == "OPTIONS":
        return JSONResponse(
            content={"status": "ok"},
            headers=cors_headers
        )
    
    # Process the request
    response = await call_next(request)
    
    # Add CORS headers to the response
    for key, value in cors_headers.items():
        response.headers[key] = value
        
    return response

# Simple in-memory cache for API responses
# Format: {cache_key: {"data": response_data, "timestamp": datetime, "ttl": seconds}}
API_CACHE: Dict[str, Dict] = {}
CACHE_TTL = 3600  # 1 hour
CASE_DETAILS_CACHE_TTL = 7200  # 2 hours

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # LLM model
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token or ""

# Indian Kanoon API configuration
IK_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")
IK_EMAIL = os.getenv("INDIAN_KANOON_EMAIL", "")
SCRAPE_FALLBACK = os.getenv("SCRAPE_INDIAN_KANOON", "false").lower() in {"1", "true", "yes"}

# Initialize embeddings with better configuration
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Check if we have a saved index
import os
from pathlib import Path

# Define the index path as an absolute path to avoid any issues
base_dir = Path(__file__).parent.absolute()
index_path = str(base_dir / "faiss_index")
vector_store = None

# Ensure the directory exists
os.makedirs(index_path, exist_ok=True)

# Get the expected embedding dimension from the embeddings model
embedding_dim = len(embeddings.embed_query("test"))
logger.info(f"Expected embedding dimension: {embedding_dim}")
logger.info(f"Using index path: {index_path}")

if os.path.exists(index_path) and os.path.isdir(index_path):
    try:
        logger.info("Loading existing FAISS index...")
        # Load the index without the embeddings first
        faiss_index = faiss.read_index(os.path.join(index_path, "index.faiss"))
        
        # Check if dimensions match
        if faiss_index.d != embedding_dim:
            logger.warning(f"Dimension mismatch: Index has {faiss_index.d} dimensions, expected {embedding_dim}")
            raise ValueError("Embedding dimension mismatch")
            
        # Now load with embeddings
        vector_store = FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"Successfully loaded FAISS index with {vector_store.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        logger.info("Creating new FAISS index...")
        vector_store = None

# If no valid index was loaded, create a new one
if vector_store is None:
    try:
        logger.info("Loading and processing constitution PDF...")
        pdf_path = os.path.join("data", "constitution.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Constitution PDF not found at {os.path.abspath(pdf_path)}")
            
        logger.info(f"Extracting text from all pages of the constitution...")
        
        # Use PyMuPDF directly for better control over text extraction
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        docs = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text with better options to avoid garbage characters
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_IMAGES | fitz.TEXT_DEHYPHENATE)
            
            # Clean up common issues in extracted text
            text = text.strip()
            
            # Skip empty pages
            if not text:
                continue
                
            # Create a document for this page
            from langchain.docstore.document import Document
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num + 1,
                    "total_pages": len(doc)
                }
            ))
        
        if not docs:
            raise ValueError("No valid text was extracted from the constitution PDF")
            
        logger.info(f"Successfully extracted text from {len(docs)} pages of the constitution")
        
        # Configure text splitter for legal documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Good balance for legal text
            chunk_overlap=200,  # Maintains context between chunks
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better handling of legal document structure
        )
        
        # Process all pages of the constitution
        logger.info("Splitting document into chunks...")
        final_documents = text_splitter.split_documents(docs)
        
        if not final_documents:
            raise ValueError("No documents were created after splitting the PDF")
            
        logger.info(f"Split into {len(final_documents)} chunks with average length of "
                  f"{sum(len(d.page_content) for d in final_documents)/len(final_documents):.0f} chars")
        
        # Create the vector store with the approach that worked in the test
        logger.info("Creating FAISS index...")
        logger.info(f"Number of documents to index: {len(final_documents)}")
        logger.info(f"First document sample: {final_documents[0].page_content[:200]}...")
        
        # Ensure the directory exists
        os.makedirs(index_path, exist_ok=True)
        logger.info(f"Created directory: {os.path.abspath(index_path)}")
        
        try:
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.document import Document
            
            # Process documents in smaller batches to avoid memory issues
            batch_size = 100
            vector_store = None
            
            for i in range(0, len(final_documents), batch_size):
                batch = final_documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(final_documents)-1)//batch_size + 1} with {len(batch)} documents")
                
                if vector_store is None:
                    # First batch - create new index
                    vector_store = FAISS.from_documents(
                        documents=batch,
                        embedding=embeddings
                    )
                else:
                    # Subsequent batches - add to existing index
                    vector_store.add_documents(batch)
            
            if vector_store is None:
                raise ValueError("No documents were added to the vector store")
                
            logger.info("FAISS index created successfully")
            logger.info(f"Index dimension: {vector_store.index.d}, Number of vectors: {vector_store.index.ntotal}")
            
            # Save the index
            logger.info(f"Saving FAISS index to {os.path.abspath(index_path)}")
            vector_store.save_local(index_path)
            
            # Verify the files were created
            required_files = ["index.faiss", "index.pkl"]
            for file in required_files:
                file_path = os.path.join(index_path, file)
                if not os.path.exists(file_path):
                    raise RuntimeError(f"Failed to create {file_path}")
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # in MB
                logger.info(f"Created {file} - Size: {file_size:.2f} MB")
            
            logger.info(f"Successfully created and saved FAISS index with {len(final_documents)} vectors")
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise RuntimeError("Failed to initialize document vector store") from e

# Initialize the retriever with a simple similarity search first
retriever = vector_store.as_retriever(
    search_type="similarity",  # Use basic similarity search
    search_kwargs={
        'k': 5,  # Number of documents to retrieve
        'fetch_k': 20  # Number of documents to consider during search
    }
)

logger.info(f"Initialized retriever with {vector_store.index.ntotal} vectors")

# Store global references for debugging
FAISS_INDEX = vector_store
EMBEDDINGS = embeddings

llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model)

# Simple in-memory chat history store
# Key: conversation_id, Value: deque of (role, content)
MAX_HISTORY_TURNS = 10  # number of user-assistant turns to keep
chat_histories: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))

# Prompt now includes chat history for better contextual answers
prompt = ChatPromptTemplate.from_template(
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
        """.strip()
    )
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# ===== Cache utility functions =====
def generate_cache_key(prefix: str, *args) -> str:
    """Generate a cache key from prefix and arguments."""
    data = "|".join(str(arg) for arg in args)
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()[:16]}"

def get_from_cache(cache_key: str) -> Optional[Dict]:
    """Get data from cache if not expired."""
    if cache_key not in API_CACHE:
        return None
    
    cache_entry = API_CACHE[cache_key]
    if datetime.now() - cache_entry["timestamp"] > timedelta(seconds=cache_entry["ttl"]):
        del API_CACHE[cache_key]
        return None
    
    logger.info(f"Cache hit: {cache_key}")
    return cache_entry["data"]

def set_cache(cache_key: str, data: Dict, ttl: int = CACHE_TTL):
    """Set data in cache with TTL."""
    API_CACHE[cache_key] = {
        "data": data,
        "timestamp": datetime.now(),
        "ttl": ttl
    }
    logger.info(f"Cache set: {cache_key}")

def cleanup_cache():
    """Remove expired cache entries."""
    expired_keys = []
    for key, entry in API_CACHE.items():
        if datetime.now() - entry["timestamp"] > timedelta(seconds=entry["ttl"]):
            expired_keys.append(key)
    
    for key in expired_keys:
        del API_CACHE[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Global variables for FAISS vector store and embeddings
FAISS_INDEX = None
EMBEDDINGS = None

# Global async HTTP client with connection pooling
HTTP_CLIENT: Optional[httpx.AsyncClient] = None

# Performance metrics
REQUEST_METRICS = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "average_response_time": 0.0
}

# Initialize FAISS index and embeddings at startup
@app.on_event("startup")
async def startup_event():
    global FAISS_INDEX, EMBEDDINGS, HTTP_CLIENT
    try:
        # Initialize async HTTP client with connection pooling
        HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            headers={"User-Agent": "NyaySarthi/1.0 Legal Research Assistant"}
        )
        logger.info("Initialized HTTP client with connection pooling")
        
        # Initialize embeddings
        EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        
        # Load FAISS index if it exists
        faiss_index_path = "faiss_index"
        if os.path.exists(faiss_index_path):
            FAISS_INDEX = FAISS.load_local(
                faiss_index_path,
                EMBEDDINGS,
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded FAISS index")
        else:
            logger.warning(f"FAISS index not found at {faiss_index_path}")
            
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global HTTP_CLIENT
    if HTTP_CLIENT:
        await HTTP_CLIENT.aclose()
        logger.info("HTTP client closed")

# ===== Case search and summarization via Indian Kanoon =====
class CaseDoc(BaseModel):
    id: str
    title: str
    court: Optional[str] = None
    date: Optional[str] = None
    citation: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None

async def search_indian_kanoon_async(query: str, limit: int = 5) -> Tuple[List[CaseDoc], Optional[str]]:
    """Async version of search_indian_kanoon for better performance.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (list of CaseDoc objects, error_message if any)
    """
    global IK_EMAIL, IK_API_KEY, HTTP_CLIENT, REQUEST_METRICS
    
    # Track metrics
    start_time = time.time()
    REQUEST_METRICS["total_requests"] += 1
    
    # Generate a cache key based on the query and limit
    cache_key = generate_cache_key("ik_search", query, limit)
    cached_result = get_from_cache(cache_key)
    if cached_result is not None:
        REQUEST_METRICS["cache_hits"] += 1
        logger.debug(f"Returning cached result for query: {query[:50]}...")
        return cached_result["cases"], cached_result.get("error")
        
    REQUEST_METRICS["cache_misses"] += 1
    logger.info(f"Searching Indian Kanoon for: {query[:100]}...")
    
    # Enhanced query processing with validation
    query = query.strip()
    if not query:
        logger.warning("Empty query provided to search_indian_kanoon")
        return [], "empty_query"
    
    if len(query) > 1000:
        logger.warning(f"Query too long ({len(query)} characters), truncating")
        query = query[:1000]
    
    # Try scraping first since it doesn't require credentials
    if os.getenv('SCRAPE_INDIAN_KANOON', 'true').lower() == 'true':
        try:
            logger.info("Attempting to scrape Indian Kanoon...")
            scraped = await scrape_indian_kanoon_search_async(query, min(limit, 5))
            if scraped:
                logger.info(f"Successfully scraped {len(scraped)} cases from Indian Kanoon")
                result = {"cases": [case.dict() for case in scraped], "error": None}
                set_cache(cache_key, result, ttl=3600)  # Cache for 1 hour
                # Update metrics
                elapsed = time.time() - start_time
                REQUEST_METRICS["average_response_time"] = (
                    (REQUEST_METRICS["average_response_time"] * (REQUEST_METRICS["total_requests"] - 1) + elapsed) / 
                    REQUEST_METRICS["total_requests"]
                )
                return scraped, None
        except Exception as e:
            logger.warning(f"Scraping attempt failed: {str(e)}")
    else:
        logger.info("Scraping is disabled via SCRAPE_INDIAN_KANOON setting")
    
    # If scraping failed or disabled, try API if credentials are available
    if not IK_EMAIL or not IK_API_KEY:
        logger.warning("Indian Kanoon API credentials not configured")
        return [], "no_credentials"
        
    logger.info("Attempting to use Indian Kanoon API...")
    try:
        params = {
            "formInput": query,
            "pagenum": 0,
            "sort_by": "relevance",
            "type": "judgments",
            "from": "01-01-1950",
            "to": datetime.now().strftime("%d-%m-%Y"),
            "format": "json"
        }
        
        if not HTTP_CLIENT:
            logger.error("HTTP client not initialized")
            return [], "http_client_error"
            
        logger.info(f"Making async API request to Indian Kanoon")
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
                    
                title = d.get("title", "").strip()
                court = (d.get("court") or d.get("docsource") or "").strip() or None
                date = d.get("date") or None
                citation = d.get("citation") or None
                url = f"https://indiankanoon.org/doc/{doc_id}/"
                
                if title:  # Only add if we have at least a title
                    docs.append(CaseDoc(
                        id=doc_id,
                        title=title,
                        court=court,
                        date=date,
                        citation=citation,
                        url=url,
                    ))
            
            if docs:  # Only cache if we got results
                logger.info(f"Found {len(docs)} cases from API")
                result = {"cases": [doc.dict() for doc in docs], "error": None}
                set_cache(cache_key, result)
                # Update metrics
                elapsed = time.time() - start_time
                REQUEST_METRICS["average_response_time"] = (
                    (REQUEST_METRICS["average_response_time"] * (REQUEST_METRICS["total_requests"] - 1) + elapsed) / 
                    REQUEST_METRICS["total_requests"]
                )
                return docs, None
            else:
                logger.warning("No valid cases found in API response")
                return [], "no_results"
        elif resp.status_code == 401:
            error_msg = "Indian Kanoon API authentication failed. Please check your credentials."
            logger.error(error_msg)
            return [], "auth_failed"
        elif resp.status_code == 429:
            error_msg = "Rate limit exceeded for Indian Kanoon API. Please try again later."
            logger.error(error_msg)
            return [], "rate_limit"
        else:
            error_msg = f"API request failed with status {resp.status_code}: {resp.text}"
            logger.error(error_msg)
            return [], f"api_error_{resp.status_code}"
            
    except httpx.TimeoutException:
        error_msg = "Request to Indian Kanoon API timed out. Please try again later."
        logger.error(error_msg)
        return [], "timeout"
    except httpx.RequestError as e:
        error_msg = f"Request to Indian Kanoon API failed: {str(e)}"
        logger.error(error_msg)
        return [], "request_failed"
    except Exception as e:
        error_msg = f"Unexpected error while searching Indian Kanoon: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], "unexpected_error"
            
    # If we get here, both scraping and API failed
    logger.warning("All search methods failed")
    return [], "search_failed"


def search_indian_kanoon(query: str, limit: int = 5) -> Tuple[List[CaseDoc], Optional[str]]:
    """Search Indian Kanoon for relevant cases using the API or fallback to scraping.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (list of CaseDoc objects, error_message if any)
    """
    global IK_EMAIL, IK_API_KEY
    
    # Generate a cache key based on the query and limit
    cache_key = generate_cache_key("ik_search", query, limit)
    cached_result = get_from_cache(cache_key)
    if cached_result is not None:
        logger.debug(f"Returning cached result for query: {query[:50]}...")
        return cached_result["cases"], cached_result.get("error")
        
    logger.info(f"Searching Indian Kanoon for: {query[:100]}...")
    
    # Enhanced query processing
    query = query.strip()
    if not query:
        logger.warning("Empty query provided to search_indian_kanoon")
        return [], "empty_query"
    
    # Try scraping first since it doesn't require credentials
    if os.getenv('SCRAPE_INDIAN_KANOON', 'true').lower() == 'true':
        try:
            logger.info("Attempting to scrape Indian Kanoon...")
            scraped = scrape_indian_kanoon_search(query, min(limit, 5))  # Limit to 5 for scraping
            if scraped:
                logger.info(f"Successfully scraped {len(scraped)} cases from Indian Kanoon")
                result = {"cases": [case.dict() for case in scraped], "error": None}
                set_cache(cache_key, result, ttl=3600)  # Cache for 1 hour
                return scraped, None
        except Exception as e:
            logger.warning(f"Scraping attempt failed: {str(e)}")
    else:
        logger.info("Scraping is disabled via SCRAPE_INDIAN_KANOON setting")
    
    # If scraping failed or disabled, try API if credentials are available
    if not IK_EMAIL or not IK_API_KEY:
        logger.warning("Indian Kanoon API credentials not configured")
        return [], "no_credentials"
        
    logger.info("Attempting to use Indian Kanoon API...")
    try:
        params = {
            "formInput": query,
            "pagenum": 0,
            "sort_by": "relevance",
            "type": "judgments",
            "from": "01-01-1950",
            "to": datetime.now().strftime("%d-%m-%Y"),
            "format": "json"
        }
        
        headers = {
            "User-Agent": "NyaySarthi/1.0 (+https://github.com/nyaysarthi/nyay-sarthi)",
            "Accept": "application/json"
        }
        
        logger.info(f"Making API request to Indian Kanoon with params: {params}")
        resp = requests.get(
            "https://api.indiankanoon.org/search/",
            params=params,
            auth=(IK_EMAIL, IK_API_KEY),
            headers=headers,
            timeout=30,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            docs: List[CaseDoc] = []
            
            for d in data.get("docs", [])[:limit]:
                doc_id = str(d.get("id", "")).strip()
                if not doc_id:
                    continue
                    
                title = d.get("title", "").strip()
                court = (d.get("court") or d.get("docsource") or "").strip() or None
                date = d.get("date") or None
                citation = d.get("citation") or None
                url = f"https://indiankanoon.org/doc/{doc_id}/"
                
                if title:  # Only add if we have at least a title
                    docs.append(CaseDoc(
                        id=doc_id,
                        title=title,
                        court=court,
                        date=date,
                        citation=citation,
                        url=url,
                    ))
            
            if docs:  # Only cache if we got results
                logger.info(f"Found {len(docs)} cases from API")
                result = {"cases": [doc.dict() for doc in docs], "error": None}
                set_cache(cache_key, result)
                return docs, None
            else:
                logger.warning("No valid cases found in API response")
                return [], "no_results"
        elif resp.status_code == 401:
            error_msg = "Indian Kanoon API authentication failed. Please check your credentials."
            logger.error(error_msg)
            return [], "auth_failed"
        elif resp.status_code == 429:
            error_msg = "Rate limit exceeded for Indian Kanoon API. Please try again later."
            logger.error(error_msg)
            return [], "rate_limit"
        else:
            error_msg = f"API request failed with status {resp.status_code}: {resp.text}"
            logger.error(error_msg)
            return [], f"api_error_{resp.status_code}"
            
    except requests.exceptions.Timeout:
        error_msg = "Request to Indian Kanoon API timed out. Please try again later."
        logger.error(error_msg)
        return [], "timeout"
    except requests.exceptions.RequestException as e:
        error_msg = f"Request to Indian Kanoon API failed: {str(e)}"
        logger.error(error_msg)
        return [], "request_failed"
    except Exception as e:
        error_msg = f"Unexpected error while searching Indian Kanoon: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], "unexpected_error"
            
    # If we get here, both scraping and API failed
    logger.warning("All search methods failed")
    return [], "search_failed"


async def scrape_indian_kanoon_search_async(query: str, limit: int = 5) -> List[CaseDoc]:
    """Async version of scrape_indian_kanoon_search for better performance.
    Returns a list of CaseDoc with best-effort fields (title, url, snippet as summary).
    """
    global HTTP_CLIENT
    
    try:
        # Clean and encode the query
        query = query.strip()
        if not query:
            return []
            
        # Build search URL with a more specific query
        base_url = "https://indiankanoon.org/search/"
        params = {
            'formInput': f'"{query}"',  # Use exact phrase matching
            'pagenum': 0,
            'sortby': 'mostrecent',
            'type': 'judgments',
            'fromdate': '01-01-1950',  # Limit to post-independence cases
            'from': '01-01-1950',
            'to': datetime.now().strftime("%d-%m-%Y"),
        }
        
        if not HTTP_CLIENT:
            logger.error("HTTP client not initialized for scraping")
            return []
        
        # Make the request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await HTTP_CLIENT.get(
                    base_url, 
                    params=params,
                    follow_redirects=True
                )
                response.raise_for_status()
                break
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch search results after {max_retries} attempts: {str(e)}")
                    return []
                await asyncio.sleep(1)  # Wait before retrying
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find all result divs with more specific selectors
        result_divs = soup.select('div.result, div.search-result')
        
        for div in result_divs[:limit]:
            try:
                # Extract title and URL
                title_elem = div.select_one('a[href^="/doc/"]')
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                doc_id = title_elem['href'].split('/')[-2]
                if not doc_id.isdigit():
                    continue
                    
                url = f"https://indiankanoon.org/doc/{doc_id}/"
                
                # Extract court and date with better parsing
                doc_info = div.select_one('.docsource_main, .doc_subtitle, .result-subtitle')
                court = None
                date = None
                citation = None
                
                if doc_info:
                    info_text = doc_info.get_text(strip=True, separator='|')
                    parts = [p.strip() for p in info_text.split('|') if p.strip()]
                    
                    # Try to extract court and date from different formats
                    for part in parts:
                        # Check for date-like patterns
                        if re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', part) or 'on ' in part.lower():
                            date = part.replace('on ', '').strip()
                        # Check for citation patterns
                        elif re.search(r'\b[A-Za-z]+\.?\s*\d+\s*\w*\s*\d+', part):
                            citation = part.strip()
                        # Assume the first non-date, non-citation part is the court
                        elif not date and not citation and len(part) < 100:  # Arbitrary length check
                            court = part
                        break
                
                # Extract snippet/summary
                snippet = ""
                snippet_el = div.select_one("p, .doc, .doc_summary")
                if snippet_el:
                    snippet = " ".join(snippet_el.get_text(" ", strip=True).split())
                
                # If no snippet, use first few words of title and metadata
                if not snippet:
                    meta_text = " ".join(
                        p.get_text(" ", strip=True) 
                        for p in div.select("div.doc_box, .docsource, .doc_date, .doc_subtitle")
                        if p.get_text(strip=True)
                    )
                    snippet = f"{title}. {meta_text}"
                
                # Clean and truncate snippet
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                summary = snippet[:250] + ("..." if len(snippet) > 250 else "")

                results.append(
                    CaseDoc(
                        id=doc_id,
                        title=title[:500],  # Limit title length
                        court=court,
                        date=date,
                        citation=citation,  # Use the extracted citation if available
                        url=url,
                        summary=summary,
                    )
                )
                
                # Respect rate limiting
                if len(results) < limit and len(results) % 2 == 0:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.warning(f"Error parsing search result: {str(e)}", exc_info=True)
                continue

        return results

    except httpx.TimeoutException:
        logger.warning("Scraping timeout")
        return []
    except httpx.RequestError as e:
        logger.error(f"Request error during scraping: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {str(e)}", exc_info=True)
        return []


def scrape_indian_kanoon_search(query: str, limit: int = 5) -> List[CaseDoc]:
    """Heuristically scrape Indian Kanoon search results page without API.
    Returns a list of CaseDoc with best-effort fields (title, url, snippet as summary).
    WARNING: Scraping may be brittle if the site markup changes. Use responsibly and respect robots.txt / ToS.
    """
    try:
        # Clean and encode the query
        query = query.strip()
        if not query:
            return []
            
        # Build search URL with a more specific query
        base_url = "https://indiankanoon.org/search/"
        params = {
            'formInput': f'"{query}"',  # Use exact phrase matching
            'pagenum': 0,
            'sortby': 'mostrecent',
            'type': 'judgments',
            'fromdate': '01-01-1950',  # Limit to post-independence cases
            'from': '01-01-1950',
            'to': datetime.now().strftime("%d-%m-%Y"),
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://indiankanoon.org/',
            'Cache-Control': 'no-cache',
        }
        
        # Make the request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    base_url, 
                    params=params, 
                    headers=headers, 
                    timeout=30,
                    allow_redirects=True
                )
                response.raise_for_status()
                break
            except (requests.RequestException, requests.Timeout) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch search results after {max_retries} attempts: {str(e)}")
                    return []
                time.sleep(1)  # Wait before retrying
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Find all result divs with more specific selectors
        result_divs = soup.select('div.result, div.search-result')
        
        for div in result_divs[:limit]:
            try:
                # Extract title and URL
                title_elem = div.select_one('a[href^="/doc/"]')
                if not title_elem:
                    continue
                    
                title = title_elem.get_text(strip=True)
                doc_id = title_elem['href'].split('/')[-2]
                if not doc_id.isdigit():
                    continue
                    
                url = f"https://indiankanoon.org/doc/{doc_id}/"
                
                # Extract court and date with better parsing
                doc_info = div.select_one('.docsource_main, .doc_subtitle, .result-subtitle')
                court = None
                date = None
                citation = None
                
                if doc_info:
                    info_text = doc_info.get_text(strip=True, separator='|')
                    parts = [p.strip() for p in info_text.split('|') if p.strip()]
                    
                    # Try to extract court and date from different formats
                    for part in parts:
                        # Check for date-like patterns
                        if re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', part) or 'on ' in part.lower():
                            date = part.replace('on ', '').strip()
                        # Check for citation patterns
                        elif re.search(r'\b[A-Za-z]+\.?\s*\d+\s*\w*\s*\d+', part):
                            citation = part.strip()
                        # Assume the first non-date, non-citation part is the court
                        elif not date and not citation and len(part) < 100:  # Arbitrary length check
                            court = part
                        break
                
                # Extract snippet/summary
                snippet = ""
                snippet_el = div.select_one("p, .doc, .doc_summary")
                if snippet_el:
                    snippet = " ".join(snippet_el.get_text(" ", strip=True).split())
                
                # If no snippet, use first few words of title and metadata
                if not snippet:
                    meta_text = " ".join(
                        p.get_text(" ", strip=True) 
                        for p in div.select("div.doc_box, .docsource, .doc_date, .doc_subtitle")
                        if p.get_text(strip=True)
                    )
                    snippet = f"{title}. {meta_text}"
                
                # Clean and truncate snippet
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                summary = snippet[:250] + ("..." if len(snippet) > 250 else "")

                results.append(
                    CaseDoc(
                        id=doc_id,
                        title=title[:500],  # Limit title length
                        court=court,
                        date=date,
                        citation=citation,  # Use the extracted citation if available
                        url=url,
                        summary=summary,
                    )
                )
                
                # Respect rate limiting
                if len(results) < limit and len(results) % 2 == 0:
                    time.sleep(1)
                    
            except Exception as e:
                logger.warning(f"Error parsing search result: {str(e)}", exc_info=True)
                continue

        return results

    except requests.Timeout:
        logger.warning("Scraping timeout")
        return []
    except requests.RequestException as e:
        logger.error(f"Request error during scraping: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {str(e)}", exc_info=True)
        return []


async def summarize_cases_async(cases: List[CaseDoc], user_case: str) -> List[CaseDoc]:
    """Async version of summarize_cases with parallel processing for better performance."""
    if not cases:
        return []
        
    # Process cases in batches for better performance
    batch_size = 3
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    all_processed_cases = []
    
    for batch in batches:
        # Create tasks for parallel processing
        tasks = []
        for case in batch:
            task = asyncio.create_task(_summarize_single_case_async(case, user_case))
            tasks.append(task)
        
        # Wait for all tasks in the batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and add successful results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error summarizing case: {str(result)}")
            else:
                all_processed_cases.append(result)
        
        # Small delay between batches to be respectful to the LLM API
        if len(batches) > 1:
            await asyncio.sleep(0.5)
    
    return all_processed_cases

async def _summarize_single_case_async(case: CaseDoc, user_case: str) -> CaseDoc:
    """Summarize a single case asynchronously."""
    try:
        # Create a compact prompt for a single case
        prompt = (
            f"You are a legal assistant. Given a user's case description and a case metadata, "
            f"write a 2-3 sentence summary and a one-line reason why it might be relevant.\n\n"
            f"User case description: {user_case}\n\n"
            f"Case to summarize:\n"
            f"- Title: {case.title}\n"
            f"- Court: {case.court or 'N/A'}\n"
            f"- Date: {case.date or 'N/A'}\n"
            f"- Citation: {case.citation or 'N/A'}\n"
            f"- URL: {case.url or 'N/A'}\n\n"
            f"Return output in this format:\n"
            f"Summary: ...\n"
            f"Relevance: ..."
        )
        
        comp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )
        text = comp.content if hasattr(comp, "content") else str(comp)
        
        # Extract summary and relevance
        summary_match = re.search(r'Summary:\s*(.+?)(?=\nRelevance:|$)', text, re.DOTALL)
        relevance_match = re.search(r'Relevance:\s*(.+)', text, re.DOTALL)
        
        summary_text = ""
        if summary_match:
            summary_text += f"Summary: {summary_match.group(1).strip()}"
        if relevance_match:
            summary_text += f"\nRelevance: {relevance_match.group(1).strip()}"
        
        case.summary = summary_text or "Unable to generate summary."
        return case
        
    except Exception as e:
        logger.error(f"Error summarizing case {case.id}: {str(e)}")
        case.summary = "Summary generation failed."
        return case


def summarize_cases(cases: List[CaseDoc], user_case: str) -> List[CaseDoc]:
    """Use LLM to produce concise, case-wise summaries and relevance notes."""
    if not cases:
        return []
    # Create a compact prompt
    bullet_context = "\n".join(
        [
            f"- Title: {c.title}\n  Court: {c.court or 'N/A'}\n  Date: {c.date or 'N/A'}\n  Citation: {c.citation or 'N/A'}\n  URL: {c.url or 'N/A'}"
            for c in cases
        ]
    )
    sys_prompt = (
        "You are a legal assistant. Given a user's case description and a list of Indian case metadata, "
        "write a 3-4 sentence summary for each case and a one-line reason why it might be relevant."
    )
    user_prompt = (
        f"User case description:\n{user_case}\n\nCases to summarize (metadata only):\n{bullet_context}\n\n"
        "Return output as numbered list, one block per case, strictly in this format:\n"
        "1) <Title> — <Court> (<Date>) [<Citation>]\n   Summary: ...\n   Relevance: ...\n"
    )

    try:
        comp = llm.invoke(sys_prompt + "\n\n" + user_prompt)
        text = comp.content if hasattr(comp, "content") else str(comp)
    except Exception:
        text = ""

    # Heuristic parse: map summaries back to cases in order
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks: List[List[str]] = []
    current: List[str] = []
    for ln in lines:
        if ln[0:2].isdigit() and ")" in ln[:5]:
            if current:
                blocks.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        blocks.append(current)

    for idx, block in enumerate(blocks):
        if idx < len(cases):
            summary_text = " ".join(block)
            cases[idx].summary = summary_text
    return cases


def synthesize_search_query(user_case: str) -> str:
    """Use the LLM to turn a free-form case description into a concise legal search query.
    The query should contain key facts, sections/articles, and issue terms.
    """
    try:
        prompt = (
            "You are a legal research assistant. Convert the user's case description into a concise search query "
            "suitable for finding Indian case law on Indian Kanoon. Keep it under 20 words, include key legal terms, "
            "relevant sections/articles if present, and avoid personal names.\n\n"
            f"Case description:\n{user_case}\n\nReturn only the query, nothing else."
        )
        comp = llm.invoke(prompt)
        query = comp.content.strip() if hasattr(comp, "content") else str(comp).strip()
        # Basic cleanup
        query = query.replace("\n", " ").strip()
        return query[:180]
    except Exception:
        # Fallback: naive truncation
        return user_case[:180]


def build_case_documents(user_input: str, limit_cases: int = 3) -> List[Document]:
    """Fetch Indian Kanoon cases for the query, retrieve details, and return chunked Documents.
    Falls back gracefully to empty list if IK not available.
    """
    docs: List[Document] = []
    try:
        # 1) If the user pasted a direct Indian Kanoon URL or doc id, fetch that first
        direct_docs: List[Document] = []
        try:
            m = re.search(r"indiankanoon\.org\/doc\/(\d+)\/", user_input)
            if m:
                doc_id = m.group(1)
                details = scrape_case_details(doc_id)
                full_text = (details.get("full_text") or "")[:12000]
                if full_text:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
                    chunks = splitter.split_text(full_text)
                    meta = {
                        "source": "indian_kanoon",
                        "case_id": doc_id,
                        "title": details.get("title") or "Case",
                        "url": details.get("url"),
                        "court": details.get("court"),
                        "date": details.get("date"),
                    }
                    for ch in chunks:
                        direct_docs.append(Document(page_content=ch, metadata=meta))
        except Exception:
            pass

        query_used = synthesize_search_query(user_input)
        cases, ik_error = search_indian_kanoon(query_used, limit=limit_cases)
        # Hard fallback: if API unauthorized/missing but scraping is possible, try scrape directly here
        if not cases and (ik_error in {"missing_credentials", "unauthorized"} or not ik_error):
            scraped = scrape_indian_kanoon_search(query_used, limit=limit_cases)
            cases = scraped
        if not cases and direct_docs:
            return direct_docs
        if not cases:
            return []
        # For each case, fetch details and chunk limited text
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        for c in cases:
            details = scrape_case_details(c.id)
            full_text = (details.get("full_text") or "")[:8000]  # limit size per case
            if not full_text:
                continue
            chunks = splitter.split_text(full_text)
            meta = {
                "source": "indian_kanoon",
                "case_id": c.id,
                "title": details.get("title") or c.title,
                "url": details.get("url") or c.url,
                "court": details.get("court") or c.court,
                "date": details.get("date") or c.date,
            }
            for ch in chunks:
                docs.append(Document(page_content=ch, metadata=meta))
    except Exception as e:
        logger.warning(f"build_case_documents failed: {e}")
        return []
    return docs

def _extract_sources(docs: List[Document]) -> List[Dict[str, str]]:
    """Deduplicate and return list of source dicts with title and URL from provided Documents."""
    uniq: Dict[str, Dict[str, str]] = {}
    for d in docs:
        meta = d.metadata or {}
        case_id = str(meta.get("case_id") or meta.get("url") or meta.get("title") or "")
        if not case_id:
            # include constitution source once
            if "constitution" not in uniq and meta.get("source") == "constitution":
                uniq["constitution"] = {"title": "Indian Constitution context", "url": ""}
            continue
        if case_id not in uniq:
            uniq[case_id] = {
                "title": str(meta.get("title") or "Case"),
                "url": str(meta.get("url") or ""),
                "court": str(meta.get("court") or ""),
                "date": str(meta.get("date") or ""),
            }
    return list(uniq.values())

async def _rerank_documents(question: str, docs: List[Document], top_k: int = 15) -> List[Document]:
    """Return top_k documents most similar to the question using FAISS index or fallback to embeddings."""
    if not docs:
        return []
        
    global FAISS_INDEX, EMBEDDINGS
    
    try:
        # If we have a FAISS index, use it for similarity search
        if FAISS_INDEX is not None and EMBEDDINGS is not None:
            # Get the most similar documents from FAISS
            similar_docs = await FAISS_INDEX.asimilarity_search(question, k=top_k)
            return similar_docs
            
        # Fallback to sentence-transformers if FAISS is not available
        if EMBEDDINGS is None:
            EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False}
            )
            
        # Encode documents and query
        query_embedding = EMBEDDINGS.embed_query(question)
        doc_embeddings = [EMBEDDINGS.embed_query(doc.page_content) for doc in docs]
        
        # Calculate cosine similarities
        import numpy as np
        from numpy.linalg import norm
        
        query_norm = norm(query_embedding)
        similarities = []
        
        for doc_emb in doc_embeddings:
            doc_norm = norm(doc_emb)
            if query_norm > 0 and doc_norm > 0:
                sim = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
            else:
                sim = 0
            similarities.append(sim)
        
        # Sort documents by similarity
        doc_similarities = list(zip(docs, similarities))
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_similarities[:top_k]]
        
    except Exception as e:
        logger.error(f"Error in document reranking: {str(e)}", exc_info=True)
        # Fallback: return first top_k documents
        return docs[:top_k]

@app.post("/chat")
async def chat(input: str = Form(...), conversation_id: Optional[str] = Form(None)):
    """Handle chat requests using the RAG pipeline for document retrieval and response generation.
    
    Args:
        input: User's input message
        conversation_id: Optional conversation ID for maintaining context
        
    Returns:
        JSON response with the assistant's response and metadata
    """
    global retriever, document_chain, FAISS_INDEX, EMBEDDINGS
    
    # Debug: Log initialization status
    logger.info(f"[RAG Debug] Initializing chat endpoint")
    logger.info(f"[RAG Debug] Retriever initialized: {retriever is not None}")
    logger.info(f"[RAG Debug] Document chain initialized: {document_chain is not None}")
    logger.info(f"[RAG Debug] FAISS index available: {FAISS_INDEX is not None}")
    logger.info(f"[RAG Debug] Embeddings available: {EMBEDDINGS is not None}")
    
    if not retriever or not document_chain:
        error_msg = "Vector store or document chain not initialized"
        logger.error(f"[RAG Error] {error_msg}")
        return JSONResponse(status_code=500, content={"error": error_msg})

    # Determine conversation id
    conv_id = conversation_id or str(uuid4())
    history = chat_histories.get(conv_id, [])
    
    # Prepare chat history text for context
    history_text = "\n".join([f"{role}: {content}" for role, content in history])
    
    try:
        logger.info(f"[RAG Debug] Processing query: {input}")
        
        # Get relevant documents using the retriever
        try:
            if hasattr(retriever, 'invoke'):
                logger.info("[RAG Debug] Using invoke() method for retrieval")
                docs = await retriever.ainvoke(input)
            else:
                logger.info("[RAG Debug] Using get_relevant_documents() for retrieval")
                docs = await retriever.aget_relevant_documents(input)
                
            logger.info(f"[RAG Debug] Retrieved {len(docs)} documents")
            
            # Debug: Log document metadata
            for i, doc in enumerate(docs[:3]):  # Log first 3 docs to avoid too much output
                logger.info(f"[RAG Debug] Doc {i+1} metadata: {getattr(doc, 'metadata', 'No metadata')}")
                logger.info(f"[RAG Debug] Doc {i+1} content preview: {getattr(doc, 'page_content', '')[:200]}...")
                
        except Exception as e:
            logger.error(f"[RAG Error] Document retrieval failed: {str(e)}", exc_info=True)
            return {
                "response": "An error occurred while searching for relevant information.",
                "conversation_id": conv_id,
                "source": "error",
                "sources": [],
                "error": f"Retrieval error: {str(e)}"
            }
        
        if not docs:
            logger.warning("[RAG Warning] No documents retrieved from vector store")
            # Try to provide more helpful feedback based on the query
            if any(term in input.lower() for term in ["fundamental rights", "article", "constitution"]):
                return {
                    "response": "I couldn't find specific information about that topic in the constitution. "
                              "The query might be too specific or the relevant sections might not be in the indexed documents. "
                              "Try rephrasing your question or asking about a broader constitutional topic.",
                    "conversation_id": conv_id,
                    "source": "constitution",
                    "sources": []
                }
            return {
                "response": "I couldn't find any relevant information in the constitution to answer your question. "
                          "The query might be outside the scope of the available constitutional documents or too specific. "
                          "Please try rephrasing your question.",
                "conversation_id": conv_id,
                "source": "constitution",
                "sources": []
            }
            
        # Ensure all documents have proper source metadata
        for doc in docs:
            meta = getattr(doc, 'metadata', {}) or {}
            meta["source"] = meta.get("source") or "constitution"
            doc.metadata = meta
        
        # Rerank documents by relevance to the query
        ranked_docs = await _rerank_documents(input, docs, top_k=5)  # Reduced top_k for better performance
        
        # Prepare the input for the document chain
        chain_input = {
            'input': input,
            'chat_history': history_text,
            'context': ranked_docs
        }
        
        # Generate response using the document chain
        if hasattr(document_chain, 'ainvoke'):
            response = await document_chain.ainvoke(chain_input)
        else:
            response = document_chain.invoke(chain_input)

        # Extract the answer from the response object
        if isinstance(response, dict):
            answer = response.get('answer', '') or response.get('text', '')
        elif hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        # Extract sources from the ranked documents
        sources = _extract_sources(ranked_docs) if ranked_docs else []

        # Update conversation history
        history.append(("user", input))
        history.append(("assistant", answer))
        chat_histories[conv_id] = history[-10:]  # Keep last 10 messages

        return {
            "response": answer, 
            "conversation_id": conv_id, 
            "source": "constitution", 
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        error_response = {
            "response": "An error occurred while processing your request. Please try again.",
            "conversation_id": conv_id,
            "error": str(e) if str(e) else "Unknown error"
        }
        history.append(("assistant", error_response["response"]))
        chat_histories[conv_id] = history[-10:]
        return JSONResponse(status_code=500, content=error_response)


@app.options("/cases/search")
async def options_cases_search():
    """Handle CORS preflight requests."""
    return {"status": "ok"}

@app.post("/cases/search")
async def cases_search(input: str = Form(...), limit: int = Form(5)):
    """Optimized endpoint to search Indian Kanoon and return summarized cases with async processing."""
    start_time = time.time()
    
    try:
        # Input validation
        if not input or not input.strip():
            raise HTTPException(
                status_code=400, 
                detail="Input query cannot be empty"
            )
        
        input = input.strip()
        if len(input) > 2000:
            raise HTTPException(
                status_code=400, 
                detail="Input query is too long (maximum 2000 characters)"
            )
        
        # Validate limit parameter
        if limit < 1 or limit > 20:
            raise HTTPException(
                status_code=400, 
                detail="Limit must be between 1 and 20"
            )
        
        logger.info(f"Processing cases search request: '{input[:100]}...' (limit: {limit})")
        
        # Check cache first for the complete request
        cache_key = generate_cache_key("cases_search", input, limit)
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached cases search result")
            response = JSONResponse(content=cached_result)
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["X-Cache"] = "HIT"
            return response
        
        # Use async query synthesis
        query_used = await asyncio.get_event_loop().run_in_executor(
            None, synthesize_search_query, input
        )
        
        # Use async search function
        cases, ik_error = await search_indian_kanoon_async(query_used, limit=limit)
        
        # Use async summarization with parallel processing
        summarized = await summarize_cases_async(cases, input) if cases else []
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        response_data = {
            "response": f"Found {len(summarized)} potentially relevant cases.",
            "cases": [c.dict() for c in summarized],
            "source": "indian_kanoon",
            "ik_error": ik_error,
            "query_used": query_used,
            "success": True,
            "processing_time": round(processing_time, 3),
            "cache_status": "MISS",
            "total_found": len(cases),
            "total_summarized": len(summarized)
        }
        
        # Cache the result for future requests
        set_cache(cache_key, response_data, ttl=1800)  # Cache for 30 minutes
        
        response = JSONResponse(content=response_data)
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        
        logger.info(f"Cases search completed in {processing_time:.3f}s, found {len(summarized)} cases")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in cases_search after {processing_time:.3f}s: {str(e)}", exc_info=True)
        
        error_response = {
            "response": "An error occurred while processing your request.",
            "cases": [],
            "source": "error",
            "ik_error": str(e),
            "success": False,
            "processing_time": round(processing_time, 3),
            "cache_status": "ERROR"
        }
        
        response = JSONResponse(content=error_response, status_code=500)
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["X-Processing-Time"] = str(round(processing_time, 3))
        return response


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def scrape_case_details(doc_id: str) -> Dict[str, Optional[str]]:
    """Fetch and parse a single Indian Kanoon case page for details.
    Returns dict with title, court, date, citation, full_text, and other metadata.
    Adds detailed logging and a hard timeout to prevent backend hangs.
    """
    import concurrent.futures
    logger.info(f"[scrape_case_details] Start for doc_id={doc_id}")
    if not doc_id or not doc_id.strip() or not doc_id.strip().isdigit():
        logger.warning(f"[scrape_case_details] Invalid document ID: {doc_id}")
        return {"error": "Invalid document ID"}
    doc_id = doc_id.strip()
    url = f"https://indiankanoon.org/doc/{doc_id}/"
    # Check cache first
    cache_key = generate_cache_key("case_details", doc_id)
    cached_result = get_from_cache(cache_key)
    if cached_result and cached_result.get("full_text"):
        logger.info(f"[scrape_case_details] Cache hit for doc_id={doc_id}")
        return cached_result

    def _scrape():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        logger.info(f"[scrape_case_details] Requesting {url}")
        time.sleep(1)
        resp = requests.get(url, headers=headers, timeout=10)
        logger.info(f"[scrape_case_details] Response {resp.status_code} for {url}")
        if resp.status_code != 200:
            logger.warning(f"[scrape_case_details] HTTP error {resp.status_code} for {url}")
            return {"url": url, "error": f"HTTP {resp.status_code}"}
        soup = BeautifulSoup(resp.text, "html.parser")
        result = {"url": url}
        logger.info(f"[scrape_case_details] Parsed HTML for doc_id={doc_id}")
        # 1. Extract title
        title_el = soup.select_one('h1.doctitle, h1.title, title')
        if title_el:
            title = title_el.get_text(strip=True)
            title = re.sub(r'\s*-\s*Indian\s*Kanoon\s*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'^\s*[0-9]+\s*', '', title).strip()
            result["title"] = title
        logger.info(f"[scrape_case_details] Extracted title for doc_id={doc_id}")
        # 2. Extract metadata (court, date, citations, etc.)
        logger.info(f"[scrape_case_details] Extracting metadata for doc_id={doc_id}")
        metadata = {}
        meta_selectors = [
            '.docsource', '.docinfo', '.subtitle', '.doc_title', 
            '.doc_citations', '.judgments', '.judgment'
        ]
        meta_text = ""
        for selector in meta_selectors:
            elements = soup.select(selector)
            if elements:
                meta_text += "\n" + "\n".join(
                    el.get_text("\n", strip=True) 
                    for el in elements
                )
        logger.info(f"[scrape_case_details] Extracted meta_text for doc_id={doc_id}")
        court = None
        date = None
        citation = None
        doc_meta = soup.find('div', class_='doc_meta')
        if doc_meta:
            meta_text = doc_meta.get_text('\n')
            court_match = re.search(r'(?:Bench|Court):\s*([^\n]+)', meta_text, re.IGNORECASE)
            if court_match:
                court = court_match.group(1).strip()
            date_match = re.search(r'(?:Judgment Date|On):\s*([^\n]+)', meta_text, re.IGNORECASE)
            if date_match:
                date = date_match.group(1).strip()
            cite_match = re.search(r'Citation:\s*([^\n]+)', meta_text, re.IGNORECASE)
            if cite_match:
                citation = cite_match.group(1).strip()
        # Extract main content - try multiple selectors
        logger.info(f"[scrape_case_details] Extracting main content for doc_id={doc_id}")
        content_selectors = [
            '#pre_1',  # Main content container
            '.judgments',
            '.judgment',
            '.doc_content',
            '.doc',
            '#content',
            '.content',
            'body'
        ]
        full_text = None
        for selector in content_selectors:
            content_el = soup.select_one(selector)
            if content_el:
                for el in content_el.select('.hidden, script, style, noscript, .noprint, .ad, .ad-container, .disclaimer, .hidden-print'):
                    el.decompose()
                paragraphs = []
                for p in content_el.find_all(['p', 'div'], recursive=True):
                    text = p.get_text(' ').strip()
                    if text and len(text) > 20:
                        paragraphs.append(text)
                full_text = '\n\n'.join(paragraphs)
                full_text = re.sub(r'\s+', ' ', full_text)
                full_text = re.sub(r'\n{3,}', '\n\n', full_text)
                if full_text and len(full_text) > 1000:
                    break
        logger.info(f"[scrape_case_details] Extracted main content for doc_id={doc_id}")
        if not full_text or len(full_text) < 1000:
            logger.info(f"[scrape_case_details] Trying aggressive content extraction for doc_id={doc_id}")
            paragraphs = [p.get_text(' ').strip() for p in soup.find_all(['p', 'div'])]
            paragraphs = [p for p in paragraphs if len(p) > 50]
            if paragraphs:
                full_text = '\n\n'.join(p for p in paragraphs)
        if full_text and len(full_text) > 50000:
            full_text = full_text[:50000] + "\n[... Content truncated due to length ...]"
        result["full_text"] = full_text
        logger.info(f"[scrape_case_details] Finished all extraction for doc_id={doc_id}")
        # 4. Extract key sections if possible (judges, facts, issues, etc.)
        logger.info(f"[scrape_case_details] Extracting sections for doc_id={doc_id}")
        sections = {}
        judge_patterns = [
            r'(?i)(?:before|hon[\'\w\s]*?|presided over by|delivered by)[\s,:]*((?:[A-Z][\w\s]+,?\s+)+(?:J\.|J J\.|JJ\.|Justice|Hon[\'\w\s]*?))',
            r'(?i)(?:JUDGMENT|JUDGMENT\\s+OF|JUDGMENT\\s+BY|JUDGMENT\\s+BY\\s+THE\\s+COURT)[\\s\\n]*(.*?)(?=\\n\\n|$)',
        ]
        for pattern in judge_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                judges_text = match.group(1).strip()
                judges_text = re.sub(r'\s+', ' ', judges_text)
                judges_text = re.sub(r'\s*,\s*', ', ', judges_text)
                sections["judges"] = judges_text
                break
        fact_patterns = [
            r'(?i)(?:FACTS|FACTS\\s+OF\\s+THE\\s+CASE|BRIEF\\s+FACTS)[\\s\\n]*(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
            r'(?i)(?:The\\s+facts[\\s\\w,]*?are[\\s\\w,]*?:|Facts[\\s\\w,]*?:)(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
        ]
        for pattern in fact_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                facts = match.group(1).strip()
                if len(facts) > 100:
                    sections["facts"] = facts
                    break
        issue_patterns = [
            r'(?i)(?:ISSUE[S]?|POINT[S]?\\s+FOR\\s+CONSIDERATION)[\\s\\n]*(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
            r'(?i)(?:The\\s+following\\s+issue[s]?[\\s\\w,]*?ar[ie]s[\\s\\w,]*?:)(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
        ]
        for pattern in issue_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                issues = match.group(1).strip()
                if len(issues) > 30:
                    sections["issues"] = issues
                    break
        if sections:
            result["sections"] = sections
        logger.info(f"[scrape_case_details] Finished sections for doc_id={doc_id}")
        set_cache(cache_key, result, ttl=86400)
        logger.info(f"[scrape_case_details] Finished and cached for doc_id={doc_id}")
        return result

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_scrape)
            return future.result(timeout=20)
    except concurrent.futures.TimeoutError:
        logger.error(f"TimeoutError in scrape_case_details for {url}: thread timeout")
        return {"url": url, "error": "TimeoutError"}
    except requests.Timeout:
        logger.error(f"Timeout while fetching case details for {url}")
        return {"url": url, "error": "Request timed out"}
    except requests.RequestException as e:
        logger.error(f"Request error while fetching case details: {str(e)}")
        return {"url": url, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Error parsing case details: {str(e)}", exc_info=True)
        return {"title": None, "court": None, "date": None, "citation": None, "url": f"https://indiankanoon.org/doc/{doc_id}/", "full_text": None, "error": f"Parsing error: {str(e)}"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        logger.info(f"[scrape_case_details] Requesting {url}")
        time.sleep(1)
        resp = requests.get(url, headers=headers, timeout=10)
        logger.info(f"[scrape_case_details] Response {resp.status_code} for {url}")
        if resp.status_code != 200:
            logger.warning(f"[scrape_case_details] HTTP error {resp.status_code} for {url}")
            return {"url": url, "error": f"HTTP {resp.status_code}"}
        soup = BeautifulSoup(resp.text, "html.parser")
        result = {"url": url}
        logger.info(f"[scrape_case_details] Parsed HTML for doc_id={doc_id}")
        
        # 1. Extract title
        title_el = soup.select_one('h1.doctitle, h1.title, title')
        if title_el:
            title = title_el.get_text(strip=True)
            # Clean up common prefixes/suffixes
            title = re.sub(r'\s*-\s*Indian\s*Kanoon\s*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'^\s*[0-9]+\s*', '', title).strip()
            result["title"] = title
        logger.info(f"[scrape_case_details] Extracted title for doc_id={doc_id}")
        
        # 2. Extract metadata (court, date, citations, etc.)
        logger.info(f"[scrape_case_details] Extracting metadata for doc_id={doc_id}")
        metadata = {}
        meta_selectors = [
            '.docsource', '.docinfo', '.subtitle', '.doc_title', 
            '.doc_citations', '.judgments', '.judgment'
        ]
        meta_text = ""
        for selector in meta_selectors:
            elements = soup.select(selector)
            if elements:
                meta_text += "\n" + "\n".join(
                    el.get_text("\n", strip=True) 
                    for el in elements
                )
        logger.info(f"[scrape_case_details] Extracted meta_text for doc_id={doc_id}")
        
        # Extract court information
        court = None
        court_patterns = [
            (r'(Supreme Court of India|SC|SC[A-Z]+)', 'Supreme Court of India'),
            (r'(High Court of [A-Za-z\s]+)', 'High Court'),
            (r'(Supreme Court|SC)\b', 'Supreme Court'),
            (r'(High Court|HC)\b', 'High Court'),
            (r'(District Court|DC)\b', 'District Court')
        ]
        
        date = None
        citation = None
        
        # Try to find metadata in the document
        doc_meta = soup.find('div', class_='doc_meta')
        if doc_meta:
            meta_text = doc_meta.get_text('\n')
            
            # Extract court
            court_match = re.search(r'(?:Bench|Court):\s*([^\n]+)', meta_text, re.IGNORECASE)
            if court_match:
                court = court_match.group(1).strip()
            
            # Extract date
            date_match = re.search(r'(?:Judgment Date|On):\s*([^\n]+)', meta_text, re.IGNORECASE)
            if date_match:
                date = date_match.group(1).strip()
            
            # Extract citation
            cite_match = re.search(r'Citation:\s*([^\n]+)', meta_text, re.IGNORECASE)
            if cite_match:
                citation = cite_match.group(1).strip()
        
        # Extract main content - try multiple selectors
        logger.info(f"[scrape_case_details] Extracting main content for doc_id={doc_id}")
        content_selectors = [
            '#pre_1',  # Main content container
            '.judgments',
            '.judgment',
            '.doc_content',
            '.doc',
            '#content',
            '.content',
            'body'
        ]
        full_text = None
        for selector in content_selectors:
            content_el = soup.select_one(selector)
            if content_el:
                for el in content_el.select('.hidden, script, style, noscript, .noprint, .ad, .ad-container, .disclaimer, .hidden-print'):
                    el.decompose()
                paragraphs = []
                for p in content_el.find_all(['p', 'div'], recursive=True):
                    text = p.get_text(' ').strip()
                    if text and len(text) > 20:
                        paragraphs.append(text)
                full_text = '\n\n'.join(paragraphs)
                full_text = re.sub(r'\s+', ' ', full_text)
                full_text = re.sub(r'\n{3,}', '\n\n', full_text)
                if full_text and len(full_text) > 1000:
                    break
        logger.info(f"[scrape_case_details] Extracted main content for doc_id={doc_id}")
        if not full_text or len(full_text) < 1000:
            logger.info(f"[scrape_case_details] Trying aggressive content extraction for doc_id={doc_id}")
            paragraphs = [p.get_text(' ').strip() for p in soup.find_all(['p', 'div'])]
            paragraphs = [p for p in paragraphs if len(p) > 50]
            if paragraphs:
                full_text = '\n\n'.join(p for p in paragraphs)
        if full_text and len(full_text) > 50000:
            full_text = full_text[:50000] + "\n[... Content truncated due to length ...]"
        
        result["full_text"] = full_text
        logger.info(f"[scrape_case_details] Finished all extraction for doc_id={doc_id}")
        # 4. Extract key sections if possible (judges, facts, issues, etc.)
        logger.info(f"[scrape_case_details] Extracting sections for doc_id={doc_id}")
        sections = {}
        judge_patterns = [
            r'(?i)(?:before|hon[\'\w\s]*?|presided over by|delivered by)[\s,:]*((?:[A-Z][\w\s]+,?\s+)+(?:J\.|J J\.|JJ\.|Justice|Hon[\'\w\s]*?))',
            r'(?i)(?:JUDGMENT|JUDGMENT\\s+OF|JUDGMENT\\s+BY|JUDGMENT\\s+BY\\s+THE\\s+COURT)[\\s\\n]*(.*?)(?=\\n\\n|$)',
        ]
        for pattern in judge_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                judges_text = match.group(1).strip()
                judges_text = re.sub(r'\s+', ' ', judges_text)
                judges_text = re.sub(r'\s*,\s*', ', ', judges_text)
                sections["judges"] = judges_text
                break
        fact_patterns = [
            r'(?i)(?:FACTS|FACTS\\s+OF\\s+THE\\s+CASE|BRIEF\\s+FACTS)[\\s\\n]*(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
            r'(?i)(?:The\\s+facts[\\s\\w,]*?are[\\s\\w,]*?:|Facts[\\s\\w,]*?:)(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
        ]
        for pattern in fact_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                facts = match.group(1).strip()
                if len(facts) > 100:
                    sections["facts"] = facts
                    break
        issue_patterns = [
            r'(?i)(?:ISSUE[S]?|POINT[S]?\\s+FOR\\s+CONSIDERATION)[\\s\\n]*(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
            r'(?i)(?:The\\s+following\\s+issue[s]?[\\s\\w,]*?ar[ie]s[\\s\\w,]*?:)(.*?)(?=\\n\\n[A-Z\\s]+\\n|$)',
        ]
        for pattern in issue_patterns:
            match = re.search(pattern, full_text or "", re.DOTALL)
            if match:
                issues = match.group(1).strip()
                if len(issues) > 30:
                    sections["issues"] = issues
                    break
        if sections:
            result["sections"] = sections
        logger.info(f"[scrape_case_details] Finished sections for doc_id={doc_id}")
        set_cache(cache_key, result, ttl=86400)
        logger.info(f"[scrape_case_details] Finished and cached for doc_id={doc_id}")
        signal.alarm(0)  # Cancel alarm
        return result
    except requests.Timeout:
        logger.error(f"Timeout while fetching case details for {url}")
        return {"url": url, "error": "Request timed out"}
    except TimeoutError as e:
        logger.error(f"TimeoutError in scrape_case_details for {url}: {str(e)}")
        return {"url": url, "error": "TimeoutError"}
    except requests.RequestException as e:
        logger.error(f"Request error while fetching case details: {str(e)}")
        return {"url": url, "error": f"Request failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Error parsing case details: {str(e)}", exc_info=True)
        return {"title": None, "court": None, "date": None, "citation": None, "url": f"https://indiankanoon.org/doc/{doc_id}/", "full_text": None, "error": f"Parsing error: {str(e)}"}


def summarize_case_detail(user_case: str, title: Optional[str], full_text: Optional[str]) -> Dict[str, object]:
    """Generate a structured legal case summary with relevance to the user's case.
    
    Args:
        user_case: The user's case description or query
        title: The title of the case being summarized
        full_text: The full text of the case (will be truncated if too long)
        
    Returns:
        Dict containing structured summary and relevance analysis
    """
    # Keep the text at a manageable length
    text = (full_text or "")[:6000]
    
    try:
        # Prepare the prompt with clear instructions
        prompt = (
            "Analyze this legal case and provide a structured response:\n\n"
            f"USER'S CASE: {user_case}\n\n"
            f"CASE TITLE: {title or 'N/A'}\n\n"
            f"CASE TEXT: {text}\n\n"
            "Provide a response with these sections:\n"
            "1. Case Summary (4-5 sentences)\n"
            "2. Key Legal Principles (bullets)\n"
            "3. Similarities to User's Case (bullets)\n"
            "4. Relevance to User's Situation (brief paragraph)"
        )
        
        # Get the LLM response
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the response into sections
        sections = {
            "summary": "",
            "legal_principles": [],
            "similarities": [],
            "relevance": "",
            "success": True
        }
        
        # Extract the case summary (first paragraph)
        summary_match = re.search(r'(?i)case summary[\s\n:]*([^\n]+(?:\n[^\n]+){0,4})', content, re.IGNORECASE)
        if summary_match:
            sections["summary"] = summary_match.group(1).strip()
        
        # Extract legal principles (bullets after 'Key Legal Principles')
        principles_match = re.search(r'(?i)key legal principles[\s\n:]*([^•\n]*(?:\n[^•\n]*)*)', content, re.IGNORECASE)
        if principles_match:
            principles_text = principles_match.group(1)
            sections["legal_principles"] = [p.strip() for p in re.findall(r'[•*-]\s*([^\n]+)', principles_text)][:5]
        
        # Extract similarities (bullets after 'Similarities to User\'s Case')
        similarities_match = re.search(r"(?i)similarities to user['\"]s case[\s\n:]*([^•\n]*(?:\n[^•\n]*)*)", content, re.IGNORECASE)
        if similarities_match:
            similarities_text = similarities_match.group(1)
            sections["similarities"] = [s.strip() for s in re.findall(r'[•*-]\s*([^\n]+)', similarities_text)][:5]
        
        # Extract relevance (paragraph after 'Relevance to User\'s Situation')
        relevance_match = re.search(r"(?i)relevance to user['\"]s situation[\s\n:]*([^\n]+(?:\n[^\n]+){0,3})", content, re.IGNORECASE)
        if relevance_match:
            sections["relevance"] = relevance_match.group(1).strip()
        
        return sections
        
    except Exception as e:
        logger.error(f"Error generating case summary: {str(e)}")
        return {
            "summary": "Failed to generate summary. Please try again later.",
            "legal_principles": [],
            "similarities": [],
            "relevance": "",
            "success": False,
            "error": str(e)
        }


## Removed redundant case_details route. Canonical implementation lives in app/routers/cases.py


@app.get("/")
def root():
    return {"message": "RAG FastAPI backend running"}
