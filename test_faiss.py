import os
import sys
import logging
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_faiss_index():
    try:
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding
        test_embedding = embeddings.embed_query("test")
        logger.info(f"Embedding dimension: {len(test_embedding)}")
        
        # Create a simple test document
        test_docs = [
            {"page_content": "The Constitution of India is the supreme law of India.", "metadata": {}},
            {"page_content": "The document lays down the framework for the government.", "metadata": {}},
            {"page_content": "It defines fundamental rights, directive principles, and duties of citizens.", "metadata": {}}
        ]
        
        # Create a simple vector store
        logger.info("Creating test FAISS index...")
        from langchain.docstore.document import Document
        documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
                    for doc in test_docs]
        
        # Create and save the index
        index_path = str(Path(__file__).parent.absolute() / "test_faiss_index")
        os.makedirs(index_path, exist_ok=True)
        
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        
        logger.info(f"Test FAISS index created successfully at {index_path}")
        
        # Test search
        query = "What is the supreme law of India?"
        logger.info(f"Testing search with query: {query}")
        
        results = vector_store.similarity_search(query, k=1)
        logger.info(f"Search result: {results[0].page_content}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_faiss_index: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_faiss_index()
    if success:
        logger.info("FAISS test completed successfully!")
    else:
        logger.error("FAISS test failed!")
