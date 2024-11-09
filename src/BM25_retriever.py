from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List, Optional
from RAG_Logger import logger

def get_BM25_retriever(docs: List[Document], k: int = 10) -> Optional[BM25Retriever]:
    """
    Initialize a BM25 retriever with the given documents.
    
    Args:
        docs (List[Document]): List of documents to index
        k (int): Number of documents to retrieve (default: 10)
        
    Returns:
        Optional[BM25Retriever]: Configured BM25 retriever or None if initialization fails
    """
    try:
        logger.info(f"Initializing BM25 retriever with {len(docs)} documents")
        logger.debug(f"Retrieval parameter k={k}")
        updated_docs = []
        for doc in docs:
            new_metadata = doc.metadata.copy()
            new_metadata['search_source'] = "BM25"

            # Create new document with updated metadata
            updated_chunk = Document(
                page_content=doc.page_content,
                metadata=new_metadata
            )

            updated_docs.append(updated_chunk)

        bm25_retriever = BM25Retriever.from_documents(updated_docs, k=10)
        logger.info("BM25 retriever initialized successfully")
        return bm25_retriever
    
    except Exception as e:
        logger.error("Fatal error in BM25 retriever initialization")
        logger.error(f"Error details: {str(e)}")
        return None
