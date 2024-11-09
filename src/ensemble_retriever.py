from langchain.retrievers import EnsembleRetriever
from RAG_Logger import logger
from typing import Optional

def get_ensemble_retriever(pinecone_retriever, bm25_retriever) -> Optional[EnsembleRetriever]:
    """
    Create an ensemble retriever combining Pinecone and BM25 retrievers.
    
    Args:
        pinecone_retriever: Initialized Pinecone retriever
        bm25_retriever: Initialized BM25 retriever
        
    Returns:
        Optional[EnsembleRetriever]: Configured ensemble retriever or None if initialization fails
    """
    try:
        logger.info("Initializing ensemble retriever")
        
        # Validate retrievers
        if not pinecone_retriever or not bm25_retriever:
            raise ValueError("Both retrievers must be provided and properly initialized")
            
        logger.debug("Creating ensemble with weights: Pinecone=0.5, BM25=0.5")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[pinecone_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        logger.info("Successfully initialized ensemble retriever")
        return ensemble_retriever
        
    except ValueError as e:
        logger.error("Invalid retriever configuration")
        logger.error(f"Error details: {str(e)}")
        return None
        
    except Exception as e:
        logger.error("Failed to create ensemble retriever")
        logger.error(f"Error details: {str(e)}")
        return None