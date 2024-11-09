from langchain_core.documents import Document
from typing import List
from langchain_cohere import CohereRerank
import os  
from RAG_Logger import logger
from dotenv import load_dotenv

load_dotenv()

def rerank_documents(documents: List[Document], query: str, top_n: int = 5) -> str:
  """
  Rerank documents using Cohere and return document contents.
  
  Args:
      documents: List of documents to rerank
      query: The query to use for reranking
      
  Returns:
      Concatenated string of reranked document contents
  """
  try:
    logger.info(f"Starting document reranking for {len(documents)} documents")
    logger.debug(f"Query: '{query}', top_n: {top_n}")
    
    if not documents:
        logger.warning("No documents provided for reranking")
        return ""
        
    # Get API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")

    # Extract text from documents
    texts = [doc.page_content for doc in documents]


    reranker = CohereRerank(
                    cohere_api_key=os.getenv("COHERE_API_KEY"),
                    model="rerank-english-v3.0"
                )
    
    # Perform reranking
    try:
        logger.debug("Performing document reranking")
        rerank_results = reranker.rerank(
            documents=texts,
            query=query,
            top_n=top_n
        )
        logger.debug(f"Successfully reranked {len(rerank_results)} documents")
    except Exception as e:
        logger.error("Failed during document reranking")
        logger.error(f"Error details: {str(e)}")
        raise

    # Create a mapping of text to original document
    text_to_doc = {doc.page_content: doc for doc in documents}
    
    # Get reranked documents with their metadata
    reranked_docs = []
    for result in rerank_results:
        # Get the original document text
        original_doc = text_to_doc[texts[result['index']]]
        
        # Create new document with rerank score in metadata
        reranked_doc = Document(
            page_content=original_doc.page_content,
            metadata={
                **original_doc.metadata,
                'rerank_score': result['relevance_score']
            }
        )
        reranked_docs.append(reranked_doc)

    logger.debug(f"Processed {len(reranked_docs)} reranked documents")
    
    # Combine reranked document contents
    reranked_text = "".join(
        [f"{doc.page_content}\n\n" for doc in reranked_docs]
    )
    
    return reranked_text
  
  except Exception as e:
        logger.error("Fatal error in document reranking process")
        logger.error(f"Error details: {str(e)}")
        return None