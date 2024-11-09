from langchain_community.document_loaders import DirectoryLoader
from RAG_Logger import logger
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional


def load_pdf_documents(directory_path: str) -> Optional[List]:
    """
    Load all PDF documents from a specified directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF files
        
    Returns:
        Optional[List]: List of loaded documents or None if loading fails
    """
    try:
        logger.info(f"Starting to load PDF documents from: {directory_path}")
        
        # Initialize DirectoryLoader with PDF loader
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        logger.info("Loading documents...")
        documents = loader.load()
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {directory_path}")
        logger.error(f"Error details: {str(e)}")
        return None
        
    except Exception as e:
        logger.error("An unexpected error occurred while loading documents")
        logger.error(f"Error details: {str(e)}")
        return None