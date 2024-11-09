from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from dotenv import load_dotenv
import time
import os
from typing import List, Optional
from langchain.schema import Document
from RAG_Logger import logger  # Assuming this is already properly configured

load_dotenv()

def get_pinecone_retriever(index_name: str, chunks: List[Document]) -> Optional[PineconeVectorStore]:
    """
    Initialize Pinecone vector database and create new index for the embeddings of transcript.
    Uses Google embeddings and converts vector database into retriever for RAG.
    
    Args:
        index_name (str): Name of the Pinecone index
        chunks (List[Document]): List of document chunks to store
        
    Returns:
        Optional[PineconeVectorStore]: Configured retriever or None if initialization fails
    """
    try:
        logger.info(f"Initializing Pinecone retriever with index: {index_name}")
        
        # Get API keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not pinecone_api_key or not google_api_key:
            raise ValueError("Missing required API keys in environment variables")
            
        # Initialize Pinecone
        logger.debug("Connecting to Pinecone")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check existing indexes
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        logger.debug(f"Found existing indexes: {existing_indexes}")
        
        # Create new index if needed
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                logger.info("Waiting for index to be ready...")
                while not pc.describe_index(index_name).status["ready"]:
                    time.sleep(1)
                logger.info("Index is ready")
                
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {index_name}")
                logger.error(f"Error details: {str(e)}")
                raise
        
        # Initialize index
        try:
            logger.debug("Connecting to index")
            index = pc.Index(index_name)
        except Exception as e:
            logger.error("Failed to connect to Pinecone index")
            logger.error(f"Error details: {str(e)}")
            raise
            
        # Initialize embeddings
        try:
            logger.debug("Initializing Google embeddings")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
        except Exception as e:
            logger.error("Failed to initialize Google embeddings")
            logger.error(f"Error details: {str(e)}")
            raise
            
        # Create vector store
        try:
            logger.debug("Creating vector store")
            vector_store = PineconeVectorStore(index=index, embedding=embeddings)
            
            # Generate UUIDs for documents
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            
            # Insert documents
            logger.info(f"Adding {len(chunks)} documents to vector store")
            vector_store.add_documents(documents=chunks, ids=uuids)
            
            # Create retriever
            logger.debug("Configuring retriever")
            pinecone_retriever = vector_store.as_retriever(search_kwargs={
                "k": 10,
            })
            
            logger.info("Successfully initialized Pinecone retriever")
            return pinecone_retriever
            
        except Exception as e:
            logger.error("Failed to create vector store or add documents")
            logger.error(f"Error details: {str(e)}")
            raise
            
    except Exception as e:
        logger.error("Fatal error in Pinecone retriever initialization")
        logger.error(f"Error details: {str(e)}")
        return None