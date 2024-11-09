from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from typing import List
from src.driver import ensemble_retriever
from src.re_ranker import rerank_documents
from dotenv import load_dotenv
from RAG_Logger import logger
load_dotenv()


      


def get_rag_chain(
    question_template: str = """Answer the question based on the following context:
    Context: {context}
    Question: {question}
    Answer: """
  ):

  try:
    logger.info("Starting RAG chain")

    # Create prompt template
    prompt = PromptTemplate(
        template=question_template,
        input_variables=["context", "question"]
    )

    # Configure LLM
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")
            
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)
        logger.debug("Successfully initialized LLM")
    except Exception as e:
        logger.error("Failed to initialize LLM")
        logger.error(f"Error details: {str(e)}")
        raise
    
    # Define the pipeline
    rag_chain = (
        {"context": lambda x: retrieve_and_rerank({"question": x}), 
            "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("Successfully initialized RAG chain")
    return rag_chain
    
  except Exception as e:
    logger.error("Fatal error in RAG chain initialization")
    logger.error(f"Error details: {str(e)}")
    raise

