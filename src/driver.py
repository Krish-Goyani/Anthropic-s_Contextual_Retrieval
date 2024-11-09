from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.data_preprocessing.data_loader import load_pdf_documents
from src.data_preprocessing.chunk_enriching import enrich_chunks_with_context
from src.retriever.pinecon_retriever import get_pinecone_retriever
from src.retriever.BM25_retriever import get_BM25_retriever
from src.retriever.ensemble_retriever import get_ensemble_retriever
from src.Ranking.re_ranker import rerank_documents
from RAG_Logger import logger
import os   
from dotenv import load_dotenv

load_dotenv()

class Driver:
    def __init__(self):
        self.docs = load_pdf_documents(directory_path="local_database")

        self.enriched_docs = enrich_chunks_with_context(self.docs)

        self.pinecone_retriever = get_pinecone_retriever(index_name="contextual-embeddings", chunks=self.enriched_docs)

        self.bm25_retriever = get_BM25_retriever(docs=self.enriched_docs)

        self.ensemble_retriever = get_ensemble_retriever(self.pinecone_retriever, self.bm25_retriever)

    def retrieve_and_rerank(self,input_dict):
        try:
            logger.info("Starting document retrieval and reranking")
            # Extract question from input dict
            if isinstance(input_dict, dict):
                question = input_dict.get("question")
            else:
                question = input_dict  # If directly passed as string

            docs = self.ensemble_retriever.invoke(question)
            reranked_context = rerank_documents(docs, question)
            logger.info("Successfully completed retrieval and reranking")
            return reranked_context
            
        except Exception as e:
            logger.error("Fatal error in retrieve_and_rerank")
            logger.error(f"Error details: {str(e)}")
            raise


    def get_rag_chain(self):
        try:
            logger.info("Initializing RAG chain")
            
            # Create prompt template
            prompt = PromptTemplate(
                template="""Answer the question based on the following context only:
                Context: {context}
                Question: {question}
                Answer: """,
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
                {
                    "context": lambda x: self.retrieve_and_rerank(x["question"]),  # Pass just the question string
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            logger.info("Successfully initialized RAG chain")
            return rag_chain
        
        except Exception as e:
            logger.error("Fatal error in get_rag_chain")
            logger.error(f"Error details: {str(e)}")
            raise






    
