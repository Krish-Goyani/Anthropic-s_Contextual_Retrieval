from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from RAG_Logger import logger
from typing import List
import time
from dotenv import load_dotenv
import os
load_dotenv()

def enrich_chunks_with_context(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 200,
    model_name: str = "gemini-1.5-flash"
    ) -> List[Document]:
    """
    Processes documents by splitting them into chunks and adding AI-generated context summaries.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        model_name: Google Generative AI model to use

        
    Returns:
        List of Document objects with enriched content
    """
    
    try:
        logger.info(f"Starting chunk enrichment process with {len(documents)} documents")
        logger.debug(f"Parameters - chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, model: {model_name}")


        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize the LLM
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")


        llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=model_name
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            template=(
                "<document> {whole_document} </document> "

                "Here is the chunk we want to situate within the whole document "

                "<chunk> {chunk_content} </chunk> "

                "Please give a short succinct context to situate this chunk within "
                "the overall document for the purposes of improving search retrieval "
                "of the chunk. Answer only with the succinct context and nothing else."
            ),
            input_variables=["whole_document", "chunk_content"]
        )

        # initialize context chain
        context_chain = prompt_template | llm

        
        enriched_documents = []
        
        # Process each document
        for doc in documents:
            # Get the full document content
            full_content = doc.page_content
            
            # Split into chunks
            chunks = text_splitter.split_text(full_content)
            
            # Process each chunk
            for chunk in chunks:
                try:
                    # Generate context using the LLM

                    context = context_chain.invoke({
                        "whole_document": full_content,
                        "chunk_content": chunk
                    }).content.strip()
                    
                    
                    # Create new metadata
                    new_metadata = doc.metadata.copy()
                    new_metadata.update({
                        'generated_context': context,
                        'original_chunk': chunk,
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'search_source' : "dense_search"
                    })

                    # Combine context with original chunk
                    enriched_content = f"{context}\n\n{chunk}"
                    
                    # Create new document with enriched content
                    enriched_doc = Document(
                        page_content=enriched_content,
                        metadata=new_metadata
                    )
                    enriched_documents.append(enriched_doc)
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    # Add original chunk without enrichment if there's an error
                    enriched_documents.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'error': str(e),
                            'processing_status': 'failed'
                        }
                    ))

        logger.info(f"Chunk enrichment completed. Processed {len(enriched_documents)} chunks in total")
        return enriched_documents
    
  
        
    except Exception as e:
        logger.error("Fatal error in chunk enrichment process")
        logger.error(f"Error details: {str(e)}")
        raise
