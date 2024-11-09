from src.data_loader import load_pdf_documents
from src.chunk_enriching import enrich_chunks_with_context
from src.pinecon_retriever import get_pinecone_retriever
from src.BM25_retriever import get_BM25_retriever
from src.ensemble_retriever import get_ensemble_retriever
from src.get_RAG_chain import get_rag_chain
def driver():
    docs = load_pdf_documents(directory_path="local_database")

    enriched_docs = enrich_chunks_with_context(docs)

    pinecone_retriever = get_pinecone_retriever(index_name="contextual-embeddings", chunks=enriched_docs)

    bm25_retriever = get_BM25_retriever(docs=enriched_docs)

    ensemble_retriever = get_ensemble_retriever(pinecone_retriever, bm25_retriever)

    rag_chain = get_rag_chain()

    return rag_chain

    
