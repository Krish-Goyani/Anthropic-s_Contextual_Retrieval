# Anthropic's Contextual RAG Implementation 🤖

An implementation of Anthropic's Contextual Retrieval Augmented Generation (RAG) system, designed to provide intelligent responses from PDF documents using advanced context-aware retrieval methods. It combines contextual understanding + traditional semantic embeddings + BM25 search algorithm + rank fusion + reranking algorithm to provide the best possible answer.

## 🌟 Features

- PDF document processing and analysis
- Context-aware document chunking
- Ensembled retrieval using both vector similarity and BM25
- Reranking of retrieved documents
- Interactive web interface

## 🏗️ Architecture

The system follows Anthropic's Contextual Retrieval preprocessing approach:

1. **Document Processing**
   - PDF ingestion and text extraction
   - Document chunking with overlap
    
2. **Contextual Preprocessing**
   - Each chunk is processed to understand its context within the document
   - Gemini generates 50-100 tokens of context for each chunk
   - Context is prepended to corresponding chunks

3. **Dual Retrieval System**
   - Vector embeddings for semantic search using Gemini embeddings
   - BM25 indexing for keyword-based retrieval
   - Combined retrieval with rank fusion 
     
4. **Re-ranking**
   - Retrieved documents are re-ranked using Cohere's ranking model
   - Relevance boosting
   - Duplicate removal

## 🚀 Getting Started

### Environment Setup

1. **Create a Python Virtual Environment**
```bash
# Using venv
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables**
Create a `.env` file in the root directory:
```env
# API Keys
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key

```

### Running the Application

```bash
streamlit run app.py
```

## 📦 Dependencies

Key packages required:
```text
langchain_community
pypdf
langchain_google_genai
langchain-pinecone
langchain-core
pinecone-client
rank_bm25
langchain-cohere
streamlit
```

## 💻 Usage

1. Launch the application
2. Upload your PDF document using the file uploader
3. Wait for the preprocessing to complete
   - Document chunking
   - Context generation
   - Index building
4. Enter your questions in the text input
5. Receive contextually-aware answers based on your document

## 🔧 Technical Implementation

- **Frontend**: Streamlit
- **Backend**: Python
- **RAG Implementation**: Based on Anthropic's Contextual Retrieval architecture
- **Key Components**:
  - Document chunking with LangChain
  - Context generation using Gemini
  - Dual retrieval system (Gemini Embeddings + BM25)
  - Cohere re-ranking
  - Gemini response generation


## 🔗 References

- [Anthropic's RAG Documentation](https://www.anthropic.com/news/contextual-retrieval)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [Cohere Reranking](https://docs.cohere.com/docs/reranking)


## 👨‍💻 Author

Made with ❤️ by [Krish Goyani](https://github.com/Krish-Goyani)
