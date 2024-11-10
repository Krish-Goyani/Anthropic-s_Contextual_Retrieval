import streamlit as st
from src.driver import Driver
import os
from RAG_Logger import logger

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False

def process_uploaded_file(uploaded_file):
    """Process the uploaded PDF file"""
    try:
        logger.info(f"Processing uploaded file: {uploaded_file.name}")
        
        # Create a temporary directory for the PDF
        if not os.path.exists("local_database"):
            os.makedirs("local_database")
            
        # Save the uploaded file
        file_path = os.path.join("local_database", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        
        # Initialize the RAG chain
        driver_instance = Driver()
        st.session_state.rag_chain = driver_instance.get_rag_chain()
        st.session_state.pdf_loaded = True
        
        # Clean up
        os.remove(file_path)
        
        logger.info("Successfully processed PDF and initialized RAG chain")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False

def main():
    st.title("Anthropic's Contextual RAG 🤖")
    st.divider()
    st.subheader("One of the most advanced RAG systems in the world 🔥", divider="red")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload section
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and not st.session_state.pdf_loaded:
        with st.spinner("Processing PDF...⏳"):
            success = process_uploaded_file(uploaded_file)
            if success:
                st.success("PDF processed successfully! ✅ ")
            else:
                st.error("Error processing PDF. Please try again. ❌")
    
    # Question answering section
    if st.session_state.pdf_loaded:
        st.subheader("Ask a Question 💭")
        
        # Question input
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer 🔍"):
            if question:
                try:
                    with st.spinner("Generating answer..."):
                        logger.info(f"Processing question: {question}")
                        response = st.session_state.rag_chain.invoke({"question": question})
                        
                        # Display response
                        st.subheader("Answer 💡:")
                        st.markdown(response)
                        
                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    st.error("Error generating answer. Please try again.")
            else:
                st.warning("Please enter a question.⚠️")
    
    # Instructions
    with st.sidebar:
        st.subheader("Instructions")
        st.write("""
        1. Upload a PDF file using the file uploader 📄
        2. Wait for the PDF to be processed ⏳
        3. Enter your question in the text box ✍️
        4. Click 'Get Answer' to see the response 🔍
        """)
        
        # Add some information about the system
        st.subheader("About")
        st.write("""
        This system is demo of Anthropic's Contextual Retrieval Augmented Generation (RAG) to answer 
        questions about your PDF document. It processes the document and 
        uses advanced AI to generate accurate answers based on the content.
        for more information about this RAG, visit [Anthropic's RAG](https://www.anthropic.com/news/contextual-retrieval)
                 
        """)
        
        st.write("Made with ❤️ by [Krish Goyani](https://github.com/Krish-Goyani)")
    
    

if __name__ == "__main__":
    main()