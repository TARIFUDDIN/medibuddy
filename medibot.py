import streamlit as st
import os
import traceback
import asyncio

# Fix event loop issues
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    pass

# Force CPU-only for PyTorch before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CUDA"] = "0"

# Rest of your code remains the same...
# Try to use FAISS CPU version
try:
    # Attempt to unload any GPU FAISS if it's loaded
    import sys
    if 'faiss' in sys.modules:
        del sys.modules['faiss']
    
    # Try to import CPU version specifically
    import faiss
except:
    pass

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Vector store path
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache the vector store to avoid reloading it on each interaction
@st.cache_resource
def get_vectorstore():
    try:
        print("Initializing embeddings with explicit CPU settings...")
        # Force CPU explicitly
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
        )
        print("Embeddings initialized successfully")
        
        # Check if vector store exists
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"Vector store path does not exist: {DB_FAISS_PATH}")
            print(f"Vector store path does not exist: {DB_FAISS_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
            return None
            
        print(f"Loading vector store from: {DB_FAISS_PATH}")
        
        # Try to load with CPU-only settings
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully")
        return db
    except Exception as e:
        error_msg = f"Failed to load vector store: {str(e)}"
        st.error(error_msg)
        print(error_msg)
        print(traceback.format_exc())  # More detailed error
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id):
    # Get token from Streamlit secrets
    hf_token = st.secrets.get("HF_TOKEN")
    
    if not hf_token:
        st.error("HuggingFace API token not found in secrets. Please configure it in your .streamlit/secrets.toml file.")
        return None
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            token=hf_token,
            temperature=0.5,
            max_length=512,
            model_kwargs={}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize language model: {str(e)}")
        return None

# Format source documents for better readability
def format_source_documents(source_docs):
    formatted_text = "**Source Documents:**\n\n"
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        # Truncate long content
        content = doc.page_content
        if len(content) > 200:
            content = content[:200] + "..."
            
        formatted_text += f"**Document {i+1}**\n"
        formatted_text += f"- **Source:** {source}\n"
        formatted_text += f"- **Page:** {page}\n"
        formatted_text += f"- **Content:** {content}\n\n"
        
    return formatted_text

def main():
    st.set_page_config(
        page_title="MediBot - Medical Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 MediBot - Your Medical Assistant")
    st.markdown("Ask any medical questions and get answers from reliable sources")
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Add the model selection option
    with st.sidebar:
        st.header("Model Settings")
        model_option = st.selectbox(
            "Select AI Model:",
            ["mistralai/Mistral-7B-Instruct-v0.3", "google/flan-t5-large"],
            index=0
        )
        
        st.header("About")
        st.markdown("MediBot uses AI to provide medical information from reliable sources.")
        st.markdown("This is for informational purposes only and not a substitute for professional medical advice.")
        
        # Add debug info
        if st.checkbox("Show Debug Info"):
            st.write(f"Current directory: {os.getcwd()}")
            st.write(f"Vector store path exists: {os.path.exists(DB_FAISS_PATH)}")
            try:
                import torch
                st.write(f"PyTorch CUDA available: {torch.cuda.is_available()}")
                st.write(f"PyTorch version: {torch.__version__}")
            except:
                st.write("PyTorch not available")
    
    # Chat input
    prompt = st.chat_input("Ask a medical question...")
    
    if prompt:
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to session state
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical expert. Use the provided context to answer the user's question accurately and concisely.
        If the context does not contain enough information, say "I don't know."
        Do not provide any information outside of the medical context provided.

        Context: {context}
        Question: {question}

        Answer:
        """
        
        # Get selected model
        HUGGINGFACE_REPO_ID = model_option
        
        # Processing indicator
        with st.chat_message("assistant"):
            with st.status("Searching medical knowledge base...", expanded=True) as status:
                try:
                    # Get vector store
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the vector store. Please check your database path.")
                        return
                    
                    # Initialize language model
                    llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID)
                    if llm is None:
                        return
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    
                    # Get response
                    status.update(label="Generating response...")
                    response = qa_chain.invoke({'query': prompt})
                    
                    # Format result
                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    # Format sources in a nice way
                    formatted_sources = format_source_documents(source_documents)
                    
                    # Complete response
                    full_response = f"{result}\n\n{formatted_sources}"
                    
                    # Update status
                    status.update(label="Done!", state="complete")
                    
                    # Display response
                    st.markdown(full_response)
                    
                    # Add to session state
                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
                    
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
                    st.info("Please check your HuggingFace API token in the secrets.toml file.")
                    print(traceback.format_exc())  # More detailed error for debugging

if __name__ == "__main__":
    main()