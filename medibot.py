import os
# Force CPU-only mode before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Try to prevent PyTorch from looking for CUDA libraries
os.environ['USE_CUDA'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '0'

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# We need a CPU-only version of PyTorch
# Define a function that will safely import torch
def import_cpu_only_torch():
    import sys
    import importlib.util
    
    # Define a custom loader that will modify PyTorch's behavior
    class CPUOnlyLoader:
        def create_module(self, spec):
            return None
            
        def exec_module(self, module):
            # Import the module normally
            sys.modules.pop(spec.name, None)  # Remove our custom module
            module = importlib.import_module(spec.name)
            
            # Force CPU-only mode
            if hasattr(module, '_C'):
                # Monkey patch _C to prevent CUDA initialization
                if hasattr(module._C, '_cuda_isDriverSufficient'):
                    module._C._cuda_isDriverSufficient = lambda: False
                if hasattr(module._C, '_cuda_getDeviceCount'):
                    module._C._cuda_getDeviceCount = lambda: 0
            
            # Override CUDA availability functions
            module.cuda = lambda *args, **kwargs: False
            module.is_available = lambda: False
            module.cuda.is_available = lambda: False
            
            # Update the sys.modules with our modified module
            sys.modules[spec.name] = module
            return module
    
    # Load PyTorch without CUDA
    spec = importlib.util.find_spec('torch')
    loader = CPUOnlyLoader()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module

# Now import torch safely
try:
    torch = import_cpu_only_torch()
    # Set CPU as default device for extra safety
    torch.set_default_device("cpu")
except Exception as e:
    st.error(f"Failed to import PyTorch: {str(e)}")
    # If we can't import PyTorch, define a minimal placeholder
    # This will allow the app to at least start
    class TorchPlaceholder:
        def set_default_device(self, device):
            pass
    torch = TorchPlaceholder()
    torch.set_default_device("cpu")

# Vector store path
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache the vector store to avoid reloading it on each interaction
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
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

if __name__ == "__main__":
    main()