import warnings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import torch  

# Force CPU usage for torch
torch.set_default_device("cpu") 

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Default model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Rest of the code remains the same...

def load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID):
    """Load a model using HuggingFace Endpoint API."""
    try:
        # Get token from Streamlit secrets
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        
        if not HF_TOKEN:
            print("HuggingFace API token not found in secrets.toml.")
            return None
            
        print(f"Loading model from HuggingFace Endpoint: {huggingface_repo_id}")
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            token=HF_TOKEN,
            temperature=0.5,
            max_length=512,
            model_kwargs={}
        )
        return llm
    except Exception as e:
        print(f"Error loading HuggingFace Endpoint: {e}")
        print("Please ensure you have set the HF_TOKEN in .streamlit/secrets.toml.")
        raise

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
You are a medical expert. Use the provided context to answer the user's question accurately and concisely.
If the context does not contain enough information, say "I don't know."
Do not provide any information outside of the medical context provided.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, 
                           input_variables=["context", "question"])
    return prompt

# Function to create the QA chain
def create_qa_chain():
    try:
        # Load Database
        DB_FAISS_PATH = "vectorstore/db_faiss"
        
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("Loading vector database...")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        print("Creating QA chain...")
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        raise

# Function to format source documents for better readability
def format_source_documents(source_docs):
    formatted_docs = []
    for i, doc in enumerate(source_docs):
        formatted_doc = {
            "index": i + 1,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        }
        formatted_docs.append(formatted_doc)
    return formatted_docs

# Main execution
if __name__ == "__main__":
    print("Setting up Enhanced Medical QA system...")
    
    try:
        qa_chain = create_qa_chain()
        
        # Interactive query loop
        while True:
            user_query = input("\nWrite Query Here (or type 'exit' to quit): ")
            
            if user_query.lower() == 'exit':
                break
                
            print("\nSearching medical knowledge base...")
            
            try:
                response = qa_chain.invoke({'query': user_query})
                
                print("\nRESULT: ", response["result"])
                print("\nSOURCE DOCUMENTS:")
                
                formatted_docs = format_source_documents(response["source_documents"])
                for doc in formatted_docs:
                    print(f"\nDocument {doc['index']}:")
                    print(f"Source: {doc['source']}, Page: {doc['page']}")
                    print(f"Content snippet: {doc['content_preview']}")
                    
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Try a different question or check your API token and connection.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Please ensure:")
        print("1. Your HF_TOKEN is set in .streamlit/secrets.toml")
        print("2. Your vectorstore/db_faiss directory exists")
        print("3. You have internet connectivity to HuggingFace")