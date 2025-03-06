import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

os.environ["HF_HOME"] = "E:\\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:\\huggingface_cache"

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            cache_folder="E:\\huggingface_cache"  
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    if not hf_token:
        st.error("HuggingFace API token not found. Please set the HF_TOKEN environment variable.")
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
def format_source_documents(source_docs):
    formatted_text = "**Source Documents:**\n\n"
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
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
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        with st.sidebar:
            st.header("Configuration")
            hf_token = st.text_input("Enter HuggingFace API Token:", type="password")
            st.info("Your token will not be stored permanently.")
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
    prompt = st.chat_input("Ask a medical question...")
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical expert. Use the provided context to answer the user's question accurately and concisely.
        If the context does not contain enough information, say "I don't know."
        Do not provide any information outside of the medical context provided.

        Context: {context}
        Question: {question}

        Answer:
        """

        HUGGINGFACE_REPO_ID = model_option

        with st.chat_message("assistant"):
            with st.status("Searching medical knowledge base...", expanded=True) as status:
                try:
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the vector store. Please check your database path.")
                        return

                    llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=hf_token)
                    if llm is None:
                        return
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )
                    status.update(label="Generating response...")
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    source_documents = response["source_documents"]
                    formatted_sources = format_source_documents(source_documents)
                    full_response = f"{result}\n\n{formatted_sources}"
                    
                    # Update status
                    status.update(label="Done!", state="complete")
                    
                    # Display response
                    st.markdown(full_response)
                    
                    # Add to session state
                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})
                    
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
                    st.info("Please check your HuggingFace API token and internet connection.")

if __name__ == "__main__":
    main()