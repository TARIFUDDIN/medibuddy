import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pypdf
import torch

# Force CPU usage for torch
torch.set_default_device("cpu")

# Step 1: Load raw PDF(s)
DATA_PATH = "Data/"

def load_pdf_files(data_path):
    documents = []
    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(data_path, pdf_file)
        try:
            loader = PyPDFLoader(file_path)
            file_docs = loader.load()
            documents.extend(file_docs)
            print(f"Successfully loaded {file_path} with PyPDFLoader")
        except Exception as e:
            print(f"Error with PyPDFLoader: {e}")
            try:
                reader = pypdf.PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={"source": file_path, "page": i}
                            )
                            documents.append(doc)
                    except Exception as page_error:
                        print(f"  Error extracting text from page {i}: {page_error}")
                print(f"Successfully loaded {file_path} with direct pypdf")
            except Exception as direct_error:
                print(f"  Failed with direct pypdf approach: {direct_error}")
                print(f"  Skipping file {file_path}")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents

def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Split into {len(text_chunks)} chunks")
    return text_chunks

def get_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
        )
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Make sure you have the sentence-transformers package installed.")
        print("Try running: pip install sentence-transformers")
        raise

if __name__ == "__main__":
    print("Step 1: Loading PDF documents...")
    documents = load_pdf_files(data_path=DATA_PATH)
    print(f"Length of PDF pages: {len(documents)}")
    
    if len(documents) == 0:
        print("No documents were loaded. Please check your data directory.")
        exit()
    
    print("\nStep 2: Creating text chunks...")
    text_chunks = create_chunks(extracted_data=documents)
    print(f"Length of Text Chunks: {len(text_chunks)}")
    
    print("\nStep 3: Loading embedding model...")
    try:
        embedding_model = get_embedding_model()
        
        print("\nStep 4: Creating and storing FAISS vector database...")
        DB_FAISS_PATH = "vectorstore/db_faiss"
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"Vector database successfully saved to {DB_FAISS_PATH}")
        
        print("\nTesting retrieval with a sample query...")
        query = "What is diabetes?"
        docs = db.similarity_search(query, k=3)
        print(f"Top document for query '{query}':")
        print(f"Source: {docs[0].metadata['source']}, Page: {docs[0].metadata['page']}")
        print(f"Content snippet: {docs[0].page_content[:150]}...\n")
        
    except Exception as e:
        print(f"Error in embedding or database creation: {e}")