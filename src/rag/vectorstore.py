import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.utils.data_loader import prepare_documents

def get_vector_store(df):
    # Get the path from environment variable or use default
    vector_store_path = os.environ.get("VECTOR_STORE_PATH", "pubmed_vectors")
    
    if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path):
        try:
            embeddings = OllamaEmbeddings(
                model="bge-m3",
                base_url="http://localhost:11434"
            )
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            doc_count = len(vector_store.index_to_docstore_id)
            if doc_count == len(df):
                return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            pass

    # Create documents from DataFrame
    documents = prepare_documents(df)

    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    vector_store.save_local(vector_store_path)
    
    return vector_store