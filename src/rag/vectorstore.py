# File: /pubmed-rag-chainlit/pubmed-rag-chainlit/src/rag/vectorstore.py

import os
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

VECTOR_STORE_PATH = "pubmed_vectors"
CSV_PATH = "cleaned_pubmed_papers.csv"

def get_vector_store(df):
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH):
        try:
            embeddings = OllamaEmbeddings(
                model="bge-m3",
                base_url="http://localhost:11434"
            )
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            doc_count = len(vector_store.index_to_docstore_id)
            if doc_count == len(df):
                return vector_store
        except Exception:
            pass

    documents = []
    for i, row in df.iterrows():
        content = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "paper_id": row["paper_id"],
                "paper_number": row["paper_number"],
                "journal": row["journal"],
                "title": row["title"]
            }
        )
        documents.append(doc)

    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store