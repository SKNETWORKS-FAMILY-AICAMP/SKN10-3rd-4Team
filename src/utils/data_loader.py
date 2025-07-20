import os
import pandas as pd
from langchain.schema import Document

def load_data(csv_path):
    """Load CSV data into a DataFrame."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers from the CSV file.")
    return df

def prepare_documents(df):
    """Prepare documents for the vector store from the DataFrame."""
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
    return documents

def save_documents(documents, save_path):
    """Save processed documents to a specified path."""
    # This function can be implemented to save documents if needed
    pass