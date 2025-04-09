from langchain_community.embeddings import OllamaEmbeddings

class EmbeddingManager:
    def __init__(self, model_name="bge-m3", base_url="http://localhost:11434"):
        self.embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)

    def create_embeddings(self, documents):
        return [self.embeddings.embed(doc.page_content) for doc in documents]

    def save_embeddings(self, embeddings, file_path):
        # Implement saving logic for embeddings
        pass

    def load_embeddings(self, file_path):
        # Implement loading logic for embeddings
        pass