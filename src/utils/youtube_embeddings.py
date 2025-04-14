import os
import pandas as pd
from typing import List, Dict, Any
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class YoutubeEmbeddings:
    def __init__(self, model_name: str = "llama2"):
        """
        유튜브 데이터 임베딩을 위한 클래스 초기화
        
        Args:
            model_name (str): Ollama 모델 이름. 기본값은 'llama2'
        """
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def load_youtube_data(self, data_dir: str) -> List[Document]:
        """
        CSV 파일에서 유튜브 데이터를 로드합니다.
        
        Args:
            data_dir (str): CSV 파일이 있는 디렉토리 경로
            
        Returns:
            List[Document]: 문서 객체 리스트
        """
        documents = []
        for file in os.listdir(data_dir):
            if file.endswith('.csv') and file.startswith('cleaned_'):
                df = pd.read_csv(os.path.join(data_dir, file))
                for _, row in df.iterrows():
                    text = f"제목: {row['title']}\n내용: {row['caption']}"
                    chunks = self.text_splitter.split_text(text)
                    for chunk in chunks:
                        metadata = {
                            "title": row['title'],
                            "video_url": row['video_url']
                        }
                        documents.append(Document(page_content=chunk, metadata=metadata))
        return documents
        
    def create_embeddings(self, documents: List[Document]) -> FAISS:
        """
        문서로부터 임베딩을 생성하고 FAISS 벡터 저장소를 반환합니다.
        
        Args:
            documents (List[Document]): 문서 객체 리스트
            
        Returns:
            FAISS: 생성된 벡터 저장소
        """
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store
    
    def save_embeddings(self, vector_store: FAISS, path: str = "data/vector_store") -> None:
        """
        FAISS 벡터 저장소를 저장합니다.
        
        Args:
            vector_store (FAISS): 저장할 벡터 저장소
            path (str): 저장 경로
        """
        vector_store.save_local(path)
        
    def load_embeddings(self, path: str = "data/vector_store") -> FAISS:
        """
        저장된 FAISS 벡터 저장소를 불러옵니다.
        
        Args:
            path (str): 불러올 경로
            
        Returns:
            FAISS: 불러온 벡터 저장소
        """
        return FAISS.load_local(path, self.embeddings) 