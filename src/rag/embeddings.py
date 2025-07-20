import os
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import traceback
import numpy as np

class BaseEmbeddingManager:
    """
    임베딩 관리를 위한 기본 클래스
    """
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        """
        임베딩 관리자 초기화
        
        Args:
            model_name (str): Ollama 모델 이름. 기본값은 'bge-m3'
            base_url (str): Ollama 서버 URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            model_kwargs={"device": "cpu"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        
    def create_embeddings(self, documents: List[Document]) -> FAISS:
        """
        문서로부터 임베딩을 생성하고 FAISS 벡터 저장소를 반환합니다.
        
        Args:
            documents (List[Document]): 문서 객체 리스트
            
        Returns:
            FAISS: 생성된 벡터 저장소
        """
        vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store = vector_store
        return vector_store
    
    def save_embeddings(self, vector_store: FAISS, path: str) -> None:
        """
        FAISS 벡터 저장소를 저장합니다.
        
        Args:
            vector_store (FAISS): 저장할 벡터 저장소
            path (str): 저장 경로
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vector_store.save_local(path)
        print(f"벡터 저장소가 {path}에 저장되었습니다.")
        
    def load_embeddings(self, path: str) -> FAISS:
        """
        저장된 FAISS 벡터 저장소를 불러옵니다.
        
        Args:
            path (str): 불러올 경로
            
        Returns:
            FAISS: 불러온 벡터 저장소
        """
        try:
            # 경로 존재 확인
            if not os.path.exists(path):
                raise FileNotFoundError(f"벡터 저장소 경로를 찾을 수 없습니다: {path}")
            
            # index.faiss 파일 확인
            faiss_path = os.path.join(path, "index.faiss")
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"index.faiss 파일을 찾을 수 없습니다: {faiss_path}")
            
            # index.pkl 파일 확인
            pkl_path = os.path.join(path, "index.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"index.pkl 파일을 찾을 수 없습니다: {pkl_path}")
            
            print(f"벡터 저장소 파일 확인 완료: {path}")
            
            try:
                vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS 벡터 저장소 로드 완료")
                
                # 벡터 저장소 상태 확인
                store_size = len(vector_store.index_to_docstore_id)
                print(f"벡터 저장소 크기: {store_size} 문서")
                
                # 벡터 차원 확인
                if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'd'):
                    print(f"벡터 차원: {vector_store.index.d}")
                
                self.vector_store = vector_store
                return vector_store
                
            except Exception as load_error:
                print(f"FAISS 로드 중 상세 오류:\n{traceback.format_exc()}")
                raise
            
        except Exception as e:
            print(f"벡터 저장소 로드 중 오류 발생: {str(e)}")
            print(f"상세 오류:\n{traceback.format_exc()}")
            raise
    
    def similarity_search(self, query: str, k: int = 2) -> List[Document]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 수
            
        Returns:
            List[Document]: 유사한 문서 리스트
        """
        try:
            if self.vector_store is None:
                raise ValueError("벡터 저장소가 로드되지 않았습니다.")
                
            print(f"쿼리 임베딩 생성 시도: {query[:100]}...")
            query_embedding = self.embeddings.embed_query(query)
            print(f"쿼리 임베딩 생성 완료 (차원: {len(query_embedding)})")
            
            # 벡터 차원 확인 및 출력
            if hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'd'):
                print(f"저장된 벡터 차원: {self.vector_store.index.d}")
                if len(query_embedding) != self.vector_store.index.d:
                    raise ValueError(f"임베딩 차원 불일치: 쿼리({len(query_embedding)}) != 저장소({self.vector_store.index.d})")
            
            print(f"유사도 검색 시작 (k={k})")
            results = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
            print(f"검색 결과: {len(results)}개 문서 찾음")
            
            return results
            
        except Exception as e:
            print(f"유사도 검색 중 오류 발생: {str(e)}")
            print(f"상세 오류:\n{traceback.format_exc()}")
            raise

class AcademicEmbeddingManager(BaseEmbeddingManager):
    """
    학술 데이터 임베딩을 위한 클래스
    """
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        super().__init__(model_name, base_url)
        
    def load_data_from_dataframe(self, df: pd.DataFrame) -> List[Document]:
        """
        데이터프레임에서 문서를 로드합니다.
        
        Args:
            df (pd.DataFrame): 데이터프레임
            
        Returns:
            List[Document]: 문서 객체 리스트
        """
        documents = []
        for _, row in df.iterrows():
            # PubMed 데이터용 형식
            text = f"제목: {row['title']}\n초록: {row['abstract']}"
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                metadata = {
                    "title": row['title'],
                    "pmid": row.get('pmid', ''),
                    "doi": row.get('doi', ''),
                    "url": row.get('url', '')
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
        return documents

class YoutubeEmbeddingManager(BaseEmbeddingManager):
    """
    유튜브 데이터 임베딩을 위한 클래스
    """
    def __init__(self, model_name: str = "bge-m3", base_url: str = "http://localhost:11434"):
        """
        유튜브 데이터 임베딩을 위한 클래스 초기화
        
        Args:
            model_name (str): Ollama 모델 이름. 기본값은 'bge-m3'
            base_url (str): Ollama 서버 URL
        """
        super().__init__(model_name, base_url)
        
    def load_youtube_data(self, csv_path: str) -> List[Document]:
        """
        CSV 파일에서 유튜브 데이터를 로드합니다.
        
        Args:
            csv_path (str): CSV 파일 경로
            
        Returns:
            List[Document]: 문서 객체 리스트
        """
        documents = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # caption 또는 summary 열 사용
            content_column = "caption" if "caption" in row else "summary"
            text = f"제목: {row['title']}\n내용: {row.get(content_column, '')}"
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                metadata = {
                    "title": row['title'],
                    "video_url": row.get('video_url', '')
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
        return documents

def get_youtube_vector_store(csv_path: str, vector_store_path: str = "vectors/youtube_vectors", 
                            model_name: str = "bge-m3", force_recreate: bool = False) -> FAISS:
    """
    유튜브 데이터에서 벡터 저장소를 생성하거나 로드합니다.
    
    Args:
        csv_path (str): 유튜브 데이터 CSV 파일 경로
        vector_store_path (str): 벡터 저장소 저장 경로
        model_name (str): 임베딩 모델 이름
        force_recreate (bool): 기존 벡터 저장소가 있어도 강제로 다시 생성할지 여부
        
    Returns:
        FAISS: FAISS 벡터 저장소
    """
    # 임베딩 객체 생성
    youtube_embeddings = YoutubeEmbeddingManager(model_name=model_name)
    
    # 기존 벡터 저장소 확인
    if not force_recreate and os.path.exists(vector_store_path) and os.path.isdir(vector_store_path):
        try:
            print(f"기존 벡터 저장소 로드 시도: {vector_store_path}")
            vector_store = youtube_embeddings.load_embeddings(vector_store_path)
            
            # 벡터 저장소가 정상적으로 로드되면 반환
            print(f"기존 벡터 저장소 로드 완료 (문서 수: {len(vector_store.index_to_docstore_id)})")
            return vector_store
            
        except Exception as e:
            print(f"기존 벡터 저장소 로드 실패: {str(e)}")
            print("새로운 벡터 저장소를 생성합니다...")
    
    # 기존 벡터 저장소 삭제 (force_recreate가 True인 경우)
    if force_recreate and os.path.exists(vector_store_path):
        print(f"기존 벡터 저장소 삭제: {vector_store_path}")
        for file in ['index.faiss', 'index.pkl']:
            file_path = os.path.join(vector_store_path, file)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # CSV 파일 로드 및 문서 생성
    print(f"유튜브 데이터 로드 중: {csv_path}")
    documents = youtube_embeddings.load_youtube_data(csv_path)
    print(f"총 {len(documents)}개의 문서가 생성되었습니다.")
    
    # 벡터 저장소 생성
    print("임베딩 생성 및 벡터 저장소 구축 중...")
    vector_store = youtube_embeddings.create_embeddings(documents)
    
    # 벡터 저장소 저장
    youtube_embeddings.save_embeddings(vector_store, vector_store_path)
    
    return vector_store