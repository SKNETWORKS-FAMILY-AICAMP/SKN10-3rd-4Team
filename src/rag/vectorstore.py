import os
import pandas as pd
from langchain.schema import Document
from src.utils.data_loader import prepare_documents
from src.rag.embeddings import BaseEmbeddingManager

def get_vector_store(df):
    """
    데이터프레임으로부터 벡터 저장소를 생성하거나 로드합니다.
    
    Args:
        df (pd.DataFrame): 문서 데이터프레임
        
    Returns:
        FAISS: FAISS 벡터 저장소
    """
    # 벡터 저장소 경로 설정
    vector_store_path = os.environ.get("VECTOR_STORE_PATH", "pubmed_vectors")
    
    # 임베딩 매니저 생성
    embedding_manager = BaseEmbeddingManager(model_name="bge-m3")
    
    # 기존 벡터 저장소 존재 확인
    if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path):
        try:
            # 벡터 저장소 로드
            vector_store = embedding_manager.load_embeddings(vector_store_path)
            
            # 문서 수 확인
            doc_count = len(vector_store.index_to_docstore_id)
            if doc_count == len(df):
                print(f"기존 벡터 저장소 로드 완료 (문서 수: {doc_count})")
                return vector_store
            else:
                print(f"문서 수 불일치: 벡터 저장소({doc_count}) != 데이터프레임({len(df)})")
        except Exception as e:
            print(f"벡터 저장소 로드 중 오류 발생: {e}")

    print("새로운 벡터 저장소 생성 중...")
    
    # 문서 준비
    documents = prepare_documents(df)
    
    # 벡터 저장소 생성
    vector_store = embedding_manager.create_embeddings(documents)
    
    # 벡터 저장소 저장
    embedding_manager.save_embeddings(vector_store, vector_store_path)
    
    return vector_store