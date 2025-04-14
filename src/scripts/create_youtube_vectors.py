import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag.embeddings import YoutubeEmbeddingManager

def main():
    # 경로 설정
    csv_path = "data/cleaned_youtube_text.csv"
    vector_store_path = "vectors/youtube_vectors"
    
    # 임베딩 클래스 인스턴스화
    embeddings = YoutubeEmbeddingManager(model_name="bge-m3")
    
    # 유튜브 데이터 로드
    print(f"유튜브 데이터 로드 중: {csv_path}")
    documents = embeddings.load_youtube_data(csv_path)
    print(f"{len(documents)}개의 문서가 생성되었습니다.")
    
    # 벡터 저장소 생성
    print("임베딩 생성 중...")
    vector_store = embeddings.create_embeddings(documents)
    
    # 벡터 저장소 저장
    print(f"벡터 저장소 저장 중: {vector_store_path}")
    embeddings.save_embeddings(vector_store, vector_store_path)
    
    # 임베딩 정보 출력
    if hasattr(vector_store, 'index') and hasattr(vector_store.index, 'd'):
        print(f"임베딩 차원: {vector_store.index.d}")
    print(f"문서 수: {len(vector_store.index_to_docstore_id)}")
    
    # 테스트 검색
    print("\n테스트 검색 수행...")
    results = embeddings.similarity_search("우울증의 증상은 무엇인가요?", k=2)
    
    print("\n검색 결과:")
    for i, doc in enumerate(results):
        print(f"\n{i+1}. 제목: {doc.metadata.get('title', '제목 없음')}")
        print(f"내용 일부: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main() 