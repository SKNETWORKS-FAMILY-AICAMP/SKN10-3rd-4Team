import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.youtube_embeddings import YoutubeEmbeddings

def main():
    # 데이터 디렉토리와 저장 경로 설정
    data_dir = "data/cleaned_youtube_data"
    save_dir = "data/vector_store"
    os.makedirs(save_dir, exist_ok=True)
    
    # 임베딩 객체 생성
    youtube_embedding = YoutubeEmbeddings(model_name="bge-m3")
    
    # 데이터 로드 및 문서 생성
    print("유튜브 데이터를 로드하고 문서를 생성합니다...")
    documents = youtube_embedding.load_youtube_data(data_dir)
    print(f"총 {len(documents)}개의 문서가 생성되었습니다.")
    
    # 벡터 저장소 생성
    print("벡터 저장소를 생성합니다...")
    vectorstore = youtube_embedding.create_embeddings(documents)
    
    # 벡터 저장소 저장
    save_path = os.path.join(save_dir, "youtube_depression")
    youtube_embedding.save_embeddings(vectorstore, save_path)
    print("벡터 저장소가 성공적으로 생성되었습니다.")
    
    # 테스트 쿼리 실행
    print("\n테스트 쿼리를 실행합니다...")
    query = "우울증의 주요 증상은 무엇인가요?"
    results = vectorstore.similarity_search(query, k=2)
    
    print("\n검색 결과:")
    for doc in results:
        print(f"\n제목: {doc.metadata['title']}")
        print(f"URL: {doc.metadata['video_url']}")
        print(f"내용: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main() 