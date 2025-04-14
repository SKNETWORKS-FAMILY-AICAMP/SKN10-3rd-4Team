import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.youtube_embeddings import YoutubeEmbeddings

def main():
    # 경로 설정
    csv_path = "data/cleaned_youtube_text.csv"
    vector_store_path = "vectors/youtube_vectors"
    
    # 임베딩 객체 생성
    youtube_embeddings = YoutubeEmbeddings(model_name="bge-m3")
    
    try:
        # CSV 파일 로드 및 문서 생성
        print("유튜브 데이터를 로드하고 문서를 생성합니다...")
        documents = youtube_embeddings.load_youtube_data(csv_path)
        print(f"총 {len(documents)}개의 문서가 생성되었습니다.")
        
        # 벡터 저장소 생성
        print("\n벡터 저장소를 생성합니다...")
        vector_store = youtube_embeddings.create_embeddings(documents)
        
        # 벡터 저장소 저장
        print("\n벡터 저장소를 저장합니다...")
        youtube_embeddings.save_embeddings(vector_store, vector_store_path)
        print(f"벡터 저장소가 {vector_store_path}에 저장되었습니다.")
        
        # 테스트 쿼리 실행
        print("\n테스트 쿼리를 실행합니다...")
        loaded_store = youtube_embeddings.load_embeddings(vector_store_path)
        results = loaded_store.similarity_search("우울증의 주요 증상은 무엇인가요?", k=2)
        
        print("\n검색 결과:")
        for doc in results:
            print(f"\n제목: {doc.metadata['title']}")
            print(f"내용: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 