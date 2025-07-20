import os
import sys
from pathlib import Path
import argparse

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag.embeddings import get_youtube_vector_store, YoutubeEmbeddingManager

def main():
    """
    유튜브 데이터를 임베딩하여 벡터 저장소를 생성하는 스크립트
    """
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='유튜브 데이터 임베딩 생성')
    parser.add_argument('--csv', type=str, default="data\cleaned_youtube_data\cleaned_youtube_text.csv",
                        help='유튜브 데이터 CSV 파일 경로')
    parser.add_argument('--output', type=str, default="vectors/youtube_vectors",
                        help='벡터 저장소 출력 경로')
    parser.add_argument('--model', type=str, default="bge-m3",
                        help='임베딩 모델 이름')
    parser.add_argument('--force', action='store_true',
                        help='기존 벡터 저장소가 있어도 강제로 다시 생성')
    parser.add_argument('--query', type=str, default="우울증의 주요 증상은 무엇인가요?",
                        help='테스트 쿼리 (검색 테스트용)')
    parser.add_argument('--k', type=int, default=2,
                        help='검색 결과 수')
    
    args = parser.parse_args()
    
    try:
        # 벡터 저장소 생성 또는 로드
        vector_store = get_youtube_vector_store(
            csv_path=args.csv,
            vector_store_path=args.output,
            model_name=args.model,
            force_recreate=args.force
        )
        
        # 테스트 쿼리 실행
        if args.query:
            print(f"\n테스트 쿼리 실행: '{args.query}'")
            # vector_store.embeddings 속성이 없으므로 유튜브 임베딩 객체를 새로 생성
            from src.rag.embeddings import YoutubeEmbeddingManager
            embeddings = YoutubeEmbeddingManager(model_name=args.model)
            embeddings.vector_store = vector_store
            results = embeddings.similarity_search(args.query, k=args.k)
            
            # 결과 출력
            print("\n검색 결과:")
            for i, doc in enumerate(results):
                print(f"\n{i+1}. 제목: {doc.metadata['title']}")
                print(f"내용: {doc.page_content[:100]}...")
                if 'video_url' in doc.metadata and doc.metadata['video_url']:
                    print(f"URL: {doc.metadata['video_url']}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 