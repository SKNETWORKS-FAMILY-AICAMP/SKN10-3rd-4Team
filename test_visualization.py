import os
import sys

# src 디렉토리를 시스템 경로에 추가
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models.workflow import WorkflowManager
from src.rag.vectorstore import get_vector_store
from src.models.llm import LLMManager
from src.utils.data_loader import load_data
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def main():
    print("LangGraph 워크플로우 시각화 테스트 시작...")
    
    # 데이터 로딩
    print("데이터 로딩 중...")
    df = load_data('data/cleaned_pubmed_papers.csv')
    print(f"CSV 파일에서 {len(df)}개의 논문 로드됨")
    
    # 벡터 저장소 설정
    print("벡터 저장소 준비 중...")
    vector_store = get_vector_store(df)
    print(f"벡터 저장소 준비 완료 (문서 {len(vector_store.index_to_docstore_id)}개)")
    
    # LLM 초기화
    print("AI 모델 초기화 중...")
    llm_manager = LLMManager(
        model_name="gemma3:4b",
        base_url="http://localhost:11434",
        streaming=True
    )
    print("AI 모델 초기화 완료")
    
    # 워크플로우 매니저 초기화
    print("워크플로우 초기화 중...")
    workflow_manager = WorkflowManager(llm_manager, vector_store)
    print("워크플로우 초기화 완료")
    
    # LangGraph 내장 시각화 실행
    print("LangGraph 내장 시각화 기능 실행 중...")
    path = workflow_manager.visualize_workflow()
    
    if path:
        print(f"워크플로우 이미지 저장됨: {path}")
    else:
        print("워크플로우 시각화 실패")

if __name__ == "__main__":
    main() 