# PubMed RAG 챗봇 with LangGraph

이 프로젝트는 LangGraph를 활용한 Retrieval-Augmented Generation(RAG) 응용 프로그램으로, Chainlit을 사용하여 PubMed 연구 논문과 상호작용할 수 있도록 설계되었습니다. 이 애플리케이션은 임베딩과 벡터 저장소를 활용하여 논문 내용을 기반으로 사용자 쿼리에 지능적인 응답을 제공합니다.

## 개요

PubMed RAG 챗봇은 사용자가 PubMed 논문과 관련된 질문을 할 수 있게 하며, 논문의 제목과 초록에서 생성된 임베딩의 벡터 저장소에서 관련 정보를 검색합니다. 이 애플리케이션은 응답 생성을 위해 Ollama 언어 모델을 활용합니다.

## LangGraph 구현

이 프로젝트는 LangGraph를 사용하여 대화 워크플로우를 구현했습니다. LangGraph는 LangChain의 확장으로, 상태 기반 그래프를 통해 복잡한 대화 흐름을 구성할 수 있게 해줍니다. 주요 구성 요소는 다음과 같습니다:

1. **워크플로우 노드**:
   - 질문 분류 (Classify): 사용자 질문이 학술적인지 상담 관련인지 분류
   - 문서 검색 (Retrieve): 관련 PubMed 논문 검색
   - 학술 응답 생성 (Generate Academic): 학술 질문에 대한 응답 생성
   - 상담 응답 생성 (Generate Counseling): 상담 질문에 대한 응답 생성

2. **조건부 라우팅**:
   - 질문 유형에 따라 워크플로우가 다른 경로로 분기
   - 학술 질문: 문서 검색 → 학술 응답 생성
   - 상담 질문: 직접 상담 응답 생성 (문서 검색 없음)

3. **스트리밍 처리**:
   - 응답이 생성되는 동안 토큰 단위로 실시간 스트리밍

## 프로젝트 구조

```
pubmed-rag-chainlit
├── app.py                  # Chainlit 애플리케이션 진입점
├── src                     # 소스 코드
│   ├── rag                 # RAG 모듈
│   │   ├── __init__.py     # RAG 패키지 초기화
│   │   ├── embeddings.py   # 임베딩 로직
│   │   ├── vectorstore.py  # 벡터 저장소 관리
│   │   └── prompts.py      # 프롬프트 템플릿
│   ├── models              # 모델 모듈
│   │   ├── __init__.py     # 모델 패키지 초기화
│   │   ├── llm.py          # LLM 로직
│   │   └── workflow.py     # LangGraph 워크플로우 관리
│   ├── visualization       # 시각화 모듈
│   │   ├── __init__.py     # 시각화 패키지 초기화
│   │   └── graph_visualizer.py # 워크플로우 시각화
│   └── utils               # 유틸리티 함수
│       ├── __init__.py     # 유틸리티 패키지 초기화
│       └── data_loader.py  # 데이터 로딩 및 처리
├── data                    # 데이터 파일 디렉토리
│   └── cleaned_pubmed_papers.csv # 정제된 PubMed 논문 데이터
├── vectors                 # 벡터 파일 디렉토리
│   └── pubmed_vectors      # PubMed 논문 벡터
├── visualization           # 시각화 결과물 디렉토리
│   └── simple_langgraph_workflow.png # 워크플로우 다이어그램
├── chainlit.md            # Chainlit 애플리케이션 문서
├── .env.example            # 환경 변수 예제
├── requirements.txt        # 프로젝트 의존성
├── config.json             # 설정
└── README.md               # 프로젝트 문서
```

## 설치

1. 저장소 복제:
   ```
   git clone <repository-url>
   cd pubmed-rag-chainlit
   ```

2. 필요한 의존성 설치:
   ```
   pip install -r requirements.txt
   ```

3. `.env.example`을 `.env`로 복사하고 필요에 따라 값을 업데이트합니다.

4. Ollama 설치 및 실행:
   ```
   # Ollama 설치 - https://ollama.ai/download
   # 필요한 모델 가져오기
   ollama pull gemma3:4b
   ollama pull bge-m3
   ```

## 사용법

애플리케이션을 실행하려면 다음 명령을 실행하세요:
```
chainlit run app.py
```

애플리케이션이 실행되면 웹 브라우저에서 접근하여 PubMed 논문과 관련된 질문을 하거나 우울증 관련 상담을 요청할 수 있습니다.

## 기여

기여는 환영합니다! 개선이나 버그 수정을 위해 이슈를 열거나 풀 리퀘스트를 제출해 주세요.

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.