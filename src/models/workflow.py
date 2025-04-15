import os
from typing import Literal, Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from src.rag.embeddings import YoutubeEmbeddingManager
from pathlib import Path
import numpy as np
from tavily import TavilyClient
from langchain.docstore.document import Document

class WorkflowManager:
    """
    LangGraph를 사용한 대화 워크플로우를 관리하는 클래스
    """
    def __init__(self, llm_manager, vector_store):
        """
        워크플로우 관리자 초기화
        
        Args:
            llm_manager: LLM 관리자 객체
            vector_store: 벡터 스토어 객체
        """
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.graph = self._create_workflow()
        self.current_config = None  # 현재 실행 중인 config를 저장하기 위한 변수
        
        # 벡터 저장소 경로 설정
        self.project_root = Path(__file__).parent.parent.parent
        self.youtube_vector_path = str(self.project_root / "vectors" / "youtube_vectors")
        
        # Tavily API 키 설정 (환경 변수에서 가져오기)
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        if not self.tavily_api_key:
            print("주의: TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")
            print("Tavily 검색을 사용하려면 .env 파일에 TAVILY_API_KEY를 설정하세요.")
        else:
            print(f"Tavily API 키가 성공적으로 로드되었습니다. (길이: {len(self.tavily_api_key)}자)")
    
    def visualize_workflow(self, output_path=None):
        """
        LangGraph의 내장 기능을 사용하여 워크플로우를 시각화합니다.
        
        Args:
            output_path (str, optional): 저장할 이미지 파일 경로. 기본값은 None.
            
        Returns:
            str: 저장된 이미지 파일 경로 또는 HTML 시각화 문자열
        """
        # 기본 저장 경로 설정
        if output_path is None:
            viz_dir = os.path.join(self.project_root, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            output_path = os.path.join(viz_dir, "langgraph_workflow.png")
        
        # 그래프 시각화 실행
        try:
            # PNG 형식으로 저장
            self.graph.visualize(output_path)
            print(f"LangGraph 워크플로우 이미지 저장됨: {output_path}")
            return output_path
        except Exception as e:
            print(f"워크플로우 시각화 중 오류 발생: {str(e)}")
            # 대체 형식 (HTML)으로 시도
            try:
                html_output = self.graph.get_graph().to_html()
                html_path = output_path.replace(".png", ".html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_output)
                print(f"HTML 형식으로 저장됨: {html_path}")
                return html_path
            except Exception as e2:
                print(f"HTML 시각화 저장 중 오류 발생: {str(e2)}")
                return None
    
    def _create_workflow(self):
        """워크플로우 생성"""
        # 상태 타입 정의
        class State(dict):
            """워크플로우 상태"""
            messages: List[Any]
            documents: List[Any] = []
            question_type: str = "academic" # 기본값은 학술 질문
            run_id: str = None  # LangSmith 추적을 위한 run_id 필드 추가
            similarity_scores: List[float] = []  # l2 거리 리스트
            use_tavily: bool = False  # 타빌리 검색 사용 여부
            tavily_results: List[Any] = []  # 타빌리 검색 결과
        
        # 현재 클래스 인스턴스에 대한 참조 저장
        workflow_manager = self
        
        # 질문 분류 노드
        def classify_question(state: State) -> State:
            """질문 유형 분류"""
            question = state["messages"][-1].content
            try:
                # 도구 호출 대신 프롬프트 기반으로 질문 분류
                question_type = workflow_manager.llm_manager.classify_question(question)
                print(f"질문 유형: {question_type}")
                return {"messages": state["messages"], "question_type": question_type, "run_id": state.get("run_id")}
            except Exception as e:
                print(f"질문 분류 중 오류 발생: {str(e)}")
                # 기본값으로 학술 질문으로 처리
                return {"messages": state["messages"], "question_type": "academic", "run_id": state.get("run_id")}
                
        # 조건부 라우팅
        def route_by_type(state: State) -> Literal["academic_path", "counseling_path"]:
            """질문 유형에 따라 다음 경로 결정"""
            question_type = state.get("question_type", "academic")
            print(f"라우팅 - 질문 유형: {question_type}")
            if question_type == "counseling":
                return "counseling_path"
            else:
                return "academic_path"
        
        # 문서 검색 노드
        def retrieve_documents(state: State) -> State:
            """관련 문서 검색"""
            question = state["messages"][-1].content
            try:
                # similarity_search_with_score 메서드 사용하여 l2 거리 획득
                docs_with_scores = workflow_manager.vector_store.similarity_search_with_score(question, k=3)
                
                # 문서와 점수 분리
                docs = []
                scores = []
                
                # 거리 임계값 설정 (L2 거리가 클수록 유사도가 낮음)
                distance_threshold = 540  # L2 거리 임계값
                
                # 임계값보다 낮은 거리(높은 유사도)를 가진 문서만 선택
                filtered_docs_with_scores = []
                for doc, score in docs_with_scores:
                    if float(score) <= distance_threshold:
                        filtered_docs_with_scores.append((doc, float(score)))
                        docs.append(doc)
                        scores.append(float(score))
                
                # 최소 L2 거리 계산
                min_distance = min(scores) if scores else float('inf')
                
                print(f"검색된 문서: {len(docs_with_scores)}개, 임계값 이하 문서: {len(docs)}개 (임계값: {distance_threshold})")
                if scores:
                    print(f"최소 L2 거리: {min_distance:.4f}, 평균 L2 거리: {sum(scores)/len(scores):.4f}")
                
                # 각 문서에 필요한 메타데이터가 있는지 확인하고 보완
                for i, doc in enumerate(docs):
                    # 메타데이터가 없거나 불완전한 경우 보완
                    if not doc.metadata:
                        doc.metadata = {}
                    
                    # source가 지정되어 있지 않으면 '논문'으로 설정
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = '논문'
                    
                    # 제목이 없는 경우 첫 줄을 제목으로 사용
                    if 'title' not in doc.metadata or not doc.metadata['title']:
                        # 문서 내용에서 첫 번째 줄 또는 첫 50자를 제목으로 사용
                        first_line = doc.page_content.split('\n')[0][:50]
                        if len(first_line) > 0:
                            doc.metadata['title'] = first_line + ('...' if len(first_line) >= 50 else '')
                        else:
                            doc.metadata['title'] = f"논문 {i+1}" 
                    
                    # paper_id가 없는 경우 기본값 설정
                    if 'paper_id' not in doc.metadata:
                        doc.metadata['paper_id'] = f"doc_{i+1}"
                    
                    # journal이 없는 경우 기본값 설정
                    if 'journal' not in doc.metadata:
                        doc.metadata['journal'] = "학술 저널"
                        
                    # 거리 정보도 메타데이터에 추가
                    doc.metadata['distance_score'] = scores[i]
                
                # 임계값 이하의 문서가 없으면 타빌리 검색 사용
                use_tavily = len(docs) == 0
                
                if use_tavily:
                    print(f"임계값({distance_threshold}) 이하의 유사한 문서가 없어 타빌리 검색으로 전환합니다.")
                else:
                    print(f"임계값({distance_threshold}) 이하의 유사한 문서 {len(docs)}개를 사용합니다.")
                
                return {
                    "messages": state["messages"], 
                    "documents": docs, 
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id"),
                    "similarity_scores": scores,
                    "use_tavily": use_tavily
                }
            except Exception as e:
                print(f"문서 검색 중 오류 발생: {str(e)}")
                return {
                    "messages": state["messages"], 
                    "documents": [], 
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id"),
                    "use_tavily": True  # 오류 발생 시 타빌리 사용
                }
        
        # 타빌리 검색 노드
        def tavily_search(state: State) -> State:
            """타빌리 API를 사용한 검색"""
            question = state["messages"][-1].content
            try:
                # 타빌리 API 키 확인
                api_key = workflow_manager.tavily_api_key
                if not api_key:
                    print("타빌리 API 키가 설정되지 않았습니다.")
                    return state  # 변경 없이 상태 그대로 반환
                
                # 타빌리 클라이언트 초기화
                client = TavilyClient(api_key=api_key)
                
                # 검색 쿼리에 의학 용어 추가하여 검색 범위 좁히기
                search_query = f"depression research {question}"
                
                # 타빌리 검색 수행
                search_result = client.search(
                    query=search_query,
                    search_depth="advanced",  # 고급 검색 사용
                    max_results=3,  # 최대 3개 결과
                    include_domains=["pubmed.ncbi.nlm.nih.gov", "nih.gov", "ncbi.nlm.nih.gov", "scholar.google.com"], # 의학 관련 도메인
                    include_answer=True,  # 요약 답변 포함
                    include_raw_content=True  # 원본 콘텐츠 포함
                )
                
                # 결과를 Document 형식으로 변환
                tavily_docs = []
                for result in search_result.get("results", []):
                    content = result.get("content", "")
                    metadata = {
                        "title": result.get("title", "검색 결과"),
                        "url": result.get("url", ""),
                        "source": "Tavily",
                        "score": result.get("score", 0)
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    tavily_docs.append(doc)
                
                # 타빌리 요약 답변이 있으면 추가
                tavily_answer = search_result.get("answer", "")
                if tavily_answer:
                    answer_doc = Document(
                        page_content=tavily_answer,
                        metadata={
                            "title": "타빌리 요약", 
                            "source": "Tavily Summary",
                            "paper_id": None,
                            "journal": None
                        }
                    )
                    tavily_docs.append(answer_doc)
                
                print(f"타빌리 검색 완료: {len(tavily_docs)}개 문서 검색됨")
                
                # 타빌리 검색 결과만 사용 (논문 문서 제외)
                all_docs = tavily_docs
                
                return {
                    "messages": state["messages"],
                    "documents": all_docs,
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id"),
                    "tavily_results": tavily_docs
                }
            except Exception as e:
                print(f"타빌리 검색 중 오류 발생: {str(e)}")
                return state  # 오류 발생 시 기존 상태 반환
        
        # 유튜브 문서 검색 노드
        def retrieve_youtube_documents(state: State) -> State:
            """유튜브 관련 문서 검색"""
            question = state["messages"][-1].content
            try:
                print("유튜브 벡터 저장소 로드 시도...")
                # 유튜브 벡터 저장소에서 관련 컨텍스트 검색
                youtube_embeddings = YoutubeEmbeddingManager(model_name="bge-m3")
                vector_store_path = workflow_manager.youtube_vector_path
                print(f"벡터 저장소 경로: {vector_store_path}")
                
                # 벡터 저장소 로드
                vector_store = youtube_embeddings.load_embeddings(vector_store_path)
                print("벡터 저장소 로드 성공")
                
                # 유사도 검색 수행
                results = youtube_embeddings.similarity_search(question, k=2)
                print(f"상담 관련 {len(results)}개의 유튜브 컨텍스트를 검색했습니다.")
                
                return {
                    "messages": state["messages"], 
                    "documents": results,
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
            except Exception as e:
                print(f"유튜브 문서 검색 중 오류 발생: {str(e)}")
                # 오류 발생 시에도 워크플로우 계속 진행
                return {
                    "messages": state["messages"], 
                    "documents": [],
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
        
        # 다음 경로 결정 (타빌리 사용 여부)
        def route_academic_search(state: State) -> Literal["use_tavily", "skip_tavily"]:
            """논문 검색 결과에 따라 타빌리 검색 사용 여부 결정"""
            use_tavily = state.get("use_tavily", False)
            if use_tavily:
                print("타빌리 검색 노드로 라우팅합니다.")
                return "use_tavily"
            else:
                print("타빌리 검색을 건너뜁니다.")
                return "skip_tavily"
        
        # 학술 응답 생성 노드
        def generate_academic_response(state: State) -> State:
            """학술 질문에 대한 응답 생성"""
            question = state["messages"][-1].content
            documents = state.get("documents", [])
            
            try:
                if not documents:
                    response = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
                else:
                    # 타빌리 검색 결과가 있는지 확인
                    tavily_results = state.get("tavily_results", [])
                    has_tavily = len(tavily_results) > 0
                    
                    # 타빌리 검색을 사용한 경우 문서 메타데이터에 소스 정보 추가
                    if has_tavily:
                        # 논문 데이터 없이 타빌리 결과만 있는 경우
                        if not [doc for doc in documents if doc.metadata.get("source") != "Tavily" and doc.metadata.get("source") != "Tavily Summary"]:
                            # 타빌리 요약 문서의 메타데이터에 정보 추가
                            for doc in documents:
                                if "source_info" not in doc.metadata:
                                    doc.metadata["source_info"] = "이 응답은 웹 검색 결과만을 기반으로 생성되었습니다. 논문 데이터베이스에서 관련 정보를 찾을 수 없었습니다."
                        else:
                            # 혼합된 경우 소스 정보 추가
                            for doc in documents:
                                if "source_info" not in doc.metadata:
                                    doc.metadata["source_info"] = "이 응답은 논문 데이터베이스와 웹 검색 결과를 기반으로 생성되었습니다."
                    
                    # 스트리밍 콜백 확인 및 전달
                    callbacks = None
                    if workflow_manager.current_config and "callbacks" in workflow_manager.current_config:
                        callbacks = workflow_manager.current_config.get("callbacks")
                        print(f"학술 응답 생성에 콜백 전달: {callbacks}")
                    
                    # 응답 생성 - document 객체를 직접 전달
                    response = workflow_manager.llm_manager.generate_response(
                        question=question, 
                        context=documents,  # 문서 객체 그대로 전달
                        callbacks=callbacks
                    )
                
                return {
                    "messages": state["messages"] + [AIMessage(content=response, tags=["final"])], 
                    "documents": documents,
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
            except Exception as e:
                print(f"학술 응답 생성 중 오류 발생: {str(e)}")
                error_message = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                return {
                    "messages": state["messages"] + [AIMessage(content=error_message, tags=["final"])], 
                    "documents": documents,
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
        
        # 상담 응답 생성 노드
        def generate_counseling_response(state: State) -> State:
            """상담 질문에 대한 응답 생성"""
            question = state["messages"][-1].content
            documents = state.get("documents", [])
            
            try:
                # 컨텍스트 구성
                context = "\n\n".join([doc.page_content for doc in documents]) if documents else ""
                
                # 스트리밍 콜백 확인 및 전달
                callbacks = None
                if workflow_manager.current_config and "callbacks" in workflow_manager.current_config:
                    callbacks = workflow_manager.current_config.get("callbacks")
                    print(f"상담 응답 생성에 콜백 전달: {callbacks}")
                
                # 통합된 generate_counseling_response 함수 사용
                response = workflow_manager.llm_manager.generate_counseling_response(
                    question=question, 
                    context=context,
                    callbacks=callbacks
                )
                
                return {
                    "messages": state["messages"] + [AIMessage(content=response, tags=["final"])], 
                    "documents": documents,
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
            except Exception as e:
                print(f"상담 응답 생성 중 오류 발생: {str(e)}")
                error_message = "죄송합니다. 상담 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                return {
                    "messages": state["messages"] + [AIMessage(content=error_message, tags=["final"])], 
                    "documents": [],
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
        
        # 그래프 생성
        workflow = StateGraph(State)
        
        # 노드 추가
        workflow.add_node("classify", classify_question)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("tavily_search", tavily_search)
        workflow.add_node("retrieve_youtube", retrieve_youtube_documents)
        workflow.add_node("generate_academic", generate_academic_response)
        workflow.add_node("generate_counseling", generate_counseling_response)
        
        # 엣지 추가
        workflow.add_edge(START, "classify")
        
        # 학술 경로와 상담 경로 정의
        workflow.add_conditional_edges(
            "classify",
            route_by_type,
            {
                "academic_path": "retrieve",
                "counseling_path": "retrieve_youtube"
            }
        )
        
        # 논문 검색 결과에 따라 타빌리 검색 경로 조건부 추가
        workflow.add_conditional_edges(
            "retrieve", 
            route_academic_search,
            {
                "use_tavily": "tavily_search", 
                "skip_tavily": "generate_academic"
            }
        )
        
        # 나머지 엣지 추가
        workflow.add_edge("tavily_search", "generate_academic")
        workflow.add_edge("retrieve_youtube", "generate_counseling")
        workflow.add_edge("generate_academic", END)
        workflow.add_edge("generate_counseling", END)
        
        # 컴파일
        return workflow.compile()
    
    def process_message(self, message: str, callbacks=None, config=None):
        """
        사용자 메시지 처리
        
        Args:
            message (str): 사용자 메시지
            callbacks: 콜백 핸들러 리스트
            config: Runnable 설정
            
        Returns:
            dict: 결과 상태
        """
        # 기본 설정
        if config is None:
            config = {}
            
        # 콜백이 있으면 설정에 추가
        if callbacks:
            config["callbacks"] = callbacks
                
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        try:
            # 워크플로우 실행
            result = self.graph.invoke(initial_state, config=RunnableConfig(**config))
            return result
        except Exception as e:
            print(f"워크플로우 실행 중 오류 발생: {str(e)}")
            return {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=f"오류가 발생했습니다: {str(e)}", tags=["final"])
                ],
                "documents": [],
                "question_type": "academic"
            }
    
    def stream_process(self, message: str, callbacks=None, config=None):
        """
        사용자 메시지 처리 (스트리밍 모드)
        
        Args:
            message (str): 사용자 메시지  
            callbacks: 콜백 핸들러 리스트
            config: Runnable 설정
            
        Returns:
            generator: (state, metadata) 튜플의 제너레이터
        """
        # 기본 설정
        if config is None:
            config = {}
            
        # 콜백이 있으면 설정에 추가
        if callbacks:
            config["callbacks"] = callbacks
            
        # 현재 config 저장 (노드에서 접근할 수 있도록)
        self.current_config = config
        
        # LangSmith 통합을 위한 설정
        # 모든 단계가 동일한 run_id를 공유하도록 설정
        if "metadata" not in config:
            config["metadata"] = {}
        
        # 고유 런 ID 생성 (모든 단계가 이 ID를 공유)
        import uuid
        run_id = str(uuid.uuid4())
        config["metadata"]["run_id"] = run_id
        config["metadata"]["run_type"] = "workflow"
                
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "run_id": run_id  # 초기 상태에 run_id 추가
        }
        
        last_state = None
        
        try:
            # 스트리밍 모드로 실행
            config["configurable"] = {"thread_id": message}  # 스레드 ID 정의하여 캐싱
            
            # LangSmith 추적에 필요한 name 및 run_type 추가
            config["name"] = "Depression-Chatbot-Workflow"
            config["tags"] = ["workflow_execution"]
            
            # 각 단계를 같은 워크플로우의 일부로 식별하기 위한 메타데이터
            config["metadata"]["workflow_id"] = run_id
            
            for output in self.graph.stream(
                initial_state, 
                config=RunnableConfig(**config),
                stream_mode="values"
            ):
                # 출력 처리
                if isinstance(output, dict):
                    # 현재 노드에서의 상태 업데이트
                    node_name = "unknown"
                    state_delta = output
                    
                    # 기존 상태값 유지
                    if last_state is None:
                        last_state = state_delta
                    else:
                        # 이전 상태 정보 유지하면서 업데이트
                        updated_state = last_state.copy()
                        for key, value in state_delta.items():
                            updated_state[key] = value
                            
                        last_state = updated_state
                        
                    # langgraph_node 메타데이터 유추 
                    if "messages" in state_delta and len(state_delta["messages"]) > len(initial_state["messages"]):
                        # 새 메시지가 추가되었다면 응답 생성 노드
                        if "question_type" in last_state and last_state["question_type"] == "counseling":
                            node_name = "generate_counseling"
                        else:
                            node_name = "generate_academic"
                    elif "documents" in state_delta and state_delta["documents"]:
                        # 문서가 추가되었다면 검색 노드
                        if "question_type" in last_state and last_state["question_type"] == "counseling":
                            node_name = "retrieve_youtube"
                        else:
                            node_name = "retrieve"
                    elif "question_type" in state_delta:
                        # 질문 유형이 추가되었다면 분류 노드
                        node_name = "classify"
                        
                    print(f"스트리밍 노드: {node_name}, 상태 키: {', '.join(state_delta.keys())}")
                    
                    # 메타데이터에 단계 정보 추가
                    metadata = {
                        "langgraph_node": node_name,
                        "workflow_run_id": run_id,  # 공통 워크플로우 ID
                        "node_type": node_name      # 현재 노드 유형
                    }
                    
                    # 상태와 노드 이름 반환
                    yield last_state, metadata
                    
                elif isinstance(output, tuple) and len(output) == 2:
                    # (node_name, state) 형식으로 반환된 경우
                    node_name, state_delta = output
                    
                    # 상태 업데이트
                    if last_state is None:
                        last_state = state_delta
                    else:
                        # 이전 상태 정보 유지하면서 업데이트
                        updated_state = last_state.copy()
                        for key, value in state_delta.items():
                            updated_state[key] = value
                            
                        last_state = updated_state
                        
                    print(f"스트리밍 노드: {node_name}, 상태 키: {', '.join(state_delta.keys())}")
                    
                    # 메타데이터에 단계 정보 추가
                    metadata = {
                        "langgraph_node": node_name,
                        "workflow_run_id": run_id,  # 공통 워크플로우 ID
                        "node_type": node_name      # 현재 노드 유형
                    }
                    
                    # 상태와 노드 이름 반환
                    yield last_state, metadata
                    
                else:
                    # 다른 형태로 반환된 경우
                    print(f"알 수 없는 출력 형식: {type(output)}")
                    
                    if last_state is None:
                        last_state = {"messages": initial_state["messages"]}
                    
                    # 메타데이터에 정보 추가
                    metadata = {
                        "langgraph_node": "unknown",
                        "workflow_run_id": run_id
                    }
                            
                    # 상태와 기본 노드 이름 반환
                    yield last_state, metadata
                
        except Exception as e:
            print(f"워크플로우 스트리밍 중 오류 발생: {str(e)}")
            # 예외 발생 시 에러 메시지와 빈 문서 목록 반환
            final_state = {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=f"오류가 발생했습니다: {str(e)}", tags=["final"])
                ],
                "documents": [],
                "question_type": "academic"
            }
            
            # 에러 메타데이터
            error_metadata = {
                "langgraph_node": "error",
                "workflow_run_id": run_id if 'run_id' in locals() else str(uuid.uuid4()),
                "error": str(e)
            }
            
            # 에러 상태 반환
            yield final_state, error_metadata 