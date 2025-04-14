import os
from typing import Literal, Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from src.utils.youtube_embeddings import YoutubeEmbeddings
from pathlib import Path

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
    
    def _create_workflow(self):
        """워크플로우 생성"""
        # 상태 타입 정의
        class State(dict):
            """워크플로우 상태"""
            messages: List[Any]
            documents: List[Any] = []
            question_type: str = "academic" # 기본값은 학술 질문
            run_id: str = None  # LangSmith 추적을 위한 run_id 필드 추가
        
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
                docs = workflow_manager.vector_store.similarity_search(question, k=3)
                print(f"{len(docs)}개의 관련 문서를 검색했습니다.")
                return {
                    "messages": state["messages"], 
                    "documents": docs, 
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
            except Exception as e:
                print(f"문서 검색 중 오류 발생: {str(e)}")
                return {
                    "messages": state["messages"], 
                    "documents": [], 
                    "question_type": state.get("question_type"),
                    "run_id": state.get("run_id")
                }
        
        # 유튜브 문서 검색 노드
        def retrieve_youtube_documents(state: State) -> State:
            """유튜브 관련 문서 검색"""
            question = state["messages"][-1].content
            try:
                print("유튜브 벡터 저장소 로드 시도...")
                # 유튜브 벡터 저장소에서 관련 컨텍스트 검색
                youtube_embeddings = YoutubeEmbeddings()  # BGE-Large-EN 사용
                vector_store_path = workflow_manager.youtube_vector_path
                print(f"벡터 저장소 경로: {vector_store_path}")
                
                # 벡터 저장소 로드
                vector_store = youtube_embeddings.load_embeddings(vector_store_path)
                print("벡터 저장소 로드 성공")
                
                # 벡터 저장소를 인스턴스 변수로 저장
                youtube_embeddings.vector_store = vector_store
                
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
        
        # 학술 응답 생성 노드
        def generate_academic_response(state: State) -> State:
            """학술 질문에 대한 응답 생성"""
            question = state["messages"][-1].content
            documents = state.get("documents", [])
            
            try:
                if not documents:
                    response = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
                else:
                    context = "\n\n".join([doc.page_content for doc in documents])
                    # 스트리밍 콜백 확인 및 전달 (워크플로우 매니저에서 config 가져오기)
                    callbacks = None
                    if workflow_manager.current_config and "callbacks" in workflow_manager.current_config:
                        callbacks = workflow_manager.current_config.get("callbacks")
                        print(f"학술 응답 생성에 콜백 전달: {callbacks}")
                    
                    # llm_manager의 generate_response 메서드 호출 (콜백 전달)
                    response = workflow_manager.llm_manager.generate_response(question, context, callbacks=callbacks)
                
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
                
                # 상담 응답 생성 (유튜브 컨텍스트 포함)
                response = workflow_manager.llm_manager.generate_counseling_response(
                    question, 
                    context=context if context else None,
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
        
        # 각 경로의 나머지 엣지 추가
        workflow.add_edge("retrieve", "generate_academic")
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