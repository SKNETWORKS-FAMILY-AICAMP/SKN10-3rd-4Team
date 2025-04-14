import os
from typing import Literal, Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import StateGraph, START, END

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
    
    def _create_workflow(self):
        """워크플로우 생성"""
        # 상태 타입 정의
        class State(dict):
            """워크플로우 상태"""
            messages: List[Any]
            documents: List[Any] = []
            question_type: str = "academic" # 기본값은 학술 질문
        
        # 질문 분류 노드
        def classify_question(state: State) -> State:
            """질문 유형 분류"""
            question = state["messages"][-1].content
            try:
                # 도구 호출 대신 프롬프트 기반으로 질문 분류
                question_type = self.llm_manager.classify_question(question)
                print(f"질문 유형: {question_type}")
                return {"messages": state["messages"], "question_type": question_type}
            except Exception as e:
                print(f"질문 분류 중 오류 발생: {str(e)}")
                # 기본값으로 학술 질문으로 처리
                return {"messages": state["messages"], "question_type": "academic"}
                
        # 조건부 라우팅
        def route_by_type(state: State) -> str:
            """질문 유형에 따라 다음 노드 결정"""
            question_type = state.get("question_type", "academic")
            if question_type == "counseling":
                return "generate_counseling"
            else:
                return "retrieve"
        
        # 문서 검색 노드
        def retrieve_documents(state: State) -> State:
            """관련 문서 검색"""
            question = state["messages"][-1].content
            try:
                docs = self.vector_store.similarity_search(question, k=3)
                print(f"{len(docs)}개의 관련 문서를 검색했습니다.")
                return {"messages": state["messages"], "documents": docs, "question_type": state.get("question_type")}
            except Exception as e:
                print(f"문서 검색 중 오류 발생: {str(e)}")
                return {"messages": state["messages"], "documents": [], "question_type": state.get("question_type")}
        
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
                    # llm_manager의 generate_response 메서드는 일반 텍스트 응답을 반환합니다
                    response = self.llm_manager.generate_response(question, context)
                
                return {
                    "messages": state["messages"] + [AIMessage(content=response, tags=["final"])], 
                    "documents": documents,
                    "question_type": state.get("question_type")
                }
            except Exception as e:
                print(f"학술 응답 생성 중 오류 발생: {str(e)}")
                error_message = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                return {
                    "messages": state["messages"] + [AIMessage(content=error_message, tags=["final"])], 
                    "documents": documents,
                    "question_type": state.get("question_type")
                }
        
        # 상담 응답 생성 노드
        def generate_counseling_response(state: State) -> State:
            """상담 질문에 대한 응답 생성"""
            question = state["messages"][-1].content
            
            try:
                # 상담 응답 생성 (문서 검색 없이)
                response = self.llm_manager.generate_counseling_response(question)
                
                return {
                    "messages": state["messages"] + [AIMessage(content=response, tags=["final"])], 
                    "documents": [],  # 상담 응답은 문서를 참조하지 않음
                    "question_type": state.get("question_type")
                }
            except Exception as e:
                print(f"상담 응답 생성 중 오류 발생: {str(e)}")
                error_message = "죄송합니다. 상담 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                return {
                    "messages": state["messages"] + [AIMessage(content=error_message, tags=["final"])], 
                    "documents": [],
                    "question_type": state.get("question_type")
                }
        
        # 그래프 생성
        workflow = StateGraph(State)
        
        # 노드 추가
        workflow.add_node("classify", classify_question)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate_academic", generate_academic_response)
        workflow.add_node("generate_counseling", generate_counseling_response)
        
        # 엣지 추가
        workflow.add_edge(START, "classify")
        # 조건부 엣지 추가 (라우터 함수 사용)
        workflow.add_conditional_edges(
            "classify",
            route_by_type,
            {
                "retrieve": "retrieve",
                "generate_counseling": "generate_counseling"
            }
        )
        workflow.add_edge("retrieve", "generate_academic")
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
                
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        last_state = None
        
        try:
            # 1. 먼저 전체 실행을 통해 분류부터 실행
            # (질문 분류를 위한 사전 단계로 실행하고 결과 캐시하기)
            config_copy = config.copy()
            config_copy["configurable"] = {"thread_id": message}  # 스레드 ID 정의하여 캐싱
            
            print("전체 워크플로우 실행 중...")
            full_result = self.graph.invoke(
                initial_state,
                config=RunnableConfig(**config_copy)
            )
            
            # 분류 결과와 문서 캐시
            question_type = full_result.get("question_type", "academic")
            documents = full_result.get("documents", [])
            
            print(f"전체 실행 후: 질문 유형: {question_type}, 문서 수: {len(documents)}")
            
            # 2. 스트리밍 모드로 다시 실행 (노드 진행 과정과 응답 생성 스트리밍 위해)
            config["configurable"] = {"thread_id": message}  # 같은 스레드 ID로 캐시 활용
            
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
                        # 첫 상태에 캐시된 값 확실히 추가
                        state_delta["question_type"] = question_type
                        
                        # 문서가 있으면 추가
                        if documents and "documents" not in state_delta:
                            state_delta["documents"] = documents
                            
                        last_state = state_delta
                    else:
                        # 이전 상태 정보 유지하면서 업데이트
                        updated_state = last_state.copy()
                        for key, value in state_delta.items():
                            updated_state[key] = value
                            
                        # 질문 유형 유지
                        if "question_type" not in state_delta and question_type:
                            updated_state["question_type"] = question_type
                            
                        last_state = updated_state
                        
                    # langgraph_node 메타데이터 유추 
                    if "messages" in state_delta and len(state_delta["messages"]) > len(initial_state["messages"]):
                        # 새 메시지가 추가되었다면 응답 생성 노드
                        if question_type == "counseling":
                            node_name = "generate_counseling"
                        else:
                            node_name = "generate_academic"
                    elif "documents" in state_delta and state_delta["documents"]:
                        # 문서가 추가되었다면 검색 노드
                        node_name = "retrieve"
                    elif "question_type" in state_delta:
                        # 질문 유형이 추가되었다면 분류 노드
                        node_name = "classify"
                        
                    print(f"스트리밍 노드: {node_name}, 상태 키: {', '.join(state_delta.keys())}")
                    
                    # 상태와 노드 이름 반환
                    yield last_state, {"langgraph_node": node_name}
                    
                elif isinstance(output, tuple) and len(output) == 2:
                    # (node_name, state) 형식으로 반환된 경우
                    node_name, state_delta = output
                    
                    # 상태 업데이트
                    if last_state is None:
                        # 첫 상태에 캐시된 값 확실히 추가
                        state_delta["question_type"] = question_type
                        
                        # 문서가 있으면 추가
                        if documents and "documents" not in state_delta:
                            state_delta["documents"] = documents
                            
                        last_state = state_delta
                    else:
                        # 이전 상태 정보 유지하면서 업데이트
                        updated_state = last_state.copy()
                        for key, value in state_delta.items():
                            updated_state[key] = value
                            
                        # 질문 유형 유지
                        if "question_type" not in state_delta and question_type:
                            updated_state["question_type"] = question_type
                            
                        last_state = updated_state
                        
                    print(f"스트리밍 노드: {node_name}, 상태 키: {', '.join(state_delta.keys())}")
                    
                    # 상태와 노드 이름 반환
                    yield last_state, {"langgraph_node": node_name}
                    
                else:
                    # 다른 형태로 반환된 경우
                    print(f"알 수 없는 출력 형식: {type(output)}")
                    
                    if last_state is None:
                        last_state = {"messages": initial_state["messages"], "question_type": question_type}
                        if documents:
                            last_state["documents"] = documents
                            
                    # 상태와 기본 노드 이름 반환
                    yield last_state, {"langgraph_node": "unknown"}
                
        except Exception as e:
            print(f"워크플로우 스트리밍 중 오류 발생: {str(e)}")
            # 예외 발생 시 에러 메시지와 빈 문서 목록 반환
            final_state = {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=f"오류가 발생했습니다: {str(e)}", tags=["final"])
                ],
                "documents": [],
                "question_type": question_type if 'question_type' in locals() else "academic"
            }
            # 에러 상태 반환
            yield final_state, {"langgraph_node": "unknown"} 