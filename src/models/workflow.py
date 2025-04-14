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
        LangGraph를 직접 사용하지 않고 토큰 단위 스트리밍을 관리하는 함수
        """
        # 초기 상태
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # 1. 질문 분류
        try:
            print("1단계: 질문 분류 수행")
            # 질문 분류 직접 수행
            question = message
            try:
                question_type = self.llm_manager.classify_question(question)
                print(f"질문 유형: {question_type}")
            except Exception as e:
                print(f"질문 분류 중 오류: {str(e)}")
                question_type = "academic"  # 기본값
                
            # 첫 번째 상태
            first_state = {
                "messages": initial_state["messages"],
                "question_type": question_type
            }
            
            # 분류 결과 반환
            yield first_state, {"langgraph_node": "classify"}
            
            # 2. 문서 검색 또는 상담 응답 생성
            documents = []
            
            if question_type == "academic":
                print("2단계: 학술 질문 - 문서 검색")
                try:
                    # 문서 검색 직접 수행
                    documents = self.vector_store.similarity_search(question, k=3)
                    print(f"검색된 문서 수: {len(documents)}")
                    
                    # 문서 검색 결과 상태
                    search_state = {
                        "messages": first_state["messages"],
                        "question_type": question_type,
                        "documents": documents
                    }
                    
                    # 문서 검색 결과 반환
                    yield search_state, {"langgraph_node": "retrieve"}
                    
                    # 3. 학술 응답 생성 (스트리밍)
                    print("3단계: 학술 응답 생성 (토큰 스트리밍)")
                    
                    # 문서 내용 컨텍스트 생성
                    if documents:
                        context = "\n\n".join([doc.page_content for doc in documents])
                        
                        # 학술 응답 토큰 단위 스트리밍
                        try:
                            # 직접 토큰 단위 스트리밍으로 응답 생성 (콜백으로 토큰 전달)
                            # 응답 버퍼 (토큰이 추가될 때마다 업데이트)
                            response_buffer = ""
                            
                            # 토큰 콜백 정의
                            class TokenCallback:
                                def __init__(self, state, node):
                                    self.state = state
                                    self.node = node
                                    
                                def on_llm_new_token(self, token, **kwargs):
                                    nonlocal response_buffer
                                    response_buffer += token
                                    
                                    # 토큰마다 현재 상태 복사 및 메시지 업데이트
                                    token_state = self.state.copy()
                                    token_state["messages"] = token_state["messages"] + [
                                        AIMessage(content=response_buffer, tags=["streaming"])
                                    ]
                                    
                                    # 토큰 단위로 상태 반환
                                    return token_state, {"langgraph_node": self.node}
                            
                            # 토큰 콜백 인스턴스 생성
                            token_callback = TokenCallback(search_state, "generate_academic")
                            
                            # 오리지널 콜백이 있으면 추가
                            streaming_callbacks = []
                            if callbacks:
                                streaming_callbacks.extend(callbacks)
                            
                            # 토큰 단위 응답 생성
                            for token in self.llm_manager.stream_response_tokens(question, context, streaming_callbacks):
                                # 토큰 콜백 호출
                                token_state = search_state.copy()
                                response_buffer += token
                                token_state["messages"] = token_state["messages"] + [
                                    AIMessage(content=response_buffer, tags=["streaming"])
                                ]
                                
                                # 토큰 단위로 상태 반환
                                yield token_state, {"langgraph_node": "generate_academic"}
                            
                            # 최종 응답 상태
                            final_state = search_state.copy()
                            final_state["messages"] = search_state["messages"] + [
                                AIMessage(content=response_buffer, tags=["final"])
                            ]
                            
                            # 최종 상태 반환
                            yield final_state, {"langgraph_node": "generate_academic"}
                            
                        except Exception as e:
                            print(f"학술 응답 생성 스트리밍 중 오류: {str(e)}")
                            error_response = "학술 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                            error_state = {
                                "messages": search_state["messages"] + [AIMessage(content=error_response, tags=["final"])],
                                "question_type": question_type,
                                "documents": documents
                            }
                            yield error_state, {"langgraph_node": "generate_academic"}
                    else:
                        # 문서가 없는 경우
                        no_docs_response = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
                        no_docs_state = {
                            "messages": search_state["messages"] + [AIMessage(content=no_docs_response, tags=["final"])],
                            "question_type": question_type,
                            "documents": []
                        }
                        yield no_docs_state, {"langgraph_node": "generate_academic"}
                        
                except Exception as e:
                    print(f"문서 검색 중 오류: {str(e)}")
                    error_response = "문서 검색 중 오류가 발생했습니다. 다시 시도해주세요."
                    error_state = {
                        "messages": first_state["messages"] + [AIMessage(content=error_response, tags=["final"])],
                        "question_type": question_type,
                        "documents": []
                    }
                    yield error_state, {"langgraph_node": "retrieve"}
                    
            else:  # 상담 질문
                print("2단계: 상담 질문 - 응답 생성 (토큰 스트리밍)")
                
                try:
                    # 상담 응답 토큰 단위 스트리밍
                    # 응답 버퍼 (토큰이 추가될 때마다 업데이트)
                    response_buffer = ""
                    
                    # 토큰 단위 상담 응답 생성
                    for token in self.llm_manager.stream_counseling_tokens(question, callbacks):
                        # 토큰마다 상태 업데이트
                        token_state = first_state.copy()
                        response_buffer += token
                        token_state["messages"] = token_state["messages"] + [
                            AIMessage(content=response_buffer, tags=["streaming"])
                        ]
                        
                        # 토큰 단위로 상태 반환
                        yield token_state, {"langgraph_node": "generate_counseling"}
                    
                    # 최종 응답 상태
                    final_state = first_state.copy()
                    final_state["messages"] = first_state["messages"] + [
                        AIMessage(content=response_buffer, tags=["final"])
                    ]
                    
                    # 최종 상태 반환
                    yield final_state, {"langgraph_node": "generate_counseling"}
                    
                except Exception as e:
                    print(f"상담 응답 생성 스트리밍 중 오류: {str(e)}")
                    error_response = "상담 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                    error_state = {
                        "messages": first_state["messages"] + [AIMessage(content=error_response, tags=["final"])],
                        "question_type": question_type,
                        "documents": []
                    }
                    yield error_state, {"langgraph_node": "generate_counseling"}
                
        except Exception as e:
            print(f"전체 워크플로우 실행 중 오류 발생: {str(e)}")
            error_state = {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content="죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.", tags=["final"])
                ],
                "question_type": "academic" if not 'question_type' in locals() else question_type,
                "documents": []
            }
            yield error_state, {"langgraph_node": "error"} 