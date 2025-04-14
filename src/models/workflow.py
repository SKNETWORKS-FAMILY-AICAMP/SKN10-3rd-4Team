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
        
        # 문서 검색 노드
        def retrieve_documents(state: State) -> State:
            """관련 문서 검색"""
            question = state["messages"][-1].content
            try:
                docs = self.vector_store.similarity_search(question, k=3)
                print(f"{len(docs)}개의 관련 문서를 검색했습니다.")
                return {"messages": state["messages"], "documents": docs}
            except Exception as e:
                print(f"문서 검색 중 오류 발생: {str(e)}")
                return {"messages": state["messages"], "documents": []}
        
        # 응답 생성 노드
        def generate_response(state: State) -> State:
            """응답 생성"""
            question = state["messages"][-1].content
            documents = state.get("documents", [])
            
            try:
                if not documents:
                    response = "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 해주시겠어요?"
                else:
                    context = "\n\n".join([doc.page_content for doc in documents])
                    # llm_manager의 generate_response 메서드는 일반 텍스트 응답을 반환합니다
                    response = self.llm_manager.generate_response(question, context)
                
                return {"messages": state["messages"] + [AIMessage(content=response, tags=["final"])], "documents": documents}
            except Exception as e:
                print(f"응답 생성 중 오류 발생: {str(e)}")
                error_message = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
                return {"messages": state["messages"] + [AIMessage(content=error_message, tags=["final"])], "documents": documents}
        
        # 그래프 생성
        workflow = StateGraph(State)
        
        # 노드 추가
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_response)
        
        # 엣지 추가
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
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
                "documents": []
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
            # 워크플로우 스트리밍 실행
            for output in self.graph.stream(
                initial_state, 
                config=RunnableConfig(**config),
                stream_mode="updates"
            ):
                # stream_mode="updates"는 (node_name, state_delta)를 반환
                node_name, state_delta = output
                
                # 완전한 상태 구성
                if last_state is None:
                    last_state = state_delta
                else:
                    # 새 상태에 이전 상태 값을 유지하면서 업데이트
                    for key, value in state_delta.items():
                        last_state[key] = value
                
                # 상태와 노드 이름 반환
                yield last_state, {"langgraph_node": node_name}
                
        except Exception as e:
            print(f"워크플로우 스트리밍 중 오류 발생: {str(e)}")
            # 예외 발생 시 에러 메시지와 빈 문서 목록 반환
            final_state = {
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=f"오류가 발생했습니다: {str(e)}", tags=["final"])
                ],
                "documents": []
            }
            # 에러 상태 반환
            yield final_state, {"langgraph_node": "generate"} 