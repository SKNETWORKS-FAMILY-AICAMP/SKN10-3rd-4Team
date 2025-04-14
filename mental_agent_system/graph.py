from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
import operator
from .agents.counselor import CounselorAgent
from .agents.psychiatric_doc import PsychiatricDocAgent
from .supervisor import SupervisorAgent

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    next: str

def should_end(state: AgentState) -> bool:
    """대화 종료 여부를 결정하는 함수"""
    # 메시지가 10개를 넘어가면 종료
    if len(state["messages"]) >= 10:
        return True
    
    # 마지막 메시지가 특정 키워드를 포함하면 종료
    last_msg = state["messages"][-1].content.lower()
    end_keywords = ["감사합니다", "안녕히 계세요", "상담 종료", "goodbye", "thank you"]
    return any(keyword in last_msg for keyword in end_keywords)

def create_graph():
    """대화 에이전트 그래프 생성"""
    # 상태 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 에이전트 인스턴스 생성
    supervisor = SupervisorAgent().get_agent()
    counselor = CounselorAgent().get_agent()
    psychiatric_doc = PsychiatricDocAgent().get_agent()
    
    # 노드 추가
    workflow.add_node("Supervisor", supervisor)
    workflow.add_node("Counselor", counselor)
    workflow.add_node("PsychiatricDoc", psychiatric_doc)

    # Supervisor에서 라우팅
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "Counselor": "Counselor",
            "PsychiatricDoc": "PsychiatricDoc",
            "FINISH": END
        }
    )

    # 전문가에서 END로 바로 이동
    workflow.add_edge("Counselor", END)
    workflow.add_edge("PsychiatricDoc", END)
    
    # 시작점 설정
    workflow.set_entry_point("Supervisor")

    return workflow.compile()

# 그래프 인스턴스 생성
graph = create_graph()

__all__ = ['graph', 'AgentState', 'create_graph']
