from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

class SupervisorAgent:
    def __init__(self, model: str = "gpt-4", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 아래 두 전문가를 조율하는 Supervisor입니다:

- Counselor: 감정, 대인관계, 스트레스 등 심리상담을 제공합니다.
- PsychiatricDoc: 정신질환, 약물치료 등 정신의학적 조언을 제공합니다.

아래 사용자의 메시지를 보고 적절한 전문가를 선택하세요.
응답은 반드시 다음 중 하나여야 합니다:
- Counselor / PsychiatricDoc / FINISH"""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        self.chain = self.prompt | self.llm
    
    def route(self, state):
        response = self.chain.invoke({"messages": state["messages"]})
        content = response.content.strip()
        if content in ["Counselor", "PsychiatricDoc", "FINISH"]:
            return {"next": content}
        else:
            # 기본적으로 Counselor로 라우팅
            return {"next": "Counselor"}
    
    def get_agent(self):
        return RunnableLambda(self.route)

# 기본 인스턴스 생성
supervisor_agent = SupervisorAgent().get_agent()

__all__ = ['SupervisorAgent', 'supervisor_agent']
