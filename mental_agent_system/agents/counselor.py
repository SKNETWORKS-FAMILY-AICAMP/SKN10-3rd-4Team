from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from mental_agent_system.utils.faiss_utils import load_faiss_from_korean_path

class CounselorAgent:
    def __init__(self, temperature: float = 0.7):
        self.template = """당신은 전문 심리상담가입니다. 내담자의 이야기를 경청하고 공감하며, 따뜻한 태도로 상담을 진행해주세요.

주요 역할:
1. 내담자의 감정을 인정하고 공감하기
2. 비판단적이고 수용적인 태도 유지하기
3. 내담자가 스스로 해결책을 찾을 수 있도록 안내하기
4. 긍정적인 행동 변화를 위한 실천 가능한 제안하기
5. 필요한 경우 정서적 지지와 격려 제공하기

상담 스타일:
- 따뜻하고 부드러운 어조 사용
- 공감적 경청과 반영
- 개방형 질문을 통한 대화 유도
- 내담자의 강점과 자원 발견 돕기
- 점진적인 행동 변화 격려

주의사항:
- 섣부른 판단이나 비판 피하기
- 일방적인 조언 대신 함께 고민하는 자세
- 내담자의 페이스 존중하기

참고할 내용: {context}

내담자의 질문: {question}

위 내용을 참고하여 상담가로서 공감적이고 따뜻한 태도로 답변해주세요."""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(temperature=temperature)
        
        # 프로젝트 루트 디렉토리 찾기
        project_root = Path(__file__).parent.parent.parent
        # vectorstore_path = project_root / "vectorstore" / "counselor_faiss_index"
        vectorstore_path = project_root / "vectorstore" / "counselor_summary_faiss_index"

        # 기존 FAISS 인덱스 로드
        self.db = load_faiss_from_korean_path(vectorstore_path)
    
    def run(self, state):
        # 메시지가 문자열인 경우와 Message 객체인 경우를 모두 처리
        if isinstance(state["messages"][-1], str):
            query = state["messages"][-1]
        else:
            query = state["messages"][-1].content
        
        # 관련 문서 검색
        docs = self.db.similarity_search(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 프롬프트 생성 및 응답
        messages = self.prompt.format_messages(
            context=context,
            question=query
        )
        response = self.llm.invoke(messages)
        
        return {"messages": [AIMessage(content=response.content, name="Counselor")]}

    def get_agent(self):
        return RunnableLambda(self.run)

# 기본 인스턴스 생성
counselor_agent = CounselorAgent().get_agent()
