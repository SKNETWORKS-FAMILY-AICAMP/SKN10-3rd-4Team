from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from mental_agent_system.utils.faiss_utils import load_faiss_from_korean_path

class PsychiatricDocAgent:
    def __init__(self, temperature: float = 0.7):
        self.template = """당신은 정신건강의학과 전문의입니다. 과학적 근거를 바탕으로 전문적이고 명확한 의견을 제시해주세요.

주요 역할:
1. 증상에 대한 전문적 분석과 평가
2. 과학적 근거 기반의 치료 방향 제시
3. 약물치료와 관련된 전문적 조언
4. 정신건강 관련 의학적 지식 전달
5. 필요한 경우 추가 검사나 치료 방향 제안

상담 스타일:
- 전문적이고 객관적인 어조 유지
- 의학적 용어를 알기 쉽게 설명
- 명확하고 구체적인 치료 계획 제시
- 과학적 근거와 연구 결과 인용
- 환자의 상태에 따른 맞춤형 조언

주의사항:
- 의학적 정확성 유지
- 환자가 이해하기 쉬운 설명 제공
- 필요한 경우 다른 전문가와의 협진 제안
- 응급상황 시 적절한 대처방안 안내

참고할 내용: {context}

환자의 질문: {question}

위 내용을 참고하여 정신과 전문의로서 전문적이고 명확한 답변을 제공해주세요."""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(temperature=temperature)
        
        # 프로젝트 루트 디렉토리 찾기
        project_root = Path(__file__).parent.parent.parent
        vectorstore_path = project_root / "vectorstore" / "psychiatric_doc_faiss_index"
        
        # 기존 FAISS 인덱스 로드
        self.db = load_faiss_from_korean_path(vectorstore_path)
    
    def run(self, state):
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
        
        return {"messages": [AIMessage(content=response.content, name="PsychiatricDoc")]}

    def get_agent(self):
        return RunnableLambda(self.run)

# 기본 인스턴스 생성
psychiatric_doc_agent = PsychiatricDocAgent().get_agent()
