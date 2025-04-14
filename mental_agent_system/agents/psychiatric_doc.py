from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from mental_agent_system.utils.faiss_utils import load_faiss_from_korean_path

class PsychiatricDocAgent:
    def __init__(self, temperature: float = 0.7):
        self.template = """당신은 정신의학 전문가입니다. 제공된 PubMed 논문들 중에서 질문과 가장 관련성이 높은 논문들만 선별하여 답변해 주세요.

        질문: {question}

        참고할 논문 내용:
        {context}

        답변 작성 과정:
        1. 제공된 논문들 중 질문과 직접적인 관련이 있는 논문들만 선별하세요
        2. 관련성이 낮거나 질문에 도움이 되지 않는 논문은 분석에서 제외하세요
        3. 선별한 논문들의 정보를 종합하여 하나의 일관된 답변을 작성하세요

        답변 작성 지침:
        1. 답변 시작 부분에 어떤 논문들이 가장 관련성이 높았는지 간략히 언급하세요
        2. 선별한 논문들 간의 공통점과 차이점을 파악하여 분석하세요
        3. 각 논문의 핵심 발견과 결론을 통합적으로 설명하세요
        4. 답변 내용을 명확하게 설명하고 논리적으로 구성하세요
        5. 선별한 논문 정보를 반드시 언급하세요 (제목, 저널)
        6. 정보가 불충분하거나 논문 간 상충되는 내용이 있는 경우 정직하게 인정하세요

        한국어로 명확하고 전문적인 답변을 제공해 주세요.
"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.llm = ChatOpenAI(temperature=temperature)
        
        # 프로젝트 루트 디렉토리 찾기
        project_root = Path(__file__).parent.parent.parent
        vectorstore_path = project_root / "vectorstore" / "psychiatric_doc_faiss_index"
        
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
        
        return {"messages": [AIMessage(content=response.content, name="PsychiatricDoc")]}

    def get_agent(self):
        return RunnableLambda(self.run)

# 기본 인스턴스 생성
psychiatric_doc_agent = PsychiatricDocAgent().get_agent()
