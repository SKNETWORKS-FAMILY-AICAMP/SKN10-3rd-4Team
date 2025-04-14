from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import openai

# 환경 변수 로드
load_dotenv()

class LLMManager:
    """LLM 관리자 클래스"""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        LLM 관리자 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델 이름
        """
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def classify_question(self, question: str) -> str:
        """
        질문 유형을 분류
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            str: 'academic' 또는 'counseling' 분류 결과
        """
        try:
            system_prompt = """당신은 우울증 관련 질문을 분류하는 전문가입니다. 
            사용자의 질문을 다음 두 가지 유형 중 하나로 분류해주세요:
            1. academic: 우울증에 대한 정보, 통계, 치료법 등 학술적 정보를 요청하는 질문
            2. counseling: 개인적인 조언, 상담, 심리적 지원을 요청하는 질문

            제공된 질문을 분석하고, 'academic' 또는 'counseling' 중 하나만 응답하세요.
            확실하지 않은 경우, 'academic'으로 분류하세요.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # 'academic' 또는 'counseling'이 아닌 경우 기본값 반환
            if "academic" in result:
                return "academic"
            elif "counseling" in result:
                return "counseling"
            else:
                return "academic"  # 기본값
                
        except Exception as e:
            print(f"질문 분류 중 오류 발생: {str(e)}")
            return "academic"  # 오류 발생 시 학술 질문으로 간주
    
    def generate_response(self, question: str, context: str) -> str:
        """
        학술 질문에 대한 응답 생성
        
        Args:
            question (str): 사용자 질문
            context (str): 검색된 문서 내용
            
        Returns:
            str: 생성된 응답
        """
        try:
            system_prompt = """당신은 우울증에 관한 질문에 답변하는 전문가입니다.
            다음 정보를 사용하여 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
            제공된 정보에 답변이 없는 경우, 그 사실을 솔직히 인정하고 해당 주제에 대해 일반적인 학술적 정보를 제공하세요.
            질문에 공감을 표현하고, 학술적 근거를 바탕으로 정보를 제공하세요.
            """
            
            context_prompt = f"""참고 정보:
            {context}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요."
    
    def generate_counseling_response(self, question: str) -> str:
        """
        상담 질문에 대한 응답 생성
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            str: 생성된 응답
        """
        try:
            system_prompt = """당신은 우울증 관련 상담을 제공하는 친절한 상담사입니다.
            사용자의 감정에 공감하고, 지지하며, 도움이 되는 관점을 제공하세요.
            구체적인 임상적 진단이나 치료법을 직접 제시하지 마세요.
            필요하면 전문 의료 서비스를 받도록 권장하세요.
            사용자의 상황에 주의 깊게 귀 기울이고, 비판하지 않으며, 존중하는 태도로 응답하세요.
            고통이나 자해 관련 내용이 언급되면, 자살예방 핫라인(1393)을 안내하고 즉시 도움을 구할 것을 권고하세요.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"상담 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 상담 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해주세요." 