import os
from langchain_community.llms import Ollama

class LLMManager:
    def __init__(self, model_name, base_url, temperature=0.1, streaming=False):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.streaming = streaming
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        # Initialize without streaming parameter
        # Streaming will be handled via callbacks when invoking
        return Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

    def generate_response(self, question, context, callbacks=None):
        prompt = self.create_prompt(question, context)
        # Pass callbacks for streaming if provided
        if self.streaming and callbacks:
            try:
                # 직접 콜백을 사용한 스트리밍 처리
                print("스트리밍 모드로 학술 응답 생성 중...")
                # invoke() 방식으로 호출 (올바른 콜백 인자 전달)
                response = self.llm.invoke(prompt, config={"callbacks": callbacks})
                return response
            except Exception as e:
                print(f"스트리밍 응답 생성 중 오류: {str(e)}")
                # 오류 발생 시 비스트리밍 모드로 폴백
                response = self.llm(prompt)
                return response
        else:
            response = self.llm(prompt)
            return response

    def classify_question(self, question):
        """
        질문을 학술적인 내용인지 상담 내용인지 분류합니다.
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            str: "academic" 또는 "counseling"
        """
        prompt = """
        다음 질문이 우울증에 대한 학술적/연구적 내용을 묻는 것인지, 
        아니면 우울증 관련 상담이나 개인적인 조언을 구하는 것인지 판단해주세요.
        
        질문: {question}
        
        분류 기준:
        - '학술적(academic)': 우울증의 원인, 증상, 치료법, 통계, 연구 결과, 약물, 치료법 등 객관적 정보를 요청하는 경우
        - '상담(counseling)': 개인적인 우울함, 감정적 어려움, 심리적 조언, 대처 방법 등을 구하는 경우
        
        다음 중 하나로만 응답해 주세요: 'academic' 또는 'counseling'
        """.format(question=question)
        
        try:
            response = self.llm(prompt)
            result = response.strip().lower()
            
            # 응답에 academic이 포함되어 있으면 academic으로, 아니면 counseling으로 간주
            if "academic" in result:
                return "academic"
            else:
                return "counseling"
        except Exception as e:
            print(f"질문 분류 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본값으로 학술적 내용으로 분류
            return "academic"

    def generate_counseling_response(self, question, callbacks=None):
        """
        상담 질문에 대한 응답을 생성합니다.
        
        Args:
            question (str): 사용자 질문
            callbacks: 콜백 핸들러
            
        Returns:
            str: 상담 응답
        """
        prompt = self.create_counseling_prompt(question)
        
        # Pass callbacks for streaming if provided
        if self.streaming and callbacks:
            try:
                # 직접 콜백을 사용한 스트리밍 처리
                print("스트리밍 모드로 상담 응답 생성 중...")
                # invoke() 방식으로 호출 (올바른 콜백 인자 전달)
                response = self.llm.invoke(prompt, config={"callbacks": callbacks})
                return response
            except Exception as e:
                print(f"스트리밍 응답 생성 중 오류: {str(e)}")
                # 오류 발생 시 비스트리밍 모드로 폴백
                response = self.llm(prompt)
                return response
        else:
            response = self.llm(prompt)
            return response

    def create_counseling_prompt(self, question):
        """
        상담 질문에 대한 프롬프트를 생성합니다.
        """
        template = """
        당신은 우울증 상담 전문가입니다. 사용자의 우울한 감정과 고민에 공감하고 도움이 되는 조언을 제공해 주세요.

        사용자 메시지: {question}

        상담 지침:
        1. 사용자의 감정에 충분히 공감하세요
        2. 판단하지 말고 경청하는 태도를 보여주세요
        3. 구체적이고 실행 가능한 조언을 제공하세요
        4. 필요하다면 전문적인 상담을 권유하세요
        5. 단, 의학적 진단이나 처방은 제공하지 마세요
        6. 자살/자해 관련 내용이 언급되면 즉시 전문가 상담을 권유하세요

        따뜻하고 공감적인 한국어로 응답해 주세요.
        """

        return template.format(question=question)

    def create_prompt(self, question, context):
        template = """
        당신은 정신의학 전문가입니다. 제공된 PubMed 논문들 중에서 질문과 가장 관련성이 높은 논문들만 선별하여 답변해 주세요.

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

        return template.format(question=question, context=context)