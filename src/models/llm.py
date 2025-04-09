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
            response = self.llm.generate([prompt], callbacks=callbacks)
            return response.generations[0][0].text
        else:
            response = self.llm(prompt)
            return response

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