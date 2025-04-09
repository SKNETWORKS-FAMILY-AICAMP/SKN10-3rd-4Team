from langchain.prompts import PromptTemplate

template = """
당신은 정신의학 전문가입니다. 제공된 PubMed 논문 내용을 참고하여 질문에 답변해 주세요.

질문: {question}

참고할 논문 내용:
{context}

답변은 다음 형식으로 작성해주세요:
1. 답변 내용을 명확하게 설명
2. 참고한 논문 정보 언급 (제목, 저널)
3. 정보가 불충분한 경우 정직하게 인정
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)