import os
from langchain_community.llms import Ollama
from typing import List, Optional, Generator, Literal, Dict, Any, Union
from langchain.callbacks.base import BaseCallbackHandler
import time
import random
from src.rag.embeddings import YoutubeEmbeddingManager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

class LLMManager:
    def __init__(self, model_name="llama2", base_url="http://localhost:11434", temperature=0.1, streaming=False):
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
        """컨텍스트 기반 응답 생성"""
        # 문서 리스트가 전달된 경우
        if isinstance(context, list):
            context = self.format_academic_context(context)
            
        prompt = self.create_prompt(question, context)
        try:
            # 콜백 처리 설정
            llm = self.llm
            if callbacks:
                llm = Ollama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    callbacks=callbacks
                )
            
            # 응답 생성
            response = llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return f"죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다: {str(e)}"

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

    def generate_counseling_response(self, question, context=None, callbacks=None):
        """
        상담 질문에 대한 응답을 생성합니다.
        
        Args:
            question (str): 사용자 질문
            context (str, optional): 유튜브 컨텍스트
            callbacks: 콜백 핸들러
            
        Returns:
            str: 상담 응답
        """
        prompt = self.create_counseling_prompt(question, context)
        
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

    def create_counseling_prompt(self, question, context=None):
        """
        상담 질문에 대한 프롬프트를 생성합니다.
        
        Args:
            question (str): 사용자 질문
            context (str, optional): 유튜브 컨텍스트
        """
        template = """
        당신은 우울증 상담 전문가입니다. 사용자의 우울한 감정과 고민에 공감하고 도움이 되는 조언을 제공해 주세요.

        사용자 메시지: {question}
        {context_section}
        상담 지침:
        1. 사용자의 감정에 충분히 공감하세요
        2. 판단하지 말고 경청하는 태도를 보여주세요
        3. 구체적이고 실행 가능한 조언을 제공하세요
        4. 필요하다면 전문적인 상담을 권유하세요
        5. 단, 의학적 진단이나 처방은 제공하지 마세요
        6. 자살/자해 관련 내용이 언급되면 즉시 전문가 상담을 권유하세요
        7. 제공된 유튜브 컨텍스트가 있다면, 이를 참고하여 더 구체적이고 실질적인 조언을 제공하세요

        따뜻하고 공감적인 한국어로 응답해 주세요.
        """

        context_section = f"\n\n참고할 유튜브 컨텍스트:\n{context}\n" if context else ""
        return template.format(question=question, context_section=context_section)

    def create_prompt(self, question, context):
        template = """
        당신은 정신의학 전문가입니다. 제공된 PubMed 논문들과 연구 자료들을 분석하여 질문에 가장 관련성이 높은 자료만 선별하여 답변해 주세요.

        질문: {question}

        참고할 논문 정보:
        {context}

        답변 작성 과정:
        1. 제공된 논문들 중 질문과 직접적인 관련이 있는 논문들만 선별하세요
        2. 관련성이 낮거나 질문에 도움이 되지 않는 논문은 분석에서 제외하세요
        3. 선별한 논문들의 정보(제목, 저널, ID 등)를 명확히 확인하세요
        4. 선별한 논문들의 정보를 종합하여 하나의 일관된 답변을 작성하세요

        답변 작성 지침:
        1. 답변 시작 부분에 참고한 논문의 제목들을 명확히 언급하세요 (예: "~라는 제목의 논문에 따르면...")
        2. 선별한 논문들 간의 공통점과 차이점을 파악하여 분석하세요
        3. 각 논문의 핵심 발견과 결론을 통합적으로 설명하세요
        4. 답변 내용을 명확하게 설명하고 논리적으로 구성하세요
        5. 선별한 논문의 메타데이터(제목, 저널, 저자 등)를 반드시 포함하여 언급하세요
        6. 정보가 불충분하거나 논문 간 상충되는 내용이 있는 경우 정직하게 인정하세요
        7. 웹 검색 결과가 포함된 경우, 논문 정보와 웹 검색 결과를 적절히 구분하여 활용하세요

        한국어로 명확하고 전문적인 답변을 제공해 주세요.
        """

        return template.format(question=question, context=context)

    def format_academic_context(self, documents):
        """
        논문 문서들을 구조화된 형식으로 변환합니다.
        
        Args:
            documents (List[Document]): 검색된 논문 문서 리스트
            
        Returns:
            str: 구조화된 문맥 정보
        """
        formatted_context = ""
        
        # 소스 정보가 있으면 먼저 추가
        source_info = None
        for doc in documents:
            if "source_info" in doc.metadata:
                source_info = doc.metadata["source_info"]
                break
                
        if source_info:
            formatted_context += f"{source_info}\n\n"
            
        # PubMed 논문과 웹 검색 결과를 분리
        pubmed_docs = []
        web_docs = []
        
        for doc in documents:
            source = doc.metadata.get('source', '')
            if source == "Tavily" or source == "Tavily Summary":
                web_docs.append(doc)
            else:
                pubmed_docs.append(doc)
        
        # 논문 정보 포맷팅
        if pubmed_docs:
            formatted_context += "== PubMed 논문 정보 ==\n\n"
            
            for i, doc in enumerate(pubmed_docs):
                title = doc.metadata.get('title', '제목 없음')
                paper_id = doc.metadata.get('paper_id', 'N/A')
                journal = doc.metadata.get('journal', 'N/A')
                pmid = doc.metadata.get('pmid', 'N/A')
                
                formatted_context += f"[논문 {i+1}]\n"
                formatted_context += f"제목: {title}\n"
                
                if pmid != 'N/A':
                    formatted_context += f"PMID: {pmid}\n"
                elif paper_id != 'N/A':
                    formatted_context += f"논문 ID: {paper_id}\n"
                    
                if journal != 'N/A':
                    formatted_context += f"저널: {journal}\n"
                
                # 본문 내용 추가
                formatted_context += f"내용:\n{doc.page_content}\n\n"
        
        # 웹 검색 결과 포맷팅
        if web_docs:
            if pubmed_docs:  # 논문 정보가 있었다면 구분선 추가
                formatted_context += "== 웹 검색 결과 ==\n\n"
            
            for i, doc in enumerate(web_docs):
                title = doc.metadata.get('title', '검색 결과')
                source = doc.metadata.get('source', 'Tavily')
                
                formatted_context += f"[검색 결과 {i+1}]\n"
                formatted_context += f"제목: {title}\n"
                
                url = doc.metadata.get('url', '')
                if url:
                    formatted_context += f"URL: {url}\n"
                
                # 본문 내용 추가
                formatted_context += f"내용:\n{doc.page_content}\n\n"
        
        return formatted_context

    def stream_response(
        self,
        question: str,
        context: str = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        response_type: str = "academic",
        model_name: str = None
    ) -> Generator[str, None, None]:
        """
        응답을 토큰 단위로 스트리밍합니다.
        
        Args:
            question (str): 사용자 질문
            context (str, optional): 응답 생성을 위한 컨텍스트
            callbacks (Optional[List[BaseCallbackHandler]]): 콜백 핸들러 리스트
            response_type (str): 응답 유형 ("academic" 또는 "counseling")
            model_name (str): 사용할 모델 이름 (None이면 현재 설정된 모델 사용)
            
        Yields:
            str: 응답 토큰
        """
        try:
            # 모델 이름 설정
            if model_name is None:
                model_name = self.model_name
                
            # 프롬프트 생성
            if response_type == "counseling":
                prompt = self.create_counseling_prompt(question, context)
            else:  # academic
                prompt = self.create_prompt(question, context)
            
            # 토큰 스트리밍 핸들러 설정
            token_handler = callbacks if callbacks else []
            
            # Ollama 설정
            llm = Ollama(
                model=model_name,
                callbacks=token_handler,
                streaming=True
            )
            
            # 응답 생성 및 스트리밍
            for token in llm.stream(prompt):
                time.sleep(0.02)  # 자연스러운 스트리밍을 위한 지연
                yield token
                
        except Exception as e:
            error_type = "상담" if response_type == "counseling" else "학술"
            print(f"{error_type} 응답 스트리밍 중 오류 발생: {str(e)}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    def stream_response_tokens(
        self,
        question: str,
        context: str = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        response_type: Literal["academic", "counseling"] = "academic",
        youtube_context: bool = True
    ) -> Generator[str, None, None]:
        """
        질문 유형에 따라 적절한 응답을 토큰 단위로 스트리밍합니다.
        
        Args:
            question (str): 사용자 질문
            context (str, optional): 논문 컨텍스트 (counseling 유형일 때는 무시됨)
            callbacks (Optional[List[BaseCallbackHandler]]): 콜백 핸들러 리스트
            response_type (str): 응답 유형 ("academic" 또는 "counseling")
            youtube_context (bool): 상담 응답시 유튜브 컨텍스트 사용 여부
            
        Yields:
            str: 응답 토큰
        """
        try:
            # 학술 질문인 경우
            if response_type == "academic":
                if not context:
                    yield "질문에 관련된 논문 컨텍스트가 필요합니다."
                    return
                
                # 학술 응답 스트리밍
                for token in self.stream_response(
                    question=question,
                    context=context, 
                    callbacks=callbacks,
                    response_type="academic",
                    model_name="llama2"
                ):
                    yield token
                    
            # 상담 질문인 경우
            else:
                context_text = ""
                if youtube_context:
                    try:
                        # 유튜브 벡터 저장소에서 관련 컨텍스트 검색
                        youtube_embeddings = YoutubeEmbeddingManager(model_name="bge-m3")
                        
                        # 프로젝트 루트 경로 확인
                        project_root = Path(__file__).parent.parent.parent
                        vector_store_path = str(project_root / "vectors" / "youtube_vectors")
                        
                        # 벡터 저장소 로드
                        vector_store = youtube_embeddings.load_embeddings(vector_store_path)
                        
                        # 컨텍스트 검색
                        results = youtube_embeddings.similarity_search(question, k=2)
                        
                        # 컨텍스트 구성
                        context_text = "\n\n".join([doc.page_content for doc in results])
                        print(f"유튜브 컨텍스트 {len(results)}개 검색 완료")
                    except Exception as e:
                        print(f"유튜브 컨텍스트 검색 중 오류: {str(e)}")
                        context_text = ""
                
                # 상담 응답 스트리밍
                for token in self.stream_response(
                    question=question,
                    context=context_text, 
                    callbacks=callbacks,
                    response_type="counseling",
                    model_name="llama2"
                ):
                    yield token
                    
        except Exception as e:
            error_type = "상담" if response_type == "counseling" else "학술"
            print(f"{error_type} 응답 스트리밍 중 오류 발생: {str(e)}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

# 모듈 수준에서 호출할 수 있는 함수들
def stream_response_tokens(
    question: str,
    context: str = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    response_type: Literal["academic", "counseling"] = "academic",
    youtube_context: bool = True
) -> Generator[str, None, None]:
    """
    질문 유형에 따라 적절한 응답을 토큰 단위로 스트리밍합니다.
    
    Args:
        question (str): 사용자 질문
        context (str, optional): 논문 컨텍스트 (counseling 유형일 때는 무시됨)
        callbacks (Optional[List[BaseCallbackHandler]]): 콜백 핸들러 리스트
        response_type (str): 응답 유형 ("academic" 또는 "counseling")
        youtube_context (bool): 상담 응답시 유튜브 컨텍스트 사용 여부
        
    Yields:
        str: 응답 토큰
    """
    # LLMManager 인스턴스 생성 및 클래스 메서드 호출
    model_name = "llama2" if response_type == "academic" else "bge-m3"
    llm_manager = LLMManager(model_name=model_name, base_url="http://localhost:11434")
    
    # 클래스 내부 메서드로 위임
    for token in llm_manager.stream_response_tokens(
        question=question, 
        context=context, 
        callbacks=callbacks, 
        response_type=response_type, 
        youtube_context=youtube_context
    ):
        yield token

# 하위 호환성을 위한 별칭 함수들
def stream_paper_response_tokens(
    question: str,
    context: str,
    callbacks: Optional[List[BaseCallbackHandler]] = None
) -> Generator[str, None, None]:
    """
    논문 기반 응답을 토큰 단위로 스트리밍합니다 (stream_response_tokens의 별칭).
    
    Args:
        question (str): 사용자의 학술적 질문
        context (str): 관련 논문 컨텍스트
        callbacks (Optional[List[BaseCallbackHandler]]): 콜백 핸들러 리스트
        
    Yields:
        str: 응답 토큰
    """
    return stream_response_tokens(
        question=question,
        context=context,
        callbacks=callbacks,
        response_type="academic",
        youtube_context=False
    )

def stream_counseling_response_tokens(
    question: str,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    youtube_context: bool = True
) -> Generator[str, None, None]:
    """
    상담 응답을 토큰 단위로 스트리밍합니다 (stream_response_tokens의 별칭).
    
    Args:
        question (str): 내담자의 질문
        callbacks (Optional[List[BaseCallbackHandler]]): 콜백 핸들러 리스트
        youtube_context (bool): 유튜브 컨텍스트 사용 여부
        
    Yields:
        str: 응답 토큰
    """
    return stream_response_tokens(
        question=question,
        context=None,
        callbacks=callbacks,
        response_type="counseling",
        youtube_context=youtube_context
    )