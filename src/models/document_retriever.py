from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class DocumentRetriever:
    """문서 검색기 클래스"""
    
    def __init__(self, vector_db_path="data/vectordb"):
        """
        문서 검색기 초기화
        
        Args:
            vector_db_path (str): 벡터 DB 경로
        """
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 벡터 DB가 존재하는지 확인
        try:
            self.vectordb = FAISS.load_local(vector_db_path, self.embeddings)
            print(f"벡터 DB 로드 완료: {vector_db_path}")
        except Exception as e:
            print(f"벡터 DB 로드 실패: {str(e)}")
            self.vectordb = None
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> list:
        """
        쿼리와 관련된 문서 검색
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 검색할 문서 수
            
        Returns:
            list: 관련 문서 리스트
        """
        try:
            if self.vectordb is None:
                print("벡터 DB가 로드되지 않았습니다.")
                return []
                
            # 관련 문서 검색
            docs = self.vectordb.similarity_search(query, k=top_k)
            return docs
        except Exception as e:
            print(f"문서 검색 중 오류 발생: {str(e)}")
            return []
    
    def format_retrieved_documents(self, docs: list) -> str:
        """
        검색된 문서들을 문자열로 포맷팅
        
        Args:
            docs (list): Document 객체 리스트
            
        Returns:
            str: 포맷팅된 문서 내용
        """
        if not docs:
            return "관련 정보를 찾을 수 없습니다."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"문서 {i+1}:\n{doc.page_content}\n")
        
        return "\n".join(formatted_docs) 