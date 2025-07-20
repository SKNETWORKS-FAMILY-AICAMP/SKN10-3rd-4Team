"""
이 파일은 더 이상 사용되지 않습니다.
모든 임베딩 관련 코드는 src/rag/embeddings.py로 통합되었습니다.

YoutubeEmbeddings 클래스 대신 src.rag.embeddings.YoutubeEmbeddingManager 클래스를 사용하세요.
"""

# 기존 클래스의 간단한 래퍼(wrapper)를 제공하여 이전 코드와의 호환성 유지
from src.rag.embeddings import YoutubeEmbeddingManager

class YoutubeEmbeddings:
    """
    호환성을 위한 레거시 클래스
    이 클래스는 더 이상 사용되지 않으며 src.rag.embeddings.YoutubeEmbeddingManager를 사용하세요.
    """
    def __init__(self, model_name="bge-m3"):
        import warnings
        warnings.warn(
            "YoutubeEmbeddings 클래스는 더 이상 사용되지 않습니다. "
            "대신 src.rag.embeddings.YoutubeEmbeddingManager를 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )
        self._manager = YoutubeEmbeddingManager(model_name=model_name)
        
    def load_embeddings(self, vector_store_path):
        """벡터 저장소 로드"""
        return self._manager.load_embeddings(vector_store_path)
        
    def similarity_search(self, query, k=3):
        """유사도 검색 수행"""
        return self._manager.similarity_search(query, k=k)
        
    @property
    def vector_store(self):
        """벡터 저장소 접근자"""
        return self._manager.vector_store
        
    @vector_store.setter
    def vector_store(self, value):
        """벡터 저장소 설정자"""
        self._manager.vector_store = value 