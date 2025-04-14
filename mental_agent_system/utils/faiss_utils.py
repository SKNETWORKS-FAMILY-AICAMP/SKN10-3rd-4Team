import os
import tempfile
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_faiss_from_korean_path(korean_path: str | Path) -> FAISS:
    """
    한글 경로의 FAISS 인덱스를 로드하는 함수
    임시 디렉토리를 사용하여 한글 경로 문제를 우회
    """
    korean_path = Path(korean_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp_index"
        
        # 원본 파일들을 임시 디렉토리로 복사
        shutil.copytree(korean_path, temp_path)
        
        # 임시 디렉토리에서 FAISS 로드
        return FAISS.load_local(
            str(temp_path),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        ) 