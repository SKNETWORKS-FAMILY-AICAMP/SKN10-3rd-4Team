import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

class DataEmbedder:
    def __init__(self):
        """
        데이터 임베딩을 위한 유틸리티 클래스
        """
        # .env 파일 로드
        load_dotenv()
        
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다. "
                ".env 파일을 확인해주세요."
            )
        
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    def _create_documents_from_youtube_data(self, df: pd.DataFrame) -> List[Document]:
        """유튜브 데이터를 Document 형식으로 변환"""
        documents = []
        for _, row in df.iterrows():
            try:
                # text 컬럼이 있는 경우 (요약 데이터)
                if 'summary' in df.columns:
                    text = row['summary']
                    metadata = {"source": row['video_url'], "title": row['title']}
                # caption이 있는 경우 (원본 데이터)
                else:
                    text = f"제목: {row['title']}\n내용: {row['caption']}"
                    metadata = {"source": row['video_url'], "title": row['title']}
                
                documents.append(Document(page_content=text, metadata=metadata))
            except KeyError as e:
                print(f"경고: 누락된 컬럼 - {e}")
                print(f"사용 가능한 컬럼:", df.columns.tolist())
                continue
        return documents

    def _create_documents_from_pubmed_data(self, df: pd.DataFrame) -> List[Document]:
        """PubMed 데이터를 Document 형식으로 변환"""
        documents = []
        for _, row in df.iterrows():
            try:
                text = f"제목: {row['title']}\n초록: {row['abstract']}\n내용: {row['content']}"
                metadata = {"source": str(row['paper_id']), "title": row['title']}
                documents.append(Document(page_content=text, metadata=metadata))
            except KeyError as e:
                print(f"경고: 누락된 컬럼 - {e}")
                print(f"사용 가능한 컬럼:", df.columns.tolist())
                continue
        return documents

    def _save_faiss_index(self, vectorstore: FAISS, final_path: str):
        """FAISS 인덱스를 임시 디렉토리에 저장 후 최종 위치로 이동"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 임시 디렉토리에 저장
            vectorstore.save_local(temp_dir)
            
            # 최종 디렉토리 생성
            os.makedirs(final_path, exist_ok=True)
            
            # 임시 디렉토리의 파일들을 최종 위치로 복사
            for file in os.listdir(temp_dir):
                src = os.path.join(temp_dir, file)
                dst = os.path.join(final_path, file)
                shutil.copy2(src, dst)

    def embed_youtube_data(self, data_dir: str, save_path: str):
        """
        유튜브 데이터를 임베딩하고 FAISS 인덱스로 저장
        Args:
            data_dir: 정제된 유튜브 데이터가 있는 디렉토리 경로
            save_path: FAISS 인덱스를 저장할 경로
        """
        all_documents = []
        
        # 디렉토리 내의 모든 CSV 파일 처리
        for file in os.listdir(data_dir):
            if file.startswith("cleaned_") and file.endswith(".csv"):
                file_path = os.path.join(data_dir, file)
                print(f"\n처리 중인 파일: {file}")
                try:
                    df = pd.read_csv(file_path)
                    documents = self._create_documents_from_youtube_data(df)
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"파일 처리 중 오류 발생 ({file}): {e}")
                    continue

        if not all_documents:
            raise ValueError("처리할 수 있는 문서가 없습니다.")

        # 텍스트 분할
        splits = self.text_splitter.split_documents(all_documents)
        
        # FAISS 인덱스 생성 및 저장
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        self._save_faiss_index(vectorstore, save_path)
        print(f"임베딩 완료: {len(splits)}개의 문서 청크가 {save_path}에 저장되었습니다.")

    def embed_pubmed_data(self, data_path: str, save_path: str):
        """
        PubMed 데이터를 임베딩하고 FAISS 인덱스로 저장
        Args:
            data_path: PubMed 데이터 CSV 파일 경로
            save_path: FAISS 인덱스를 저장할 경로
        """
        try:
            df = pd.read_csv(data_path)
            documents = self._create_documents_from_pubmed_data(df)
            
            if not documents:
                raise ValueError("처리할 수 있는 문서가 없습니다.")

            # 텍스트 분할
            splits = self.text_splitter.split_documents(documents)
            
            # FAISS 인덱스 생성 및 저장
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            self._save_faiss_index(vectorstore, save_path)
            print(f"임베딩 완료: {len(splits)}개의 문서 청크가 {save_path}에 저장되었습니다.")
        except Exception as e:
            print(f"PubMed 데이터 처리 중 오류 발생: {e}")
            raise 

    def embed_single_youtube_file(self, file_path: str, save_path: str):
        """
        단일 유튜브 데이터 CSV 파일을 임베딩하고 FAISS 인덱스로 저장
        Args:
            file_path: 정제된 유튜브 데이터 CSV 파일 경로
            save_path: FAISS 인덱스를 저장할 경로
        """
        try:
            print(f"\n처리 중인 파일: {file_path}")
            df = pd.read_csv(file_path)
            documents = self._create_documents_from_youtube_data(df)
            
            if not documents:
                raise ValueError("처리할 수 있는 문서가 없습니다.")

            # 텍스트 분할
            splits = self.text_splitter.split_documents(documents)
            
            # FAISS 인덱스 생성 및 저장
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            self._save_faiss_index(vectorstore, save_path)
            print(f"임베딩 완료: {len(splits)}개의 문서 청크가 {save_path}에 저장되었습니다.")
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")
            raise 