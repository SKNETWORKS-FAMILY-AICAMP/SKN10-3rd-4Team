import os
from mental_agent_system.utils.embedding_utils import DataEmbedder

def main():
    # 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    youtube_data_dir = os.path.join(base_dir, "data", "cleaned_youtube_data", "cleaned_data")
    pubmed_data_path = os.path.join(base_dir, "data", "cleaned_pubmed_papers.csv")
    youtube_text_path = os.path.join(base_dir, "data", "cleaned_youtube_text.csv")

    # FAISS 인덱스 저장 경로
    counselor_index_path = os.path.join(base_dir, "vectorstore", "counselor_faiss_index")
    psychiatric_doc_index_path = os.path.join(base_dir, "vectorstore", "psychiatric_doc_faiss_index")
    counselor_index_path_summary = os.path.join(base_dir, "vectorstore", "counselor_faiss_index_summary")
    # 임베더 초기화
    embedder = DataEmbedder()
    
    # 상담가용 유튜브 데이터 임베딩
    print("상담가 데이터 임베딩 시작...")
    embedder.embed_youtube_data(youtube_data_dir, counselor_index_path)

    # 상담가용 유튜브 데이터 임베딩 (요약)
    embedder.embed_youtube_data(youtube_data_dir, counselor_index_path_summary)

    # 정신과의사용 PubMed 데이터 임베딩
    print("\n정신과의사 데이터 임베딩 시작...")
    embedder.embed_pubmed_data(pubmed_data_path, psychiatric_doc_index_path)

if __name__ == "__main__":
    main() 