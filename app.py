import os
import pandas as pd
import chainlit as cl
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Configuration
VECTOR_STORE_PATH = "vectors/pubmed_vectors"
CSV_PATH = "data/cleaned_pubmed_papers.csv"

# Prompt template
PROMPT_TEMPLATE = """
당신은 정신의학 전문가입니다. 제공된 PubMed 논문 내용을 참고하여 질문에 답변해 주세요.

질문: {question}

참고할 논문 내용:
{context}

답변은 다음 형식으로 작성해주세요:
1. 답변 내용을 명확하게 설명
2. 참고한 논문 정보 언급 (제목, 저널)
3. 정보가 불충분한 경우 정직하게 인정
"""

# Load CSV data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    cl.logger.info(f"CSV 파일에서 {len(df)}개의 논문 로드됨")
    return df

# Get or create vector store
def get_vector_store(df):
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH):
        try:
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model="bge-m3",
                base_url="http://localhost:11434"
            )
            
            # Load existing vector store
            vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            
            # Check document count
            doc_count = len(vector_store.index_to_docstore_id)
            cl.logger.info(f"기존 벡터 저장소에서 {doc_count}개 문서 발견")
            
            if doc_count == len(df):
                cl.logger.info("벡터 저장소가 최신 상태입니다. 기존 벡터 저장소를 사용합니다.")
                return vector_store
            else:
                cl.logger.info(f"벡터 저장소({doc_count}개)와 CSV({len(df)}개)의 문서 수가 다릅니다.")
                cl.logger.info("벡터 저장소를 새로 생성합니다.")
        except Exception as e:
            cl.logger.error(f"벡터 저장소 로드 중 오류 발생: {e}")
            cl.logger.info("벡터 저장소를 새로 생성합니다.")
    else:
        cl.logger.info("벡터 저장소가 존재하지 않습니다. 새로 생성합니다.")
    
    # Create new vector store
    cl.logger.info("문서를 처리하여 벡터 저장소 생성 중...")
    
    # Create document objects
    documents = []
    for i, row in df.iterrows():
        content = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "paper_id": row["paper_id"],
                "paper_number": row["paper_number"],
                "journal": row["journal"],
                "title": row["title"]
            }
        )
        documents.append(doc)
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save vector store
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    cl.logger.info(f"{len(documents)}개 문서의 벡터 저장소가 '{VECTOR_STORE_PATH}'에 저장되었습니다.")
    
    return vector_store

@cl.on_chat_start
async def on_chat_start():
    # Send initial message
    await cl.Message(
        content="🔬 PubMed 논문 검색 챗봇에 오신 것을 환영합니다! 정신의학 관련 질문을 해주세요."
    ).send()
    
    # Load data
    try:
        df = load_data(CSV_PATH)
    except Exception as e:
        await cl.Message(content=f"CSV 데이터 로드 실패: {str(e)}").send()
        return
    
    # Set up vector store
    with cl.Step("벡터 저장소 준비 중...") as step:
        try:
            vector_store = get_vector_store(df)
            step.output = f"벡터 저장소 준비 완료 (문서 {len(vector_store.index_to_docstore_id)}개)"
        except Exception as e:
            step.output = f"벡터 저장소 준비 실패: {str(e)}"
            await cl.Message(content="벡터 저장소 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Initialize LLM
    with cl.Step("AI 모델 초기화 중...") as step:
        try:
            llm = Ollama(
                model="gemma3:4b",
                base_url="http://localhost:11434",
                temperature=0.1
            )
            step.output = "AI 모델 초기화 완료"
        except Exception as e:
            step.output = f"AI 모델 초기화 실패: {str(e)}"
            await cl.Message(content="AI 모델 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Create prompt template
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Store chain in user session
    cl.user_session.set("qa_chain", qa_chain)
    
    await cl.Message(content="준비 완료! 질문을 입력해주세요.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get question from user message
    question = message.content
    
    # Get chain from user session
    qa_chain = cl.user_session.get("qa_chain")
    if not qa_chain:
        await cl.Message(content="세션이 만료되었습니다. 새로고침 후 다시 시도해주세요.").send()
        return
    
    # Show thinking message
    thinking_msg = cl.Message(content="생각 중...")
    await thinking_msg.send()
    
    # Process query
    try:
        with cl.Step("논문 검색 및 응답 생성 중...") as step:
            response = qa_chain({"query": question})
            step.output = "응답 생성 완료"
    except Exception as e:
        # Handle error - instead of updating, send a new message
        await cl.Message(content=f"오류가 발생했습니다: {str(e)}").send()
        return
    
    # Instead of updating, send a new message with the result
    await cl.Message(content=response["result"]).send()
    
    # Display source documents
    sources_text = "### 참고 논문\n\n"
    for i, doc in enumerate(response["source_documents"]):
        sources_text += f"**{i+1}.** 논문 ID: {doc.metadata.get('paper_id', 'N/A')}\n"
        sources_text += f"   제목: {doc.metadata.get('title', 'N/A')}\n"
        sources_text += f"   저널: {doc.metadata.get('journal', 'N/A')}\n\n"
    
    # Send sources as a new message
    await cl.Message(content=sources_text).send()