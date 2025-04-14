import os
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

# Import from modules
from src.utils.data_loader import load_data
from src.rag.vectorstore import get_vector_store
from src.rag.prompts import PROMPT
from src.models.llm import LLMManager

# Configuration
CSV_PATH = "data/cleaned_pubmed_papers.csv"
VECTOR_STORE_PATH = "vectors/pubmed_vectors"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 다른 포트로 실행하려면 환경 변수 설정: CHAINLIT_PORT=8001 chainlit run app.py

@cl.on_chat_start
async def on_chat_start():
    # Send initial message
    await cl.Message(
        content="🔬 우울증 관련 챗봇에 오신 것을 환영합니다! 학술적 내용이나 상담 관련 질문을 해주세요."
    ).send()
    
    # Load data
    try:
        df = load_data(CSV_PATH)
        cl.logger.info(f"CSV 파일에서 {len(df)}개의 논문 로드됨")
    except Exception as e:
        await cl.Message(content=f"CSV 데이터 로드 실패: {str(e)}").send()
        return
    
    # Set up vector store
    with cl.Step("벡터 저장소 준비 중...") as step:
        try:
            # Set vector_store path in the environment
            os.environ["VECTOR_STORE_PATH"] = VECTOR_STORE_PATH
            vector_store = get_vector_store(df)
            step.output = f"벡터 저장소 준비 완료 (문서 {len(vector_store.index_to_docstore_id)}개)"
        except Exception as e:
            step.output = f"벡터 저장소 준비 실패: {str(e)}"
            await cl.Message(content="벡터 저장소 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Initialize LLM
    with cl.Step("AI 모델 초기화 중...") as step:
        try:
            llm_manager = LLMManager(
                model_name=LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                streaming=True
            )
            llm = llm_manager.llm
            step.output = "AI 모델 초기화 완료"
        except Exception as e:
            step.output = f"AI 모델 초기화 실패: {str(e)}"
            await cl.Message(content="AI 모델 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    # Store chain in user session
    cl.user_session.set("qa_chain", qa_chain)
    cl.user_session.set("retriever", vector_store.as_retriever(search_kwargs={"k": 3}))
    cl.user_session.set("llm_manager", llm_manager)
    
    await cl.Message(content="준비 완료! 질문을 입력해주세요.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get question from user message
    question = message.content
    
    # Get resources from user session
    qa_chain = cl.user_session.get("qa_chain")
    retriever = cl.user_session.get("retriever")
    llm_manager = cl.user_session.get("llm_manager")
    
    if not qa_chain or not retriever or not llm_manager:
        await cl.Message(content="세션이 만료되었습니다. 새로고침 후 다시 시도해주세요.").send()
        return
    
    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    # Custom streaming callback handler
    class ChainlitStreamingHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs):
            cl.run_sync(msg.stream_token(token))
    
    try:
        # 질문 분류
        with cl.Step("질문 분류 중...") as step:
            question_type = llm_manager.classify_question(question)
            step.output = f"질문 유형: {'학술적 질문' if question_type == 'academic' else '상담 질문'}"
            
        # 상담 질문인 경우
        if question_type == "counseling":
            with cl.Step("상담 응답 생성 중...") as step:
                # 상담 질문에 대한 응답 생성
                response = llm_manager.generate_counseling_response(
                    question, 
                    callbacks=[ChainlitStreamingHandler()]
                )
                
                # 스트리밍이 작동하지 않은 경우 메시지 업데이트
                if not msg.content:
                    msg.content = response
                    await msg.update()
                    
                step.output = "상담 응답 생성 완료"
        
        # 학술적 질문인 경우 
        else:
            # Retrieve relevant documents
            with cl.Step("관련 논문 검색 중...") as step:
                try:
                    docs = retriever.get_relevant_documents(question)
                    step.output = f"{len(docs)}개의 관련 논문을 찾았습니다."
                except Exception as e:
                    step.output = f"논문 검색 실패: {str(e)}"
                    await cl.Message(content="논문 검색 중 오류가 발생했습니다.").send()
                    return
            
            # Get context from documents for direct LLM response
            context = "\n\n".join([doc.page_content for doc in docs])
            
            with cl.Step("학술 응답 생성 중...") as step:
                # Use direct LLM call with streaming instead of chain
                # This ensures we see the tokens as they're generated
                response = await qa_chain.ainvoke(
                    {"query": question},
                    {"callbacks": [ChainlitStreamingHandler()]}
                )
                
                # If streaming didn't work, update the message with final result
                if not msg.content:
                    msg.content = response["result"]
                    await msg.update()
                
                step.output = "학술 응답 생성 완료"
                
                # Display source documents for academic questions
                sources_text = "### 참고 논문\n\n"
                for i, doc in enumerate(response["source_documents"]):
                    sources_text += f"**{i+1}.** 논문 ID: {doc.metadata.get('paper_id', 'N/A')}\n"
                    sources_text += f"   제목: {doc.metadata.get('title', 'N/A')}\n"
                    sources_text += f"   저널: {doc.metadata.get('journal', 'N/A')}\n\n"
                
                # Send sources as a new message
                await cl.Message(content=sources_text).send()
    
    except Exception as e:
        await cl.Message(content=f"오류가 발생했습니다: {str(e)}").send()
        msg.content = "응답 생성 중 오류가 발생했습니다."
        await msg.update()
        return
    
    # Update message if it's still empty
    if not msg.content:
        msg.content = "응답 생성이 완료되었지만 내용이 표시되지 않습니다. 다시 시도해주세요."
        await msg.update()