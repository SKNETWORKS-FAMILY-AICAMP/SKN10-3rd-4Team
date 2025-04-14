import os
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
import logging

# Import from modules
from src.utils.data_loader import load_data
from src.rag.vectorstore import get_vector_store
from src.models.llm import LLMManager
from src.models.workflow import WorkflowManager
from src.visualization.graph_visualizer import visualize_langgraph_workflow, visualize_simple_langgraph_workflow

# Configuration
CSV_PATH = "data/cleaned_pubmed_papers.csv"
VECTOR_STORE_PATH = "vectors/pubmed_vectors"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 로깅 레벨 설정 - 토큰 스트리밍 중 과도한 로깅 방지
cl.logger.setLevel(logging.INFO)

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
            step.output = "AI 모델 초기화 완료"
        except Exception as e:
            step.output = f"AI 모델 초기화 실패: {str(e)}"
            await cl.Message(content="AI 모델 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Initialize workflow manager
    with cl.Step("워크플로우 초기화 중...") as step:
        try:
            workflow_manager = WorkflowManager(llm_manager, vector_store)
            step.output = "워크플로우 초기화 완료"
            
            # 워크플로우 시각화
            try:
                # 기존 이미지 파일 삭제 (있으면)
                viz_path = "visualization/simple_langgraph_workflow.png"
                if os.path.exists(viz_path):
                    os.remove(viz_path)
                    
                # 새 시각화 이미지 생성
                graph_path = visualize_simple_langgraph_workflow()
                cl.logger.info(f"새 워크플로우 이미지 생성됨: {graph_path}")
                
                elements = [
                    cl.Image(name="workflow_diagram", display="inline", path=graph_path)
                ]
                await cl.Message(content="LangGraph 워크플로우 다이어그램:", elements=elements).send()
            except Exception as e:
                cl.logger.error(f"워크플로우 시각화 실패: {str(e)}")
        except Exception as e:
            step.output = f"워크플로우 초기화 실패: {str(e)}"
            await cl.Message(content="워크플로우 초기화 중 오류가 발생했습니다.").send()
            return
    
    # Store workflow manager in user session
    cl.user_session.set("workflow_manager", workflow_manager)
    
    await cl.Message(content="준비 완료! 질문을 입력해주세요.").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Get question from user message
    question = message.content
    
    # Get workflow manager from user session
    workflow_manager = cl.user_session.get("workflow_manager")
    
    if not workflow_manager:
        await cl.Message(content="세션이 만료되었습니다. 새로고침 후 다시 시도해주세요.").send()
        return
    
    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    # Custom streaming callback handler
    class ChainlitStreamingHandler(BaseCallbackHandler):
        def __init__(self):
            # 스트리밍 응답이 시작되었는지 추적
            self.streaming_started = False
        
        def on_llm_new_token(self, token: str, **kwargs):
            # 스트리밍 시작 표시
            self.streaming_started = True
            # 토큰을 즉시 스트리밍
            cl.run_sync(msg.stream_token(token))
    
    try:
        # 스트리밍 핸들러 생성
        streaming_handler = ChainlitStreamingHandler()
        
        # 워크플로우 실행 변수 초기화
        documents = []
        question_type = "academic"  # 기본값
        last_node = None
        response_content = ""
        response_received = False
        
        # 워크플로우 실행 상태 추적
        classify_done = False
        retrieve_done = False
        generation_started = False
        is_token_streaming = False
        
        # 스트리밍 모드로 워크플로우 실행
        for state, metadata in workflow_manager.stream_process(
            question, 
            callbacks=[streaming_handler]
        ):
            # 현재 노드 이름
            current_node = metadata.get("langgraph_node", "unknown")
            
            # 토큰 스트리밍 중에는 로깅 최소화
            if streaming_handler.streaming_started and is_token_streaming:
                # 토큰 스트리밍 상태 유지
                continue
                
            # 새로운 노드로 전환될 때 상태 업데이트
            if last_node != current_node:
                cl.logger.info(f"노드 전환: {last_node} -> {current_node}")
                last_node = current_node
                
                # 질문 분류 완료
                if current_node == "classify" and not classify_done:
                    classify_done = True
                    if "question_type" in state:
                        question_type = state["question_type"]
                        cl.logger.info(f"질문 유형 분류 완료: {question_type}")
                
                # 문서 검색 완료
                elif current_node == "retrieve" and not retrieve_done:
                    retrieve_done = True
                    if "documents" in state and isinstance(state["documents"], list):
                        documents = state["documents"]
                        cl.logger.info(f"문서 검색 완료: {len(documents)}개 문서 찾음")
                
                # 응답 생성 시작
                elif (current_node == "generate_academic" or current_node == "generate_counseling") and not generation_started:
                    generation_started = True
                    is_token_streaming = True
                    cl.logger.info(f"응답 생성 시작: {current_node}")
            
            # 상태에서 응답 메시지 확인 - 최종 응답 확인
            if isinstance(state, dict) and "messages" in state:
                for msg_item in state["messages"]:
                    if not isinstance(msg_item, HumanMessage) and hasattr(msg_item, "content"):
                        message_content = msg_item.content
                        # 태그에 final이 있는지 확인
                        is_final = hasattr(msg_item, "tags") and "final" in msg_item.tags
                        
                        # 최종 응답이면 토큰 스트리밍 종료 표시
                        if is_final:
                            is_token_streaming = False
                            cl.logger.info("최종 응답 수신됨")
                        
                        # 새 응답 내용이 있는 경우
                        if message_content and message_content != response_content:
                            response_content = message_content
                            response_received = True
                            
                            # 토큰 기반 스트리밍이 이미 응답을 처리 중이면 스킵
                            if not streaming_handler.streaming_started:
                                cl.logger.info("토큰 스트리밍이 시작되지 않아 전체 응답 업데이트")
                                msg.content = message_content
                                await msg.update()
        
        # 토큰 스트리밍이 성공적으로 이루어졌는지 확인
        if streaming_handler.streaming_started:
            cl.logger.info("토큰 스트리밍이 성공적으로 완료됨")
        # 스트리밍이 없었지만 응답이 있는 경우
        elif response_received:
            cl.logger.info("스트리밍 없이 전체 응답이 수신됨")
            # 메시지 이미 업데이트됨
        # 응답이 전혀 없는 경우
        else:
            cl.logger.warning("응답이 수신되지 않음, 워크플로우 응답 직접 실행")
            
            # 응답 생성을 위한 전체 워크플로우 실행 시도
            try:
                # 비스트리밍으로 완전한 응답 가져오기
                result = workflow_manager.process_message(question)
                
                # 응답 찾기
                if isinstance(result, dict) and "messages" in result:
                    found_response = False
                    for msg_item in result["messages"]:
                        if not isinstance(msg_item, HumanMessage) and hasattr(msg_item, "content"):
                            found_response = True
                            msg.content = msg_item.content
                            await msg.update()
                            break
                    
                    if not found_response:
                        msg.content = "응답을 생성하지 못했습니다. 다시 시도해주세요."
                        await msg.update()
                else:
                    msg.content = "응답을 생성하지 못했습니다. 다시 시도해주세요."
                    await msg.update()
            except Exception as e:
                cl.logger.error(f"직접 응답 생성 중 오류: {str(e)}")
                msg.content = "응답을 생성하는 데 문제가 발생했습니다. 다시 시도해주세요."
                await msg.update()
        
        # 메시지가 비어 있는 경우 기본 메시지 설정
        if not msg.content:
            msg.content = "죄송합니다. 응답을 생성하지 못했습니다. 다시 질문해 주세요."
            await msg.update()
        
        # 질문 유형에 따라 추가 정보 표시
        cl.logger.info(f"최종 질문 유형: {question_type}, 문서 수: {len(documents)}")
        
        # 학술 질문의 경우 소스 문서 정보 표시
        if question_type == "academic" and documents:
            sources_text = "### 참고 논문\n\n"
            for i, doc in enumerate(documents[:3]):
                sources_text += f"**{i+1}.** 논문 ID: {doc.metadata.get('paper_id', 'N/A')}\n"
                sources_text += f"   제목: {doc.metadata.get('title', 'N/A')}\n"
                sources_text += f"   저널: {doc.metadata.get('journal', 'N/A')}\n\n"
            
            # 소스 정보를 새 메시지로 전송
            await cl.Message(content=sources_text).send()
    
    except Exception as e:
        cl.logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}", exc_info=True)
        await cl.Message(content=f"오류가 발생했습니다: {str(e)}").send()
        msg.content = "응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        await msg.update()
        return