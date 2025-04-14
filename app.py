import os
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage

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
        def on_llm_new_token(self, token: str, **kwargs):
            cl.run_sync(msg.stream_token(token))
    
    try:
        # 워크플로우 실행 (스트리밍 처리)
        last_state = None
        documents = []
        question_type = "academic"  # 기본값
        
        # 스트리밍 모드로 워크플로우 실행
        for state, metadata in workflow_manager.stream_process(
            question, 
            callbacks=[ChainlitStreamingHandler()]
        ):
            # 노드 이름
            node = metadata.get("langgraph_node", "unknown")
            cl.logger.info(f"현재 노드: {node}")
            
            # 워크플로우 진행 중에 정보 추출
            # 문서 저장 (retrieve 노드에서)
            if "documents" in state and isinstance(state["documents"], list):
                documents = state["documents"]
            
            # 질문 유형 저장
            if "question_type" in state:
                question_type = state["question_type"]
                cl.logger.info(f"질문 유형 갱신: {question_type}")
                
            # 상태 업데이트
            last_state = state
                
        # 응답 메시지가 스트리밍되지 않은 경우 최종 상태에서 응답 업데이트
        if not msg.content and last_state:
            # 메시지 찾기
            ai_message_found = False
            for message in last_state.get("messages", []):
                if hasattr(message, "type") and message.type == "ai":
                    ai_message_found = True
                    msg.content = message.content
                    await msg.update()
                    break
                    
                # AIMessage의 경우 확인
                if not isinstance(message, HumanMessage) and hasattr(message, "content"):
                    ai_message_found = True
                    msg.content = message.content
                    await msg.update()
                    break
                    
            # 메시지 객체로부터 직접 내용 가져오기 시도
            if not ai_message_found and last_state.get("messages"):
                for message in last_state.get("messages"):
                    if hasattr(message, "content") and not isinstance(message, HumanMessage):
                        msg.content = message.content
                        await msg.update()
                        break
            
            # 여전히 내용이 없으면 기본 메시지 설정
            if not msg.content:
                msg.content = "응답을 생성했지만 내용을 표시할 수 없습니다."
                await msg.update()
                
        # 최종 질문 유형과 결과 로깅
        cl.logger.info(f"최종 질문 유형: {question_type}, 문서 수: {len(documents)}")
        
        # 학술 질문에 대해서만 소스 문서 표시
        if question_type == "academic" and documents:
            sources_text = "### 참고 논문\n\n"
            for i, doc in enumerate(documents[:3]):
                sources_text += f"**{i+1}.** 논문 ID: {doc.metadata.get('paper_id', 'N/A')}\n"
                sources_text += f"   제목: {doc.metadata.get('title', 'N/A')}\n"
                sources_text += f"   저널: {doc.metadata.get('journal', 'N/A')}\n\n"
            
            # 소스 정보를 새 메시지로 전송
            await cl.Message(content=sources_text).send()
    
    except Exception as e:
        cl.logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}")
        await cl.Message(content=f"오류가 발생했습니다: {str(e)}").send()
        msg.content = "응답 생성 중 오류가 발생했습니다."
        await msg.update()
        return
    
    # 메시지가 여전히 비어있으면 업데이트
    if not msg.content:
        msg.content = "응답 생성이 완료되었지만 내용이 표시되지 않습니다. 다시 시도해주세요."
        await msg.update()