import os
import sys
import logging
import chainlit as cl
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain.globals import set_verbose
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv

# src 디렉토리를 시스템 경로에 추가
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

# 환경 변수 로드
load_dotenv()

# LangSmith 설정 확인 및 활성화
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    print("LangSmith 추적이 활성화되어 있습니다.")
    set_verbose(True)  # 자세한 로깅 활성화

# Tavily API 키 확인
tavily_api_key = os.environ.get("TAVILY_API_KEY")
if tavily_api_key:
    print("Tavily API 키가 설정되었습니다.")
else:
    print("경고: Tavily API 키가 설정되지 않았습니다. 웹 검색 기능이 작동하지 않을 수 있습니다.")

# Import from modules
from src.utils.data_loader import load_data
from src.rag.vectorstore import get_vector_store
from src.models.llm import LLMManager
from src.models.workflow import WorkflowManager
from src.visualization.graph_visualizer import visualize_langgraph_workflow,  visualize_current_workflow


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
    
    # Set up vector stores
    with cl.Step("벡터 저장소 준비 중...") as step:
        try:
            # Set vector_store path in the environment
            os.environ["VECTOR_STORE_PATH"] = VECTOR_STORE_PATH
            
            # 논문 벡터 저장소 로드
            vector_store = get_vector_store(df)
            cl.logger.info(f"논문 벡터 저장소 준비 완료 (문서 {len(vector_store.index_to_docstore_id)}개)")
            
            # 유튜브 벡터 저장소 로드
            youtube_vector_path = "vectors/youtube_vectors"
            if os.path.exists(youtube_vector_path):
                from src.utils.youtube_embeddings import YoutubeEmbeddings
                youtube_embeddings = YoutubeEmbeddings()
                youtube_vector_store = youtube_embeddings.load_embeddings(youtube_vector_path)
                cl.logger.info(f"유튜브 벡터 저장소 준비 완료")
            else:
                cl.logger.warning("유튜브 벡터 저장소를 찾을 수 없습니다.")
                youtube_vector_store = None
                
            step.output = f"벡터 저장소 준비 완료 (논문: {len(vector_store.index_to_docstore_id)}개, 유튜브: {'사용 가능' if youtube_vector_store else '없음'})"
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
                # 전체 경로로 이미지 파일 경로 지정
                project_root = os.getcwd()
                viz_dir = os.path.join(project_root, "visualization")
                viz_path = os.path.join(viz_dir, "langgraph_workflow.png")
                
                # 디렉토리가 없으면 생성
                if not os.path.exists(viz_dir):
                    os.makedirs(viz_dir, exist_ok=True)
                    cl.logger.info(f"visualization 디렉토리 생성됨: {viz_dir}")
                
                # 이미지가 이미 존재하면 새로 생성하지 않고 기존 이미지 사용
                if os.path.exists(viz_path) and os.path.getsize(viz_path) > 0:
                    cl.logger.info(f"기존 워크플로우 이미지 사용: {viz_path}")
                    graph_path = viz_path
                else:
                    # 이미지가 없거나 비어있는 경우 LangGraph 내장 시각화 기능 사용
                    cl.logger.info(f"LangGraph 내장 시각화 기능으로 워크플로우 이미지 생성 시작...")
                    graph_path = workflow_manager.visualize_workflow(viz_path)
                    if graph_path:
                        cl.logger.info(f"새 워크플로우 이미지 생성됨: {graph_path}")
                    else:
                        # 시각화 실패 시 기존 방식으로 시도
                        cl.logger.warning("LangGraph 내장 시각화 실패, NetworkX 방식으로 시도")
                        graph_path = visualize_current_workflow()
                        cl.logger.info(f"NetworkX로 새 워크플로우 이미지 생성됨: {graph_path}")
                
                # 이미지 파일 존재 확인
                if os.path.exists(graph_path):
                    # 이미지 표시
                    elements = [
                        cl.Image(name="workflow_diagram", display="inline", path=graph_path)
                    ]
                    await cl.Message(content="LangGraph 워크플로우 다이어그램:", elements=elements).send()
                    
                    # HTML 파일도 있는지 확인하여 링크 제공
                    html_path = graph_path.replace(".png", ".html")
                    if os.path.exists(html_path):
                        # HTML 파일 경로에서 상대 URL 생성
                        html_rel_path = os.path.relpath(html_path, project_root)
                        await cl.Message(content=f"[인터랙티브 워크플로우 다이어그램]({html_rel_path})").send()
                else:
                    cl.logger.error(f"이미지 파일이 존재하지 않음: {graph_path}")
                    await cl.Message(content="워크플로우 다이어그램을 표시할 수 없습니다.").send()
            except Exception as e:
                cl.logger.error(f"워크플로우 시각화 실패: {str(e)}")
                await cl.Message(content=f"워크플로우 시각화 중 오류: {str(e)}").send()
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
        def __init__(self, message):
            self.message = message
            self.token_buffer = ""
            self.is_active = True
            print("스트리밍 핸들러 초기화됨")
        
        def on_llm_start(self, serialized, prompts, **kwargs):
            print(f"LLM 스트리밍 시작: {len(prompts)} 프롬프트 처리 중")
        
        def on_llm_new_token(self, token: str, **kwargs):
            if not self.is_active:
                return
                
            try:
                if token:
                    #print(f"토큰 수신: {token[:10]}{'...' if len(token) > 10 else ''}")
                    self.token_buffer += token
                    # 버퍼가 충분히 찼거나 특정 문자가 있으면 즉시 전송
                    if len(self.token_buffer) > 10 or any(c in self.token_buffer for c in ['\n', '.', '!', '?']):
                        cl.run_sync(self.message.stream_token(self.token_buffer))
                        self.token_buffer = ""
            except Exception as e:
                print(f"스트리밍 토큰 처리 중 오류: {str(e)}")
                self.is_active = False
        
        def on_llm_end(self, response, **kwargs):
            # 버퍼에 남은 토큰 모두 전송
            if self.token_buffer:
                try:
                    cl.run_sync(self.message.stream_token(self.token_buffer))
                except Exception as e:
                    print(f"마지막 토큰 전송 중 오류: {str(e)}")
            print("LLM 스트리밍 완료")
        
        def on_llm_error(self, error, **kwargs):
            print(f"LLM 오류 발생: {str(error)}")
            self.is_active = False
    
    try:
        # 워크플로우 실행 (스트리밍 처리)
        last_state = None
        documents = []
        question_type = "academic"  # 기본값
        workflow_run_id = None      # 워크플로우 실행 ID 추적용
        
        # 스트리밍 핸들러 생성
        streaming_handler = ChainlitStreamingHandler(msg)
        
        # 스트리밍 모드로 워크플로우 실행
        for state, metadata in workflow_manager.stream_process(
            question, 
            callbacks=[streaming_handler]
        ):
            # 노드 이름
            node = metadata.get("langgraph_node", "unknown")
            
            # 워크플로우 ID 추적 (LangSmith 통합을 위해)
            if workflow_run_id is None and "workflow_run_id" in metadata:
                workflow_run_id = metadata["workflow_run_id"]
                cl.logger.info(f"워크플로우 실행 ID: {workflow_run_id}")
            
            # 단계 정보 로깅
            cl.logger.info(f"워크플로우 단계: {node}, Run ID: {metadata.get('workflow_run_id', 'N/A')}")
            
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
        cl.logger.info(f"최종 질문 유형: {question_type}, 문서 수: {len(documents)}, 워크플로우 ID: {workflow_run_id}")
        
        # 학술 질문에 대해서만 소스 문서 표시
        if question_type == "academic" and documents:
            # 타빌리 검색 결과가 있는지 확인 (소스가 "Tavily"인 문서)
            tavily_docs = [doc for doc in documents if doc.metadata.get("source") == "Tavily" or doc.metadata.get("source") == "Tavily Summary"]
            pubmed_docs = [doc for doc in documents if doc.metadata.get("source") != "Tavily" and doc.metadata.get("source") != "Tavily Summary"]
            
            # 실제 논문 소스가 있는 경우에만 참고 논문 표시
            if pubmed_docs and len(pubmed_docs) > 0:
                sources_text = "### 참고 논문\n\n"
                for i, doc in enumerate(pubmed_docs[:3]):
                    sources_text += f"**{i+1}.** 논문 ID: {doc.metadata.get('paper_id', 'N/A')}\n"
                    sources_text += f"   제목: {doc.metadata.get('title', 'N/A')}\n"
                    sources_text += f"   저널: {doc.metadata.get('journal', 'N/A')}\n\n"
                
                # 소스 정보를 새 메시지로 전송
                sources_msg = cl.Message(content=sources_text)
                if workflow_run_id:
                    sources_msg.metadata = {"parent_run_id": workflow_run_id, "type": "sources"}
                await sources_msg.send()
            
            # 타빌리 검색 결과 표시
            if tavily_docs:
                # 타빌리만 사용한 경우 설명 추가
                if not pubmed_docs:
                    intro_text = "### 웹 검색 결과\n\n관련 논문이 데이터베이스에 없어 웹 검색으로 정보를 찾았습니다.\n\n"
                else:
                    intro_text = "### 웹 검색 결과\n\n"
                
                tavily_text = intro_text
                
                # 타빌리 요약을 먼저 표시
                summary_docs = [doc for doc in tavily_docs if doc.metadata.get("title") == "타빌리 요약"]
                search_result_docs = [doc for doc in tavily_docs if doc.metadata.get("title") != "타빌리 요약"]
                
                # 요약 정보 표시
                if summary_docs:
                    summary_doc = summary_docs[0]
                    tavily_text += f"**AI 요약:** {summary_doc.page_content}\n\n"
                    if search_result_docs:
                        tavily_text += "**검색 결과:**\n\n"
                
                # 검색 결과 표시
                for i, doc in enumerate(search_result_docs):
                    title = doc.metadata.get("title", "검색 결과")
                    url = doc.metadata.get("url", "")
                    
                    tavily_text += f"**{i+1}.** {title}\n"
                    if url:
                        tavily_text += f"   [원본 링크]({url})\n"
                    # 내용 미리보기 (너무 길면 생략)
                    content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    tavily_text += f"   {content_preview}\n\n"
                
                # 타빌리 소스 정보 메시지 전송
                await cl.Message(content=tavily_text, author="웹 검색").send()
        
        # 상담 질문에 대한 소스 문서 표시 추가
        elif question_type == "counseling" and documents:
            # 소스 문서 헤더 표시
            await cl.Message(content="### 참고 자료", author="시스템").send()
            
            # 각 문서를 개별 메시지로 표시
            for i, doc in enumerate(documents):
                title = doc.metadata.get("title", "제목 없음")
                url = doc.metadata.get("video_url", "")
                
                # 유튜브 URL에서 비디오 ID 추출
                video_id = ""
                if url and "youtube.com" in url:
                    if "v=" in url:
                        video_id = url.split("v=")[1].split("&")[0]
                    elif "youtu.be/" in url:
                        video_id = url.split("youtu.be/")[1].split("?")[0]
                
                # 제목과 링크 메시지 생성
                source_content = f"**{i+1}.** {title}\n\n"
                
                # 유튜브 URL이 있는 경우 링크 추가
                if url:
                    source_content += f"[유튜브 영상 바로가기]({url})\n\n"
                
                # 비디오 문서 내용 메시지 전송
                await cl.Message(content=source_content, author="시스템").send()
                
                # 유튜브 썸네일 이미지 별도 전송
                if video_id:
                    elements = []
                    # 유튜브 플레이어 추가하려고 시도
                    try:
                        # 비디오 요소 추가 시도
                        elements.append(
                            cl.Video(
                                name=f"youtube_video_{i}",
                                url=url,
                                display="inline"
                            )
                        )
                    except Exception as e:
                        # 비디오 요소가 실패하면 이미지로 대체
                        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
                        elements.append(
                            cl.Image(
                                name=f"youtube_thumbnail_{i}",
                                url=thumbnail_url,
                                display="inline"
                            )
                        )
                    
                    # 비디오/이미지 요소가 포함된 메시지 별도 전송
                    await cl.Message(content="", author="시스템", elements=elements).send()
    
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
        
    # LangSmith 트레이서가 모든 요청을 보낼 때까지 대기
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        try:
            await cl.make_async(wait_for_all_tracers)()
            cl.logger.info("모든 LangSmith 트레이스 전송 완료")
        except Exception as e:
            cl.logger.error(f"LangSmith 트레이스 대기 중 오류: {str(e)}")