import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
from dotenv import load_dotenv
import os
from mental_agent_system.graph import create_graph, graph, AgentState

load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])

@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append(message.content)

    # 초기 상태 설정
    inputs = {
        "messages": messages,
        "next": "Supervisor"
    }
    
    # 그래프 실행
    config = {"recursion_limit": 50}
    result = graph.invoke(inputs, config)
    
    # 마지막 메시지 전송
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            await cl.Message(content=msg.content).send()

    # 메시지 기록 업데이트
    cl.user_session.set("messages", result["messages"])
