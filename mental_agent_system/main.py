import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from uuid import uuid4
from dotenv import load_dotenv
import os
from mental_agent_system.graph import create_graph

load_dotenv()

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", [])

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    history = cl.user_session.get("messages") or []
    history.append(HumanMessage(content=user_input))

    inputs = {"messages": history}
    config = RunnableConfig(configurable={"thread_id": str(uuid4())})
    graph = create_graph()
    result = graph.invoke(inputs, config)

    last_msg = result["messages"][-1]
    history.append(last_msg)
    await cl.Message(content=last_msg.content).send()
    cl.user_session.set("messages", history)
