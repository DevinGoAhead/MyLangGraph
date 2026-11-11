import os
import requests
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Tools
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    if "shanghai" in city.lower() or "上海" in city:
        return "上海天气晴朗，气温20度"
    return f"{city}天气未知"

@tool
def search_online(query: str) -> str:
    """通过搜索引擎搜索最新信息和新闻"""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "模拟搜索结果：黄仁勋在GTC大会上发布了全新AI芯片。"
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        return str(res.json().get("organic", [])[:2])
    except Exception as e:
        return f"搜索失败: {e}"

@tool
def insert_db(name: str, age: int | None = None, email: str | None = None, phone: str | None = None) -> str:
    """将用户信息存储到数据库"""
    return f"成功插入用户: {name}"

tools = [get_weather, search_online, insert_db]

# LLM setup with Gemini
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

class FinalResponse(BaseModel):
    decision: Literal["call_function", "final_answer"] = Field(
        description="如果需要调用外部工具完成任务，选择 call_function；如果可以直接回答用户，选择 final_answer。"
    )
    response: str = Field(
        description="当 decision 为 final_answer 时，填写给用户的最终回复内容；当为 call_function 时，可以留空或简要说明。"
    )

structured_llm = llm.with_structured_output(FinalResponse, method="function_calling")
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_router(state: AgentState):
    first_user_msg = state["messages"][0]
    output = structured_llm.invoke([first_user_msg])
    return {"messages": [output]}

def router_condition(state: AgentState) -> Literal["call_function", "final_answer"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "decision"):
        return last_msg.decision
    elif isinstance(last_msg, dict) and "decision" in last_msg:
        return last_msg["decision"]
    return "final_answer"

def excute_function(state: AgentState):
    first_user_msg = state["messages"][0]
    message_with_tool_calls = llm_with_tools.invoke([first_user_msg])
    if not getattr(message_with_tool_calls, "tool_calls", None):
        return {"messages": [message_with_tool_calls.content or "已处理"]}
    return {"messages": [message_with_tool_calls]}

def final_answer(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "response"):
        return {"messages": [last_msg.response]}
    elif isinstance(last_msg, dict) and "response" in last_msg:
        return {"messages": [last_msg["response"]]}
    return {"messages": ["完成"]}

tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("agent_router", agent_router)
builder.add_node("excute_function", excute_function)
builder.add_node("tools", tool_node)
builder.add_node("final_answer", final_answer)

builder.add_edge(START, "agent_router")
builder.add_conditional_edges(
    "agent_router",
    router_condition,
    {
        "call_function": "excute_function",
        "final_answer": "final_answer",
    }
)
builder.add_edge("excute_function", "tools")
builder.add_edge("tools", END)
builder.add_edge("final_answer", END)

graph = builder.compile()

# Test queries
queries = [
    "介绍下你自己",
    "上海天气怎么样",
    "给我说一下关于黄仁勋的最新的新闻",
]

for q in queries:
    print(f"\n--- Testing query: {q} ---")
    res = graph.invoke({"messages": [HumanMessage(content=q)]})
    msgs = res["messages"]
    print(f"Total messages: {len(msgs)}")
    for m in msgs:
        print("  >", type(m).__name__, getattr(m, "content", m))
