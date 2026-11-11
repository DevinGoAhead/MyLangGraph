import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# Schema definitions from course
class ConversationalResponse(BaseModel):
    """Use this for general conversations, self-introduction, greetings, knowledge questions, and any chat that does NOT require real-time search, weather, or saving user info."""
    response: str = Field(description="A conversational response to the user's query")

class WeatherLoc(BaseModel):
    """Call weather tool"""
    city: str = Field(description="The city for weather")

class SearchQuery(BaseModel):
    """Call search tool"""
    query: str = Field(description="Search query")

class UserInfo(BaseModel):
    """Call insert db tool"""
    name: str = Field(description="User name")

class FinalResponse(BaseModel):
    final_output: ConversationalResponse | SearchQuery | WeatherLoc | UserInfo

# Tools
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}天气晴朗，气温20度"

@tool
def search_online(query: str) -> str:
    """通过搜索引擎搜索最新信息和新闻"""
    return f"搜索结果: 关于{query}的最新信息"

@tool
def insert_db(name: str) -> str:
    """将用户信息存储到数据库"""
    return f"成功插入用户: {name}"

tools = [get_weather, search_online, insert_db]
tool_node = ToolNode(tools)

# LLM
qwen_llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_structed_output = qwen_llm.with_structured_output(FinalResponse, method="function_calling")
model_with_tools = qwen_llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def chat_with_model(state: AgentState):
    messages = state["messages"]
    response = llm_structed_output.invoke(messages)
    return {"messages": [response]}

def excute_function(state: AgentState):
    messages = state["messages"][-1]
    message_with_tool_calls = model_with_tools.invoke(str(messages))

    if not message_with_tool_calls.tool_calls:
        return {"messages": [message_with_tool_calls.content]}

    response = tool_node.invoke({"messages": [message_with_tool_calls]})
    response_content = response["messages"][0].content
    return {"messages": [response_content]}

def final_answer(state: AgentState):
    message = state["messages"][-1]
    assert isinstance(message, FinalResponse)
    output = message.final_output
    assert isinstance(output, ConversationalResponse)
    return {"messages": [output.response]}

def generate_branch(state: AgentState):
    message = state["messages"][-1]
    assert isinstance(message, FinalResponse)
    return isinstance(message.final_output, ConversationalResponse)

agent_graph = StateGraph(AgentState)
agent_graph.add_node("chat_with_model", chat_with_model)
agent_graph.add_node("excute_function", excute_function)
agent_graph.add_node("final_answer", final_answer)

agent_graph.set_entry_point("chat_with_model")
agent_graph.add_conditional_edges(
    "chat_with_model", generate_branch, {True: "final_answer", False: "excute_function"}
)
agent_graph.set_finish_point("excute_function")
agent_graph.set_finish_point("final_answer")

agent_graph = agent_graph.compile()

for text in ["介绍下你自己", "上海天气怎么样", "给我说一下关于黄仁勋的最新的新闻"]:
    print(f"\nTesting: {text}")
    res = agent_graph.invoke({"messages": [HumanMessage(content=text)]})
    print("Result messages:")
    for m in res["messages"]:
        print("  -", type(m), getattr(m, "content", m))
