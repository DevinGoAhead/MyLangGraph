import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage

load_dotenv()

class ConversationalResponse(BaseModel):
    response: str = Field(description="A conversational response to the user query")

class WeatherLoc(BaseModel):
    city: str = Field(description="The city for weather")

class SearchQuery(BaseModel):
    query: str = Field(description="Search query")

class UserInfo(BaseModel):
    name: str = Field(description="User name")

class FinalResponse(BaseModel):
    final_output: ConversationalResponse | SearchQuery | WeatherLoc | UserInfo

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

s_llm = llm.with_structured_output(FinalResponse, method="function_calling")
res = s_llm.invoke([HumanMessage(content="介绍下你自己")])
print("res:", res)
print("type(res):", type(res))
print("isinstance(res, FinalResponse):", isinstance(res, FinalResponse))
