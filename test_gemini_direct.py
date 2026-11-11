import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API Key present:", bool(api_key))

models_to_test = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

for model_name in models_to_test:
    try:
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        class Response(BaseModel):
            message: str = Field(description="A greeting response")

        s_llm = llm.with_structured_output(Response, method="function_calling")
        res = s_llm.invoke("Hello")
        print(f"Success with {model_name}: {res}")
        break
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
