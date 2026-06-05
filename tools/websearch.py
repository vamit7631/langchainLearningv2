from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)

@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search

@tool("calculator", description="To calculate arithmetic expression ")
def calc(expression: str) -> str:
    return str(eval(expression))

print(calc.invoke("2*2"))