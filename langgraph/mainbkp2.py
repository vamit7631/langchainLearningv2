from fastapi import FastAPI
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str) -> str:
    """Get the current weather information for a city."""
    sample_data = {
        "bengaluru": "28°C, partly cloudy",
        "bangalore": "28°C, partly cloudy",
        "bhopal": "16°C, rainy",
        "mumbai": "33°C, sunny",
    }
    return sample_data.get(city.lower())

@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b` 
    Args: 
        a: First int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract `a` and `b` 
    Args: 
        a: First int
        b: second int
    """
    return a - b

tools = [get_weather, add, subtract]


model = ChatOllama(model="llama3.1", temperature=0)
model_with_tools = model.bind_tools(tools)


def mock_llm(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

def should_continue(state: MessagesState):

    last_message = state["messages"][-1]

    # If LLM requested a tool
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise finish
    return END


graph = StateGraph(MessagesState)
graph.add_node("mock_llm", mock_llm)
graph.add_node("tool_node",tool_node)
graph.add_edge(START, "mock_llm")
graph.add_conditional_edges("mock_llm", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "mock_llm")
graph = graph.compile()



app = FastAPI()


class ChatRequest(BaseModel):
    message: str

@app.post("/chat/v1")
async def chatfn(req: ChatRequest):
    result = graph.invoke({"messages": [{"role": "user", "content": req.message}]})
    return {"output": result["messages"][-1].content}