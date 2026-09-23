from fastapi import FastAPI
from langgraph.graph import START, END, StateGraph, MessagesState 

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "Hello World!"}]}


graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

app = FastAPI()


@app.post("/chat")
async def chatfn():
    result = graph.invoke({"messages": [{"role": "ai", "content": "Hi!"}]})
    return result