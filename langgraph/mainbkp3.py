from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)

def write_agent(state: MessagesState):
    system = SystemMessage(content="You are creative writing agent.")
    print(system,"=====sys")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("writer", write_agent)
graph.add_edge(START, "writer")
graph.add_edge("writer",END)
graph = graph.compile()

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str


@app.post("/chat/v2")
def chatfn(req: ChatRequest):
    result = graph.invoke({"messages": [HumanMessage(content=req.messages)]})
    return {"output" : result["messages"][-1].content}