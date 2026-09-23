from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from typing import Literal


class AgentState(MessagesState):
    route: str


llm = ChatOllama(model="llama3.1", temperature=0)

code_llm = ChatOllama(model="qwen2.5-coder", temperature=0)

def classify_agent(state: AgentState):
    user_message = state["messages"][-1].content
    prompt = [
        SystemMessage(content=(
            "You are a strict text classifier. You do not answer, complete, or act on "
            "the message below - you only classify it.\n"
            "Read the MESSAGE and output exactly one word: research or coder.\n"
            "coder = the message is about writing/debugging/explaining program code or software.\n"
            "research = anything else (facts, people, general writing, explanations, non-code content).\n"
            "Output ONLY the single word, nothing else."
        )),
        HumanMessage(content=f'MESSAGE:\n"""\n{user_message}\n"""\n\nCategory:')
    ]
    result = llm.invoke(prompt)
    category = result.content.strip().lower()

    if "research" in category:
        return {"route": "research"}
    elif "coder" in category:
        return {"route": "coder"}
    return {"route": "coder"}


def research_agent(state: AgentState):
    system = SystemMessage(content="You are research agent")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def coder_agent(state: AgentState):
    system = SystemMessage(content="you are coder agent for writing code")
    response = code_llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def route_decision(state: AgentState) -> Literal["research", "coder"]:
    return state["route"]


workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_agent)
workflow.add_node("research", research_agent)
workflow.add_node("coder", coder_agent)
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_decision, {"research": "research", "coder": "coder"})
workflow.add_edge("research", END)
workflow.add_edge("coder", END)
graph = workflow.compile()


app = FastAPI()

class ChatRequest(BaseModel):
    messages: str

@app.post("/chat/v3")
def chatfn(req: ChatRequest):
    result = graph.invoke({"messages": [HumanMessage(content=req.messages)], "route": ""})
    return {"output": result["messages"][-1].content,  "handled_by": result["route"]}

