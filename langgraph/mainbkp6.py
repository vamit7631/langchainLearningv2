from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama
from typing import Literal, Annotated, TypedDict
import json

class AgentState(TypedDict):
    messages:Annotated[list, add_messages]
    steps: int
    next: str

MAX_STEPS = 6
llm = ChatOllama(model="llama3.1", temperature=0)

coder_llm = ChatOllama(model="qwen2.5-coder", temperature=0)

MEMBERS = ["researcher", "coder"]
def supervisor(state: AgentState):
    system = SystemMessage(content=(
        f"You are a supervisor managing these workers: {MEMBERS}. "
        "Given the conversation so far, decide who should act next. "
        "'researcher' gathers facts/information. 'coder' writing code for functionality "
        "If the task is fully complete and ready to give to the user, respond with 'FINISH'. "
        "Reply with ONLY one word: researcher, coder, or FINISH."
    ))
    result = llm.invoke([system] + state["messages"])
    decision = result.content.strip().lower()

    if "finish" in decision:
        nxt = "FINISH"
    elif "coder" in decision:
        nxt = "coder"
    elif "researcher" in decision:
        nxt = "researcher"
    else:
        nxt = "researcher"

    if state["steps"] >= MAX_STEPS:
        nxt = "FINISH"

    return {"next": nxt, "steps": state["steps"] + 1}

def research_agent(state: AgentState):
    system = SystemMessage(content="You are research agent")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def coder_agent(state: AgentState):
    system = SystemMessage(content="You are coder agent")
    response = coder_llm.invoke([system] + state["messages"])
    return {"messages": [response]}

def route_decision(state: AgentState) -> Literal["researcher", "coder"]:
    return state["next"]

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", research_agent)
workflow.add_node("coder", coder_agent)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_decision, {
    "researcher": "researcher",
    "coder": "coder",
    "FINISH": END
})
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder","supervisor")
graph = workflow.compile()

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str

@app.post("/chat/v5")
async def chatfn(req: ChatRequest):
    result = await graph.ainvoke({"messages": [HumanMessage(content=req.messages)], "next":"", "steps":0})
    return {"output": result["messages"][-1].content, "steps_taken": result["steps"]}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_gen():
        async for event in graph.astream(
            {"messages": [HumanMessage(content=req.messages)], "next": "", "steps": 0},
            stream_mode="updates",   # yields {node_name: partial_state} per step
        ):
            for node_name, node_output in event.items():
                payload = {"node": node_name}

                if "next" in node_output:
                    payload["decision"] = node_output["next"]

                if "messages" in node_output:
                    last = node_output["messages"][-1]
                    payload["content"] = getattr(last, "content", str(last))

                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
    