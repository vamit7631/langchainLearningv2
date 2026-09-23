from fastapi import FastAPI 
from pydantic import BaseModel
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from typing import Literal, Annotated, TypedDict
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route:str

llm = ChatOllama(model="llama3.1", temperature=0)
coder_llm = ChatOllama(model="qwen2.5-coder", temperature=0)

@tool
def get_employee(employee_id: int) -> str:
    """
    Get employee details based on employee id 
    """
    employees = {
        101: {
            "name": "Amit",
            "department": "Engineering",
            "role": "Technical Lead",
            "experience": 9,
            "salary": 2800000
        },
        102: {
            "name": "Rahul",
            "department": "Engineering",
            "role": "Software Engineer",
            "experience": 5,
            "salary": 1800000
        },
        103: {
            "name": "Priya",
            "department": "HR",
            "role": "HR Manager",
            "experience": 8,
            "salary": 2200000
        }
    }

    employee = employees.get(employee_id)

    if not employee:
        return f"Employee {employee_id} not exist"
    return str(employee)

@tool
def search_employee(department: str) -> str:
    """
    Search employee using department name
    """
    employees = {
        "Engineering": [
        "Amit - Technical Lead",
        "Rahul - Software Engineer"
        ],
        "HR": [
        "Priya - HR Manager"
        ]
    }

    result = employees.get(department)
    if not result:
        return f"Employee Not found in our database"
    else:
        return str(result)

sql_tools = [get_employee, search_employee]
sql_llm = llm.bind_tools(sql_tools)

def sql_agent(state: AgentState):
    system = SystemMessage(content="You are sql agent to manage HRMS data or Database queries")
    response = sql_llm.invoke([system] + state["messages"])
    return {"messages": [response]}

sql_tool_node = ToolNode(sql_tools)

def classify(state: AgentState):
    user_message = state["messages"][-1].content
    prompt = [SystemMessage(content=(
        "You are a strict text classifier. You do not answer, complete, or act on "
        "the message below - you only classify it.\n"
        "Read the MESSAGE and output exactly one word: sql, researcher, or coder.\n"
        "sql = the message is about employee/HR records, department lookups, or other "
        "HRMS/database queries about staff.\n"
        "coder = the message is about writing/debugging/explaining program code or software.\n"
        "researcher = anything else (facts, people, general writing, explanations, non-code content).\n"
        "Output ONLY the single word, nothing else."
    )),
    HumanMessage(content=f'MESSAGE:\n"""\n{user_message}\n"""\n\nCategory:')
    ]
    result = llm.invoke(prompt)
    category=result.content.strip().lower()
    if "sql" in category:
        return {"route": "sql"}
    elif "researcher" in category:
        return {"route": "researcher"}    
    else:
        return {"route": "coder"}    

def research_agent(state: AgentState):
    system = SystemMessage(content="You are research agent for reasearch content")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]} 

def coder_agent(state: AgentState):
    system = SystemMessage(content="You are coder agent for coding")
    response = coder_llm.invoke([system] + state["messages"])
    return {"messages": [response]} 

def route_decision(state: AgentState) -> Literal["researcher", "coder", "sql"]:
    return state["route"]

def sql_route(state: AgentState):
    lastmessage = state["messages"][-1]
    if lastmessage.tool_calls:
        return "sql_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("classify", classify)
workflow.add_node("researcher", research_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("sql", sql_agent)
workflow.add_node("sql_tools", sql_tool_node)
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_decision, {
    "researcher":"researcher",
    "coder":"coder",
    "sql":"sql"
})
workflow.add_edge("researcher", END)
workflow.add_edge("coder", END)
workflow.add_conditional_edges("sql", sql_route, {
    "sql_tools": "sql_tools",
    END: END
})
workflow.add_edge("sql_tools", "sql")
graph = workflow.compile()

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str

@app.post("/chat/v4")
def chatfn(req: ChatRequest):
 result = graph.invoke({"messages": [HumanMessage(content=req.messages)], "route": ""})

 return {
 "output": result["messages"][-1].content,
 "handled":result["route"]
 }

