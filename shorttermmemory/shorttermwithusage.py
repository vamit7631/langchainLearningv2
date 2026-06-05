from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-5.4")

def get_user_info() -> str:
    """Information about current user"""
    return "No user profile on file"

agent = create_agent(
    model,
    tools=[get_user_info],
    checkpointer=InMemorySaver()
)

thread_config = { "configurable" : {"thread_id" : "1"}}

response = agent.invoke(
    {"messages" : [{"role": "user", "content": "Hi! I am Amit."}]},
    thread_config,
)["messages"][-1].content

print(response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)["messages"][-1].content

print(response) 
