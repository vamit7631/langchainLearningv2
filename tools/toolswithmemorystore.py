from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Any 
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool

load_dotenv()
model = ChatOpenAI(model="gpt-5.4-mini")

@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str: 
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "User saved successfully!"


@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Get user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})