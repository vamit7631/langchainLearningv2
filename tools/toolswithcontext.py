from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


USER_DATABASE = {
    "user123": {
        "name": "Amit Verma",
        "account_type": "Premium",
        "balance": 5000,
        "email": "vamit7631@gmail.com",
    },
    "user456": {
        "name": "Ankit Sharma",
        "account_type": "Standard",
        "balance": 1200,
        "email": "ankits@gmail.com",
    },
}

@dataclass
class UserContext:
    user_id: str 


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id
    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (
            f"Account holder: {user['name']}\n"
            f"Type: {user['account_type']}\n"
            f"Balance: ${user['balance']}"
        )
    return "User not found"


@tool
def get_email(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's email address."""

    user_id = runtime.context.user_id

    if user_id not in USER_DATABASE:
        return "User not found"

    return USER_DATABASE[user_id]["email"]

model = ChatOpenAI(model="gpt-5.4-mini")

agent = create_agent(
    model,
    tools=[get_account_info,get_email],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

result = agent.invoke(
    {
        "messages": [
             {
                "role": "user",
                "content": (
                    "What is my current balance and "
                    "what email is associated with my account?"
                ),
            }
        ]
    },
    config={
        "configurable": {
            "thread_id": str(uuid7()),
        }
    },
    context=UserContext(user_id="user123"),
)


print("\n=== Agent Response ===\n")

for message in result["messages"]:
    print(f"{message.type.upper()}:")
    print(message.content)
    print()