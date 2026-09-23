from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------
# Tools
# -----------------------------------

def read_email_tool(email_id: str) -> str:
    """Read an email."""
    return f"Email content for ID {email_id}: Meeting scheduled for tomorrow at 10 AM."


def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {recipient} with subject '{subject}'."


# -----------------------------------
# Agent
# -----------------------------------

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-5.4",
    tools=[read_email_tool, send_email_tool],
    checkpointer=checkpointer,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": [
                        "approve",
                        "edit",
                        "reject",
                    ]
                },
                "read_email_tool": False,  # No approval required
            }
        )
    ],
)

config = {
    "configurable": {
        "thread_id": "email-thread-1"
    }
}

# -----------------------------------
# User Request
# -----------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Send an email to john@example.com "
                    "with subject 'Project Update' "
                    "and body 'The deployment was successful.'"
                ),
            }
        ]
    },
    config=config,
)

print(result)