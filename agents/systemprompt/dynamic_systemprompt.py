from typing import TypedDict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import dynamic_prompt , ModelRequest


load_dotenv()

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt dynamically based on user role."""

    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are helpful ai assistant"

    if user_role == 'admin':
        return (
            f"{base_prompt} "
            "Provide 10 line details"
        )
    if user_role == "manager":
        return (
            f"{base_prompt} "
            "Provide 5 line details"           
        )
    
    return base_prompt


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

agent = create_agent(
    model = model,
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

user_question = input("Enter your question:  ")
user_role = input("Enter your role : ")

result = agent.invoke(
    {
        "messages": [
            {
            "role" : "user",
            "content" : user_question
            }
        ]
    },
    context={
        "user_role": user_role
    }
)


print("\n===== RESPONSE =====\n")


print(result["messages"][-1].content)