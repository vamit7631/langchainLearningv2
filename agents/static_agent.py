from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini")

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expression"""
    return str(eval(expression))


tools = [calculator]

agent = create_agent(
    model=model,
    tools=tools
)

result = agent.invoke(
    {
        "messages": [
            {
                "role" : "user",
                "content" : "What is 25 * 4 ?"
            }
        ]
    }
)

print(result['messages'][-1].content)