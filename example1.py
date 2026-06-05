from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()


def get_whether(city: str) -> str:
    """Get weather information for a city."""
    return f"It's always sunny in {city}"

model = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=model,
    tools=[get_whether],
    system_prompt="You are helpful assistant"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role" : "user", 
                "content": "Please tell me the whether of Bhopal in Madhya Pradesh"
            }
        ]
    }
)

print(result["messages"][-1].content_blocks)