from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)


def get_weather(city: str) -> str: 
    """Get weather of the city"""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}"

agent = create_agent(
    model=model,
    tools=[get_weather]
)

for chunk in agent.stream(
    {"messages" : [{"role": "user", "content": "What is weather in Bhopal ?"}]},
    stream_mode=["updates", "custom"],
    version="v2"
):
    print(f"stream mode: {chunk['type']}")
    print(f"content: {chunk['data']}")
    print("\n")

