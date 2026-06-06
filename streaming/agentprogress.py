from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)


def get_weather(city: str) -> str: 
    """Get weather of the city"""
    return f"It's always sunny in {city}"

agent = create_agent(
    model=model,
    tools=[get_weather],
    checkpointer=InMemorySaver()
)

config = {"configurable" : { "thread_id": str(uuid7())}}

for chunk in agent.stream(
    {"messages" : [{"role": "user", "content": "What is weather in Bhopal ?"}]},
    config=config,
    stream_mode="updates",
    version="v2"
):
    if chunk["type"] == "updates":
        for step, data in chunk["data"].items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")

