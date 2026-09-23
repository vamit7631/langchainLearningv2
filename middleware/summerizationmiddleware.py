from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool


load_dotenv()

@tool
def weather_tool(city: str) -> str:
    """Get weather information of city"""
    return f"The weather in {city} is 28°C and sunny."

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# -------------------------
# Agent 1
# Trigger summarization when:
# - Tokens >= 4000
# Keep:
# - Last 20 messages
# -------------------------

agent = create_agent(
    model="gpt-5.4",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )
    ]
)


response = agent.invoke(
    {
        "messages": [
            {
            "role": "user",
            "content": "What is weather in Bangalore?"
            }
        ]
    }
)

print("Agent 1 Response:")
print(response)

# -------------------------
# Agent 2
# Trigger summarization when:
# - Tokens >= 3000 OR
# - Messages >= 6
# Keep:
# - Last 20 messages
# -------------------------


agent2 = create_agent(
    model="gpt-5.4",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=[
                ("tokens", 3000),
                ("messages", 6),
            ],
            keep=("messages", 20),
        ),
    ],
)

conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "What is 100 + 200?"},
    {"role": "assistant", "content": "300"},
    {"role": "user", "content": "What is 50 * 4?"},
    {"role": "assistant", "content": "200"},
]


response2 = agent2.invoke(
    {
        "messages": conversation
    }
)

print("\nAgent 2 Response:")
print(response2)


# -------------------------
# Agent 3
# Context-window based summarization
#
# Trigger:
# - Context reaches 80%
#
# Keep:
# - Reduce to 30% of context
# ------------------------- 


agent3 = create_agent(
    model="gpt-5.4",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)

response3 = agent3.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain machine learning in detail."
            }
        ]
    }
)

print("\nAgent 3 Response:")
print(response3)