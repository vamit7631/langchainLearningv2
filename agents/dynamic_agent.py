from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from datetime import datetime


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def calculator(expression: str) -> str:
    """
    Useful for mathematical calculations.
    Example: 45 * 20
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def current_time() -> str:
    """
    Returns current system time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def weather(city: str) -> str:
    """
    Get weather information for a city.
    """
    return f"The weather in {city} is 27°C and cloudy."


ENABLE_CALCULATOR = True
ENABLE_TIME = True
ENABLE_WEATHER = True

tools = []

if ENABLE_CALCULATOR:
    tools.append(calculator)

if ENABLE_TIME:
    tools.append(current_time)

if ENABLE_WEATHER:
    tools.append(weather)


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are a smart AI assistant.
    
    Use tools whenever needed.
    Answer clearly and concisely.
    """
)


while True:

    user_input = input("\nUser: ")

    if user_input.lower() == "exit":
        break
    
    response = agent.invoke(
        {
            "messages" : [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )


    print("\nAI: ", response["messages"][-1].content)