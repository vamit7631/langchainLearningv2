from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents.middleware import (
    wrap_model_call, ModelRequest, ModelResponse
)

load_dotenv()

# =====================================================
# MODELS
# =====================================================

basic_model = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0
)

advanced_model = ChatOpenAI(
    model="gpt-5.4",
    temperature=0
)

# =====================================================
# TOOLS
# =====================================================


@tool
def calculator(expression: str) -> str:
    """ Evaluate mathematical expression """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def weather(city: str) -> str:
    """ Get weather information """
    return f"{city} weather is 30°C and sunny."


tools = [calculator, weather]

# =====================================================
# DYNAMIC MODEL ROUTING MIDDLEWARE
# =====================================================

@wrap_model_call
async def dynmaic_model_route(request : ModelRequest, handler) -> ModelResponse:
    user_message = request.messages[-1].content.lower()

    complex_keywords = [
        "analyze",
        "architecture",
        "research",
        "strategy",
        "detailed",
        "optimize",
        "microservices",
        "ai system",
    ]

    if any(word in user_message for word in complex_keywords):
        print("\nUsing ADVANCED model\n")
        request = request.override(
            model = advanced_model
        )
    else:
        print("\nUsing BASIC model\n")
        request = request.override(
            model = basic_model
        )
    
    response = await handler(request)
    return response

# =====================================================
# CREATE AGENT
# =====================================================

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[dynmaic_model_route]
)

# =====================================================
# HELPER FUNCTION
# =====================================================

async def ask_agent(query: str):

    response = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })

    # Final clean output
    return response["messages"][-1].content



# =====================================================
# MAIN
# =====================================================

async def main():

    # -----------------------------------------
    # SIMPLE QUERY
    # -----------------------------------------

    answer1 = await ask_agent(
        "What is 25 * 4 ?"
    )

    print("Simple Query Output:")
    print(answer1)

    # -----------------------------------------
    # COMPLEX QUERY
    # -----------------------------------------

    answer2 = await ask_agent(
        "Analyze microservices architecture for healthcare AI platform"
    )

    print("\nComplex Query Output:")
    print(answer2)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    import asyncio

    asyncio.run(main())