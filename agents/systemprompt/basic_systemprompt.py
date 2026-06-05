from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=model,
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "Hi I'm Jarvis How may i help you!"
            },
            {
                "type": "text",
                "text": "Content related to country details"
            }
        ]
    )
)

response =  agent.invoke(
    {
        "messages": [
            HumanMessage("Please tell me capital of India and prime minister of India ?")
        ]
    }
)

print(response['messages'][-1].content)