from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

load_dotenv()

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    gender: str

model = ChatOpenAI(model="gpt-5.4-mini")

agent = create_agent(
    model=model,
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [
        {
            "role" : "user",
            "content" : "I'm Amit Verma. I'm from Bhopal, My phone number is 9893273970. I'm male and my email id vamit7631@gmail.com."
        }
    ]
 }
)

print(result["structured_response"])