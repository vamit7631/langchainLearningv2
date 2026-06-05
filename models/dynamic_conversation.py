from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)

conversation = [
    SystemMessage(
        content="You are helpful assistant"
    )
]


while True:
    user_input = input("You : ")

    if user_input.lower() == "exit":
        break
    
    conversation.append(HumanMessage(content=user_input))

    response = model.invoke(conversation)

    print("AI : " , response.content)
    conversation.append(AIMessage(content=response.content))