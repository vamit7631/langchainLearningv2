from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)

conversation = [
    SystemMessage(
        content="You are helpful assistant that translates English to hindi"
    ),
    HumanMessage(
        content="I love programming"
    ),
    AIMessage(
        content="मुझे प्रोग्रामिंग पसंद है"
    ),
    HumanMessage(
        content="I love you"
    ),
]

response = model.invoke(conversation)

print("Full Response:")
print(response)

# Print only generated text
print("\nGenerated Text:")
print(response.content)