from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

conversation = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Hindi."
    ),

    HumanMessage(
        content="I love programming."
    ),

    AIMessage(
        content="झे प्रोग्रामिंग पसंद है"
    ),

    HumanMessage(
        content="Translate: I love building applications."
    )
]

print("AI: ", end="")

# Streaming response
for chunk in model.stream(conversation):
    print(chunk.content, end="", flush=True)