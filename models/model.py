import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-5.4-mini", model_provider="openai")

response = model.invoke("Why do parrots talk?")

print(response.content)


