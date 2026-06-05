from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

# Multiple conversations/prompts
inputs = [
    [HumanMessage(content="What is Python?")],
    [HumanMessage(content="What is Machine Learning?")],
    [HumanMessage(content="What is Generative AI?")]
]

# Batch processing
responses = model.batch(inputs)

# Print responses
for i, response in enumerate(responses, start=1):
    print(f"\nResponse {i}:")
    print(response.content)