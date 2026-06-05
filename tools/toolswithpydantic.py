from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini")


class WeatherInput(BaseModel):
    location : str = Field(description="City Name or Coordinates")
    units : Literal["celcius", "fahrenheit"] = Field(
        default="celcius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str= "celcius", include_forecast: bool = False) -> str:
    """ Get current weather and optional forecast"""
    temp = 22 if units == "celcius" else 72
    print(units,"=======")
    result = f"Current weather in {location}: {temp} degrees {units}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


response1 = get_weather.invoke({
    "location":"Mumbai"
})
print(response1)