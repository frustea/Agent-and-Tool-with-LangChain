# main.py
import os
import requests
import datetime
import wikipedia
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set API key for OpenAI
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class TriangleValidator(BaseModel):
    side_a: float = Field(..., description="Length of side A")
    side_b: float = Field(..., description="Length of side B")
    side_c: float = Field(..., description="Length of side C")

@tool(args_schema=TriangleValidator)
def validate_triangle(side_a: float, side_b: float, side_c: float) -> str:
    if side_a + side_b > side_c and side_b + side_c > side_a and side_c + side_a > side_b:
        return "Yes, they can form a triangle."
    return "No, they cannot form a triangle."

class WeatherDataInput(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")

@tool(args_schema=WeatherDataInput)
def fetch_weather_data(latitude: float, longitude: float) -> str:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {'latitude': latitude, 'longitude': longitude, 'hourly': 'temperature_2m', 'forecast_days': 1}
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        current_utc_time = datetime.datetime.utcnow()
        times = [datetime.datetime.fromisoformat(t.replace('Z', '+00:00')) for t in data['hourly']['time']]
        temperatures = data['hourly']['temperature_2m']
        closest_time_index = min(range(len(times)), key=lambda i: abs(times[i] - current_utc_time))
        return f"The current temperature is {temperatures[closest_time_index]}°C"
    else:
        raise Exception(f"API request failed with status code: {response.status_code}")

@tool
def search_wikipedia(query: str) -> str:
    try:
        page_titles = wikipedia.search(query)
        summaries = [f"Page: {title}\nSummary: {wikipedia.summary(title)}" for title in page_titles[:3]]
        return "\n\n".join(summaries) if summaries else "No relevant Wikipedia pages found."
    except Exception as e:
        return f"Error fetching Wikipedia data: {e}"