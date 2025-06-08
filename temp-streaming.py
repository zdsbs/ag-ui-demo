import os
from dotenv import load_dotenv
import json
load_dotenv()

from openai import OpenAI
client = OpenAI()

import requests

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]

input_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]

stream = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
    stream=True
)

tool_call = {}

for event in stream:
    if event.type == 'response.output_item.added':
        tool_call[event.output_index] = event.item;
    elif event.type == 'response.function_call_arguments.delta':
        index = event.output_index

        if tool_call[index]:
            tool_call[index].arguments += event.delta

print(tool_call)

# Get the first (and only) tool call from the dictionary
tool_call_obj = next(iter(tool_call.values()))
args = json.loads(tool_call_obj.arguments)

result = get_weather(args["latitude"], args["longitude"])

input_messages.append(tool_call_obj)  # append model's function call message
input_messages.append({                               # append result message
    "type": "function_call_output",
    "call_id": tool_call_obj.call_id,
    "output": str(result)
})

final_stream = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
    stream=True
)

# Initialize an empty string to build up the response
complete_response = ""

for event in final_stream:
    if event.type == 'response.output_text.delta':
        # Print each delta as it comes in
        print(event.delta, end='', flush=True)
        complete_response += event.delta
    elif event.type == 'response.completed':
        # Print a newline at the end
        print()