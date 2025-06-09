from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ag_ui.core import (
  RunAgentInput,
  Message,
  EventType,
  RunStartedEvent,
  RunFinishedEvent,
  TextMessageStartEvent,
  TextMessageContentEvent,
  TextMessageEndEvent,
  ToolCallStartEvent,
  ToolCallArgsEvent,
  ToolCallEndEvent,
  CustomEvent
)
from ag_ui.encoder import EventEncoder
import uuid
from openai import OpenAI
from dotenv import load_dotenv
import json
from pprint import pformat

# OpenAI model to use
OPENAI_MODEL = "gpt-4.1"

app = FastAPI(title="AG-UI Endpoint")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env file
load_dotenv()

import requests

def get_weather(latitude, longitude):
    try:
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if 'current' not in data or 'temperature_2m' not in data['current']:
            return "Weather data not available"
            
        return f"Current temperature: {data['current']['temperature_2m']}°C"
    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Error processing weather data: {str(e)}"

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

def text_message_start(encoder, message_id, role="assistant"):
    yield encoder.encode(
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role=role
        )
    )

def run_started(encoder, thread_id, run_id):
    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id
        )
    )

def custom_event_agent_name(encoder, agent_name):
    yield encoder.encode(
        CustomEvent(
            type=EventType.CUSTOM,
            name="agent_name",
            value=agent_name
        )
    )

def text_message_content(encoder, message_id, delta):
    yield encoder.encode(
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=delta
        )
    )

def text_message_end(encoder, message_id):
    yield encoder.encode(
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id
        )
    )

def tool_call_args(encoder, tool_call_id, delta):
    yield encoder.encode(
        ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=delta
        )
    )

def tool_call_start(encoder, tool_call_id, tool_call_name, parent_message_id):
    yield encoder.encode(
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name=tool_call_name,
            parent_message_id=parent_message_id
        )
    )

def tool_call_end(encoder, tool_call_id):
    yield encoder.encode(
        ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id
        )
    )

def run_finished(encoder, thread_id, run_id):
    yield encoder.encode(
        RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=thread_id,
            run_id=run_id
        )
    )

def stream_final_response(encoder, client, input_messages, message_id):
    final_stream = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        stream=True
    )
    complete_response = ""
    for event in final_stream:
        if event.type == 'response.output_text.delta':
            print(event.delta, end='', flush=True)
            complete_response += event.delta
            yield from text_message_content(encoder, message_id, event.delta)
        elif event.type == 'response.completed':
            yield from text_message_end(encoder, message_id)
            print(f"\n--- Final Response ---\n{complete_response}\n----------------------")

def event_generator(input_data, agent_name):
    # Create an event encoder to properly format SSE events
    encoder = EventEncoder()
    if agent_name:
        yield from custom_event_agent_name(encoder, agent_name)
    # Send run started event
    yield from run_started(encoder, input_data.thread_id, input_data.run_id)

    # Initialize OpenAI client
    client = OpenAI()

    # Generate a message ID for the assistant's response
    message_id = str(uuid.uuid4())

    # Send text message start event
    yield from text_message_start(encoder, message_id, "assistant")
    # Create a streaming completion request
    input_messages = [{"role": "user", "content": input_data.messages[0].content}]

    stream = client.responses.create(
        model=OPENAI_MODEL,
        input=input_messages,
        tools=tools,
        stream=True
    )

    tool_call = {}

    for event in stream:
        #if the event is a function call, add it to the tool_call dictionary
        if event.type == 'response.output_text.delta':
            print(event.delta, end='', flush=True)
            yield from text_message_content(encoder, message_id, event.delta)
        elif event.type == 'response.completed':
            print()
            yield from text_message_end(encoder, message_id)
        elif event.type == 'response.output_item.added' and event.item.type == "function_call":
            tool_call[event.output_index] = event.item
        elif event.type == 'response.function_call_arguments.delta':
            index = event.output_index
            if tool_call[index]:
                tool_call[index].arguments += event.delta
                #encode the event
                yield from tool_call_args(encoder, event.item_id, event.delta)
    
    if tool_call:
        tool_call_obj = next(iter(tool_call.values()))
        args = json.loads(tool_call_obj.arguments)
        input_messages.append(tool_call_obj)  # append model's function call message
        yield from tool_call_start(encoder, tool_call_obj.id, tool_call_obj.name, message_id)
        yield from tool_call_args(encoder, tool_call_obj.id, tool_call_obj.arguments)
        result = get_weather(args["latitude"], args["longitude"])
        input_messages.append({                               # append result message
            "type": "function_call_output",
            "call_id": tool_call_obj.call_id,
            "output": str(result)
        })
        yield from tool_call_end(encoder, tool_call_obj.id)
        yield from stream_final_response(encoder, client, input_messages, message_id)

    # Send run finished event
    yield from run_finished(encoder, input_data.thread_id, input_data.run_id)

@app.post("/awp")
async def my_endpoint(input_data: RunAgentInput):
    print(input_data.state)
    agent_name = input_data.state.get('agent')
    print(agent_name)

    return StreamingResponse(
        event_generator(input_data, agent_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)