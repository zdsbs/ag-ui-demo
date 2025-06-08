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
  ToolCallEndEvent
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

def print_event(event):
    """Pretty print an event with its type and relevant data."""
    print("\n" + "="*80)
    print(f"Event Type: {event.type}")
    print("-"*80)
    
    if hasattr(event, "item"):
        print("Item:")
        item = event.item
        print(f"  Name: {item.name}")
        print(f"  Status: {item.status}")
        if hasattr(item, "arguments"):
            try:
                args = json.loads(item.arguments)
                print("  Arguments:")
                print(f"    {pformat(args, indent=4)}")
            except:
                print(f"  Arguments: {item.arguments}")
    
    if hasattr(event, "delta"):
        print("Delta:")
        print(f"  {event.delta}")
    
    if hasattr(event, "arguments"):
        try:
            args = json.loads(event.arguments)
            print("Arguments:")
            print(f"  {pformat(args, indent=2)}")
        except:
            print(f"Arguments: {event.arguments}")
    
    if hasattr(event, "response"):
        resp = event.response
        print("Response:")
        if hasattr(resp, "text"):
            print(f"  Text: {resp.text}")
        if hasattr(resp, "tools"):
            print("  Tools:")
            for tool in resp.tools:
                print(f"    - {tool.name}: {tool.description}")
        if hasattr(resp, "output"):
            print("  Output:")
            for item in resp.output:
                if hasattr(item, "arguments"):
                    try:
                        args = json.loads(item.arguments)
                        print(f"    {item.name}: {pformat(args, indent=4)}")
                    except:
                        print(f"    {item.name}: {item.arguments}")
    
    print("="*80 + "\n")

@app.post("/awp")
async def my_endpoint(input_data: RunAgentInput):

    async def event_generator():
        # Create an event encoder to properly format SSE events
        encoder = EventEncoder()

        # Send run started event
        yield encoder.encode(
          RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
          )
        )

        # Initialize OpenAI client
        client = OpenAI()

        # Generate a message ID for the assistant's response
        message_id = str(uuid.uuid4())

        # Send text message start event
        yield encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant"
            )
        )
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
            initial_response = ""
            if event.type == 'response.created':
                print("created")
                yield encoder.encode(
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant"
                    )
                )
            if event.type == 'response.output_text.delta':
                print(event.delta, end='', flush=True)
                initial_response += event.delta
                yield encoder.encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=event.delta
                    )
                )
            if event.type == 'response.completed':
                print()
                yield encoder.encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id
                    )
                )
            if event.type == 'response.output_item.added' and event.item.type == "function_call":
                tool_call[event.output_index] = event.item;
            elif event.type == 'response.function_call_arguments.delta':
                index = event.output_index
                if tool_call[index]:
                    tool_call[index].arguments += event.delta
                    #encode the event
                    yield encoder.encode(
                        ToolCallArgsEvent(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=event.item_id,
                            delta=event.delta
                        )
                    )
        
        if tool_call:
            tool_call_obj = next(iter(tool_call.values()))
            input_messages.append(tool_call_obj)  # append model's function call message
            yield encoder.encode(
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call_obj.id,
                    tool_call_name=tool_call_obj.name,
                    parent_message_id=message_id
                )
            )         
            #TODO dynamically call the tool
            args = json.loads(tool_call_obj.arguments)

            result = get_weather(args["latitude"], args["longitude"])

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
                    yield encoder.encode(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=message_id,
                            delta=event.delta
                        )
                    )
                elif event.type == 'response.completed':
                    # Print a newline at the end
                    yield encoder.encode(
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=message_id
                        )
                    )
                    print()


        # Send text message end event
        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id
            )
        )

        # Send run finished event
        yield encoder.encode(
          RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=input_data.thread_id,
            run_id=input_data.run_id
          )
        )

    return StreamingResponse(
        event_generator(),
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