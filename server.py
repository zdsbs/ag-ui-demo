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
from functools import singledispatchmethod

# NOTE: The exact event types might need to be adjusted based on your version of the openai library.
from openai.types.responses import (
    Response, 
    ResponseTextDeltaEvent, 
    ResponseCompletedEvent, 
    ResponseOutputItemAddedEvent, 
    ResponseFunctionCallArgumentsDeltaEvent
)

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

def send_ui_text_message_start(encoder, message_id, role="assistant"):
    yield encoder.encode(
        TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role=role
        )
    )

def send_ui_run_started(encoder, thread_id, run_id):
    yield encoder.encode(
        RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id
        )
    )

def send_ui_custom_event_agent_name(encoder, agent_name):
    yield encoder.encode(
        CustomEvent(
            type=EventType.CUSTOM,
            name="agent_name",
            value=agent_name
        )
    )

def send_ui_text_message_content(encoder, message_id, delta):
    yield encoder.encode(
        TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=message_id,
            delta=delta
        )
    )

def send_ui_text_message_end(encoder, message_id):
    yield encoder.encode(
        TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id
        )
    )

def send_ui_tool_call_args(encoder, tool_call_id, delta):
    yield encoder.encode(
        ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=delta
        )
    )

def send_ui_tool_call_start(encoder, tool_call_id, tool_call_name, parent_message_id):
    yield encoder.encode(
        ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=tool_call_id,
            tool_call_name=tool_call_name,
            parent_message_id=parent_message_id
        )
    )

def send_ui_tool_call_end(encoder, tool_call_id):
    yield encoder.encode(
        ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=tool_call_id
        )
    )

def send_ui_run_finished(encoder, thread_id, run_id):
    yield encoder.encode(
        RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=thread_id,
            run_id=run_id
        )
    )

class EventHandler:
    def __init__(self, encoder: EventEncoder, message_id: str):
        self.encoder = encoder
        self.message_id = message_id
        self.tool_calls = {}
        self.streamed_response = ""

    def handle_stream(self, stream):
        """Processes an entire stream of events, yielding encoded SSE events."""
        self.tool_calls = {}
        self.streamed_response = ""
        for event in stream:
            generated_events = self.process_event(event)
            if generated_events:
                yield from generated_events

    @singledispatchmethod
    def process_event(self, event):
        """Default handler for unknown event types. Returns None."""
        pass

    @process_event.register
    def _(self, event: ResponseTextDeltaEvent):
        """Handles text delta events and yields a TextMessageContentEvent."""
        print(event.delta, end='', flush=True)
        self.streamed_response += event.delta
        return send_ui_text_message_content(self.encoder, self.message_id, event.delta)

    @process_event.register
    def _(self, event: ResponseCompletedEvent):
        """Handles the completion of a text message and yields a TextMessageEndEvent."""
        print(f"\n--- Streamed Message ---\n{self.streamed_response}\n----------------------")
        return send_ui_text_message_end(self.encoder, self.message_id)

    @process_event.register
    def _(self, event: ResponseOutputItemAddedEvent):
        """Handles the addition of a new item, specifically looking for function calls."""
        if hasattr(event, "item") and event.item.type == "function_call":
            self.tool_calls[event.output_index] = event.item

    @process_event.register
    def _(self, event: ResponseFunctionCallArgumentsDeltaEvent):
        """Handles streaming arguments for a function call, yielding ToolCallArgsEvents."""
        index = event.output_index
        if index in self.tool_calls:
            self.tool_calls[index].arguments += event.delta
            return send_ui_tool_call_args(self.encoder, event.item_id, event.delta)

    def prepare_next_turn(self, input_messages: list):
        """
        If any tool calls were received, this method executes them, appends the results
        to the message history, and yields the appropriate UI events.
        """
        if not self.tool_calls:
            return

        # This implementation only handles the first tool call.
        tool_call_obj = next(iter(self.tool_calls.values()))
        
        yield from send_ui_tool_call_start(self.encoder, tool_call_obj.id, tool_call_obj.name, self.message_id)
        yield from send_ui_tool_call_args(self.encoder, tool_call_obj.id, tool_call_obj.arguments)

        input_messages.append(tool_call_obj)

        try:
            args = json.loads(tool_call_obj.arguments)
            result = get_weather(args["latitude"], args["longitude"])
        except (json.JSONDecodeError, KeyError) as e:
            result = f"Error processing tool arguments: {e}"

        input_messages.append({
            "type": "function_call_output",
            "call_id": tool_call_obj.call_id,
            "output": str(result)
        })
        
        yield from send_ui_tool_call_end(self.encoder, tool_call_obj.id)


def event_generator(input_data, agent_name):
    # Create an event encoder to properly format SSE events
    encoder = EventEncoder()
    if agent_name:
        yield from send_ui_custom_event_agent_name(encoder, agent_name)
    # Send run started event
    yield from send_ui_run_started(encoder, input_data.thread_id, input_data.run_id)

    # Initialize OpenAI client
    client = OpenAI()

    # Generate a message ID for the assistant's response
    message_id = str(uuid.uuid4())

    # Send text message start event for the entire interaction
    yield from send_ui_text_message_start(encoder, message_id, "assistant")
    
    input_messages = [{"role": "user", "content": input_data.messages[0].content}]
    handler = EventHandler(encoder, message_id)
    active_tools = tools

    while True:
        stream = client.responses.create(
            model=OPENAI_MODEL,
            input=input_messages,
            tools=active_tools,
            stream=True
        )

        yield from handler.handle_stream(stream)
        
        if handler.tool_calls:
            yield from handler.prepare_next_turn(input_messages)
            # We've used a tool, so the next turn should just be a text response.
            active_tools = None 
        else:
            # No tool calls, so the agent is done.
            break

    # Send run finished event
    yield from send_ui_run_finished(encoder, input_data.thread_id, input_data.run_id)

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