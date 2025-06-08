from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
import os
from dotenv import load_dotenv


app = FastAPI(title="AG-UI Endpoint")
# Load environment variables from .env file
load_dotenv()

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

        # Convert AG-UI messages to OpenAI messages format
        openai_messages = []
        for msg in input_data.messages:
            if msg.role in ["user", "system", "assistant"]:
                # Only include the content if it's a text message
                if getattr(msg, 'type', 'text') == 'text':
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

        # Create a streaming completion request using Responses API
        print("Sending request to OpenAI with messages:", openai_messages)
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=openai_messages,
            tools=[],
            tool_choice="auto",
            stream=True
        )
        # Process the streaming response
        print("Starting to process stream...")
        for chunk in stream:
            print("Received chunk:", chunk)
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                if content:  # Only send non-empty content
                    print("Sending text content:", content)
                    event = TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=content
                    )
                    encoded = encoder.encode(event)
                    print("Encoded event:", encoded)
                    yield encoded
            
            # Handle tool calls if present
            if hasattr(chunk, "tool_calls"):
                for tool_call in chunk.tool_calls:
                    if tool_call.type == "function_call":
                        # Send tool call start event
                        yield encoder.encode(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=tool_call.call_id,
                                tool_call_name=tool_call.name,
                                parent_message_id=message_id
                            )
                        )
                        
                        # Send tool call args event
                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=tool_call.call_id,
                                args=tool_call.arguments
                            )
                        )
                        
                        # Send tool call end event
                        yield encoder.encode(
                            ToolCallEndEvent(
                                type=EventType.TOOL_CALL_END,
                                tool_call_id=tool_call.call_id
                            )
                        )

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