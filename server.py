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
  TextMessageEndEvent
)
from ag_ui.encoder import EventEncoder
import uuid
from openai import OpenAI

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

        # Convert input messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in input_data.messages
        ]

        # Generate a message ID for the assistant's response
        message_id = uuid.uuid4()

        # Send text message start event
        yield encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=str(message_id),
                role="assistant"
            )
        )

        # Create a streaming completion request
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=openai_messages,
            stream=True
        )

        # Process the streaming response and send content events
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield encoder.encode(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=str(message_id),
                        delta=content
                    )
                )

        # Send text message end event
        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=str(message_id)
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
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)