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

from dotenv import load_dotenv

# OpenAI model to use
OPENAI_MODEL = "gpt-4"

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

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in input_data.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        print(openai_tools)
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
            model=OPENAI_MODEL,
            messages=openai_messages,
            tools=[],
            stream=True
        )

        # Track the current tool call
        current_tool_call = None
        current_args = ""
        tool_call_results = []
        assistant_message = None

        # Process the streaming response and send content events
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                    yield encoder.encode(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=str(message_id),
                            delta=content
                        )
                    )
                elif hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                    for tool_call in delta.tool_calls:
                        if tool_call.id and tool_call.function.name:
                            # This is the start of a new tool call
                            if current_tool_call is not None:
                                # Send the previous tool call events
                                yield encoder.encode(
                                    ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=str(current_tool_call.id),
                                        tool_call_name=current_tool_call.function.name,
                                        parent_message_id=str(message_id)
                                    )
                                )
                                yield encoder.encode(
                                    ToolCallArgsEvent(
                                        type=EventType.TOOL_CALL_ARGS,
                                        tool_call_id=str(current_tool_call.id),
                                        delta=current_args
                                    )
                                )
                                yield encoder.encode(
                                    ToolCallEndEvent(
                                        type=EventType.TOOL_CALL_END,
                                        tool_call_id=str(current_tool_call.id)
                                    )
                                )
                                # Add the tool call result to the list
                                tool_call_results.append({
                                    "role": "tool",
                                    "tool_call_id": str(current_tool_call.id),
                                    "name": current_tool_call.function.name,
                                    "content": "Tool call completed"  # Replace with actual tool result
                                })
                            # Start tracking the new tool call
                            current_tool_call = tool_call
                            current_args = tool_call.function.arguments
                        elif current_tool_call is not None and tool_call.function.arguments:
                            # This is a continuation of the current tool call
                            current_args += tool_call.function.arguments

        # Send any remaining tool call events
        if current_tool_call is not None:
            yield encoder.encode(
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=str(current_tool_call.id),
                    tool_call_name=current_tool_call.function.name,
                    parent_message_id=str(message_id)
                )
            )
            yield encoder.encode(
                ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=str(current_tool_call.id),
                    delta=current_args
                )
            )
            yield encoder.encode(
                ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=str(current_tool_call.id)
                )
            )
            # Add the final tool call result
            tool_call_results.append({
                "role": "tool",
                "tool_call_id": str(current_tool_call.id),
                "name": current_tool_call.function.name,
                "content": "Tool call completed"  # Replace with actual tool result
            })

        # If we have tool call results, make another request to get the final response
        if tool_call_results:
            # Add the assistant's message with tool calls
            openai_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": str(current_tool_call.id),
                    "type": "function",
                    "function": {
                        "name": current_tool_call.function.name,
                        "arguments": current_args
                    }
                }]
            })
            print(openai_messages)
            # Add tool call results to messages
            openai_messages.extend(tool_call_results)
            
            # Make another request to get the final response
            final_stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=openai_messages,
                stream=True
            )

            # Process the final response
            for chunk in final_stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content)
                        content = delta.content
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