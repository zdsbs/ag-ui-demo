# Server Structure Overview

## Main Components

### 1. Streaming Response System
- Uses FastAPI's `StreamingResponse`
- Implements Server-Sent Events (SSE) with proper headers
- Uses an `EventEncoder` to format events properly
- Handles different event types (text messages, tool calls, run status)

### 2. OpenAI Streaming API Integration
- Uses the streaming interface of OpenAI's API
- Handles various event types from the API:
  - `response.created`
  - `response.output_text.delta`
  - `response.completed`
  - `response.output_item.added`
  - `response.function_call_arguments.delta`
- Manages the streaming state and message accumulation

### 3. AG-UI Server Implementation
- Implements the AG-UI protocol for agent interactions
- Handles thread and run IDs
- Manages the agent state
- Formats responses according to AG-UI's event structure
- Supports custom events (like agent name)

## Secondary Component

### Weather Tool
- A simple demonstration tool using Open-Meteo API
- Takes latitude and longitude as parameters
- Returns current temperature
- Serves as an example of how to implement tools in the system

## Note
The weather tool is just an example implementation, while the core functionality revolves around the streaming agent system and OpenAI integration. 