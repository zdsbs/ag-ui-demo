# Server TODOs

- Build `input_messages` from the entire `input_data.messages` list to preserve chat history for multi-turn conversations.
  - This will ensure the OpenAI API sees the full conversation and handles multi-turn chat correctly.

- Update the client/server call to send the full message history, not just the latest message.
  - For example, with curl, the JSON payload should look like:

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi, how can I help you?"},
    {"role": "user", "content": "What's the weather in Paris?"}
  ],
  "thread_id": "your-thread-id",
  "run_id": "your-run-id",
  "state": {"agent": "your-agent-name"}
}
```

- Your curl command might look like:

```sh
curl -X POST http://localhost:8000/awp \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Where `payload.json` contains the full message history as shown above. 