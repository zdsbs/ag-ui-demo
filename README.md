# AG-UI Demo

This is a demo project implementing the AG-UI protocol with OpenAI integration.

## Prerequisites

- Python 3.11
- pip (Python package manager)
- OpenAI API key

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key
```

## Running the Server

Start the server with:
```bash
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`.

## Testing the Endpoint

You can test the endpoint using curl:

```bash
curl -X POST http://localhost:8000/awp \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "test-thread",
    "run_id": "test-run",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 