# RAG System

This repository contains a small personal retrieval-augmented generation (RAG) application with:

- a FastAPI backend for question answering and PDF ingestion
- a Streamlit frontend for chat and file upload
- an in-memory vector store built with LangChain
- an OpenAI-backed embedding model and chat model

The system boots with content from `data/note.txt` and can ingest additional PDF files through the API.

## Architecture

The project is split into two services:

- `api.py`: FastAPI service exposing `/invoke`, `/upload`, and `/health`
- `app.py`: Streamlit chat UI that talks to the API
- `mini_rag.py`: RAG pipeline setup, chunking, embedding, retrieval, and response generation

At startup, the backend:

1. Loads `data/note.txt`
2. Chunks the text
3. Builds an `InMemoryVectorStore` using OpenAI embeddings
4. Creates a LangGraph pipeline with retrieval and response steps

When a PDF is uploaded:

1. The file is saved into `data/`
2. `PDFPlumberLoader` reads the document
3. The document is chunked with `RecursiveCharacterTextSplitter`
4. The chunks are added to the same in-memory vector store

## Requirements

- Python 3.10+
- An OpenAI API key
- `uv` for the simplest local setup

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key
```

## Local Development

Install dependencies with `uv`:

```bash
uv sync
```

Start the backend:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

Start the frontend in another terminal:

```bash
uv run streamlit run app.py
```

Open:

- Streamlit UI: `http://localhost:8501`
- FastAPI service: `http://localhost:8000`

## Docker Compose

The repository includes a two-service Docker Compose setup:

```bash
docker compose up --build
```

This starts:

- `api` on port `8000`
- `app` on port `8501`

The API container loads `.env`, and `./data` is mounted into `/app/data` so uploaded files persist on the host.

## API Endpoints

### `GET /health`

Health check endpoint.

Example response:

```json
{"status":"ok"}
```

### `POST /invoke`

Ask a question against the current indexed content.

Request body:

```json
{
  "question": "What is in my notes?"
}
```

Response body:

```json
{
  "answer": "..."
}
```

### `POST /upload`

Upload a PDF to add it to the current in-memory knowledge base.

Form field:

- `file`: PDF file

Example success response:

```json
{
  "status": "ok",
  "file_name": "example.pdf",
  "saved_path": "./data/example.pdf",
  "pages": 10,
  "chunks": 42
}
```

## Project Files

- `app.py`: Streamlit client
- `api.py`: FastAPI server
- `mini_rag.py`: RAG logic
- `compose.yaml`: local multi-container setup
- `Dockerfile.service`: backend image
- `Dockerfile.app`: frontend image
- `data/`: seed data and uploaded PDFs

## Current Behavior and Constraints

- The vector store is in memory, so indexed state is rebuilt when the API process restarts.
- The initial knowledge base comes from `data/note.txt`.
- Uploaded PDFs are saved to disk, but their embeddings are not persisted separately.
- The frontend is configured to upload only PDF files.
- The backend uses OpenAI models directly, so `OPENAI_API_KEY` must be present before startup.

## Example Workflow

1. Start the API and Streamlit app
2. Open the Streamlit UI
3. Ask a question based on `data/note.txt`
4. Upload a PDF
5. Ask follow-up questions using the newly ingested content

## Notes

This README describes the repository as it exists today. It reflects the current implementation rather than an idealized production architecture.
