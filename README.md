# DocQA — Personal Document Q&A

A RAG (Retrieval-Augmented Generation) application that lets you upload PDFs and ask questions about their content. Answers are grounded in the document with source citations.

## Features

- **PDF Upload & Indexing** — Upload any PDF; it's chunked and embedded into a FAISS vector store
- **Conversational Q&A** — Ask natural-language questions and get cited answers
- **Configurable LLMs** — Swap models via `llms.json` (default: Groq Llama 3 70B)
- **Local Embeddings** — Uses `all-MiniLM-L6-v2` from HuggingFace (no API key needed)
- **Clean UI** — Responsive chat interface with drag-and-drop upload

## Project Structure

```
project/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Load and parse llms.json
│   ├── llm_factory.py       # Dynamically instantiate LLMs
│   ├── rag_pipeline.py      # Chunking, embedding, retrieval, generation
│   ├── vector_store.py      # In-memory FAISS store management
│   ├── llms.json            # LLM configurations
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── uploads/                 # Temporary PDF storage (auto-cleaned)
├── .env.example
└── README.md
```

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your Groq API key

Copy the example env file and fill in your key:

```bash
# Linux / macOS
export GROQ_API_KEY=your_key_here

# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"
```

You can get a free API key at [https://console.groq.com](https://console.groq.com).

### 3. Run the server

```bash
cd backend
uvicorn main:app --reload
```

The app will be available at **http://127.0.0.1:8000**.

## Usage

1. Open `http://127.0.0.1:8000` in your browser
2. Select an LLM from the sidebar dropdown
3. Drag-and-drop (or click to browse) a PDF file and click **Upload & Index**
4. Start asking questions in the chat panel — answers will include source citations

## Adding More LLMs

Edit `backend/llms.json` to add new models. Each entry needs:

```json
{
  "My-Model": {
    "import_module": "langchain_groq",
    "import_class": "ChatGroq",
    "display_name": "My Model Name",
    "max_input_chars": 25000,
    "config": {
      "model_name": "model-id",
      "api_key": "ENV:MY_API_KEY",
      "temperature": 0.3
    }
  }
}
```

Set `"managerLLM"` to the key of your default model.

## Tech Stack

- **Backend:** FastAPI, LangChain, FAISS, HuggingFace Embeddings, PyPDF
- **Frontend:** Vanilla HTML / CSS / JavaScript
- **LLM:** Groq (qwen/qwen3-32b) via LangChain
