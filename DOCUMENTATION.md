# DocTalk — Complete Project Documentation

### A Beginner's Guide to Building a RAG-Based Document Q&A Application

**Author:** Shivam Kumar  
**Version:** 1.0  
**Date:** March 2026

---

## Table of Contents

1. [What is DocTalk?](#1-what-is-doctalk)
2. [What is RAG?](#2-what-is-rag-retrieval-augmented-generation)
3. [Tech Stack Explained](#3-tech-stack-explained)
4. [Project Structure](#4-project-structure)
5. [How the App Works (End-to-End Flow)](#5-how-the-app-works-end-to-end-flow)
6. [Backend Deep Dive](#6-backend-deep-dive)
   - 6.1 [main.py — The API Server](#61-mainpy--the-api-server)
   - 6.2 [config.py — Configuration Loader](#62-configpy--configuration-loader)
   - 6.3 [llm_factory.py — Dynamic LLM Loading](#63-llm_factorypy--dynamic-llm-loading)
   - 6.4 [rag_pipeline.py — The RAG Brain](#64-rag_pipelinepy--the-rag-brain)
   - 6.5 [vector_store.py — Session Memory](#65-vector_storepy--session-memory)
   - 6.6 [llms.json — LLM Configuration File](#66-llmsjson--llm-configuration-file)
7. [Frontend Deep Dive](#7-frontend-deep-dive)
   - 7.1 [index.html — Page Structure](#71-indexhtml--page-structure)
   - 7.2 [style.css — Styling & Animations](#72-stylecss--styling--animations)
   - 7.3 [app.js — Application Logic](#73-appjs--application-logic)
8. [Key Concepts for Beginners](#8-key-concepts-for-beginners)
9. [Setup Guide](#9-setup-guide)
10. [API Reference](#10-api-reference)
11. [Common Issues & Troubleshooting](#11-common-issues--troubleshooting)
12. [What You Can Learn From This Project](#12-what-you-can-learn-from-this-project)

---

## 1. What is DocTalk?

DocTalk is a **Personal Document Q&A** web application. Users upload a PDF document, and the app lets them ask natural-language questions about its content. The answers come directly from the document, with citations showing which pages the information came from.

### Example Use Case
1. You upload a 50-page research paper on climate change
2. You type: "What are the main causes of rising sea levels?"
3. DocTalk searches the document, finds the relevant paragraphs, sends them to an AI model, and returns a precise answer with page references

---

## 2. What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that makes AI models smarter by giving them relevant information before asking them to answer. Here's the problem it solves:

### The Problem
Large Language Models (LLMs) like GPT, Llama, etc. are trained on general data. They don't know the content of YOUR specific documents. If you ask them about your PDF, they'll either hallucinate (make up answers) or say they don't know.

### The Solution — RAG
RAG works in 3 steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
│                                                                 │
│  STEP 1: INDEXING (happens once, when PDF is uploaded)          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐              │
│  │  Upload   │───▶│  Split   │───▶│  Convert to  │──▶ Store    │
│  │   PDF     │    │  into    │    │  Embeddings  │   in FAISS  │
│  │           │    │  Chunks  │    │  (vectors)   │              │
│  └──────────┘    └──────────┘    └──────────────┘              │
│                                                                 │
│  STEP 2: RETRIEVAL (happens for each question)                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  User's   │───▶│  Convert to  │───▶│  Search for  │          │
│  │  Question │    │  Embedding   │    │  similar     │          │
│  │           │    │  (vector)    │    │  chunks      │          │
│  └──────────┘    └──────────────┘    └──────────────┘          │
│                                              │                  │
│  STEP 3: GENERATION                          ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │  Send question + relevant chunks to the LLM      │          │
│  │  LLM reads the chunks and generates an answer     │          │
│  │  Answer includes citations [Context 1], etc.      │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### What are Embeddings?
Embeddings are numerical representations (vectors) of text. Similar meanings produce similar vectors. For example:
- "dog" → [0.2, 0.8, 0.1, ...]
- "puppy" → [0.21, 0.79, 0.12, ...]  (very close to "dog")
- "car" → [0.9, 0.1, 0.7, ...]  (far from "dog")

This lets us find text chunks that are **semantically similar** to a question, even if they don't share exact keywords.

---

## 3. Tech Stack Explained

### Backend

| Technology | What It Is | Why We Use It |
|---|---|---|
| **Python 3.12** | Programming language | Most popular for AI/ML; vast library ecosystem |
| **FastAPI** | Web framework | Modern, fast, auto-generates API docs, async support |
| **Uvicorn** | ASGI server | Runs the FastAPI app; supports async/await |
| **LangChain** | AI orchestration framework | Provides abstractions for LLMs, embeddings, vector stores |
| **langchain_groq** | Groq integration for LangChain | Connects to Groq's ultra-fast LLM inference API |
| **FAISS** | Vector database (by Facebook AI) | Efficient similarity search over embeddings |
| **HuggingFace Embeddings** | Embedding model | Converts text to vectors locally (free, no API key) |
| **PyPDF** | PDF parser | Extracts text from PDF files |
| **Pydantic** | Data validation | Validates API request/response schemas |
| **python-dotenv** | Environment loader | Reads `.env` file for API keys |

### Frontend

| Technology | What It Is | Why We Use It |
|---|---|---|
| **HTML5** | Page structure | Standard markup language |
| **CSS3** | Styling & animations | Custom properties (variables), flexbox, keyframes |
| **Vanilla JavaScript** | Application logic | No framework needed — keeps it simple and educational |

### External Services

| Service | What It Does | Cost |
|---|---|---|
| **Groq API** | Runs the LLM (e.g., Llama 3, Qwen) | Free tier available |
| **all-MiniLM-L6-v2** | Embedding model | 100% free, runs locally |

---

## 4. Project Structure

```
rag-project/
├── backend/
│   ├── main.py              # FastAPI server — all API routes
│   ├── config.py            # Reads llms.json, provides helper functions
│   ├── llm_factory.py       # Creates LLM instances dynamically
│   ├── rag_pipeline.py      # Core RAG: chunk → embed → retrieve → generate
│   ├── vector_store.py      # In-memory store for FAISS indexes per session
│   ├── llms.json            # LLM model configurations
│   └── requirements.txt     # Python dependencies with pinned versions
├── frontend/
│   ├── index.html           # Single-page HTML layout
│   ├── style.css            # All styling and responsive design
│   └── app.js               # Client-side application logic
├── uploads/                 # Temporary PDF storage (auto-cleaned)
├── .env                     # Your API keys (not committed to git)
├── .env.example             # Template showing required env vars
├── .gitignore               # Files excluded from git
└── README.md                # Quick-start guide
```

---

## 5. How the App Works (End-to-End Flow)

### Flow 1: Page Load
```
Browser                        Server
  │                              │
  │  GET /llms                   │
  │─────────────────────────────▶│  config.py reads llms.json
  │                              │  Returns list of available LLMs
  │◀─────────────────────────────│
  │  Populates dropdown          │
  │  Generates session UUID      │
```

### Flow 2: Upload PDF
```
Browser                        Server
  │                              │
  │  POST /upload                │
  │  (file + session_id)         │
  │─────────────────────────────▶│  1. Save PDF to uploads/
  │                              │  2. PyPDF extracts text from each page
  │                              │  3. RecursiveCharacterTextSplitter chunks it
  │                              │  4. HuggingFace model embeds each chunk
  │                              │  5. FAISS indexes all embeddings
  │                              │  6. Store FAISS index under session_id
  │                              │  7. Delete the uploaded PDF file
  │◀─────────────────────────────│
  │  { chunk_count, filename }   │
  │  Enable chat input           │
```

### Flow 3: Ask a Question
```
Browser                        Server
  │                              │
  │  POST /ask                   │
  │  {question, session_id,      │
  │   llm_id}                    │
  │─────────────────────────────▶│  1. Get FAISS store for session
  │                              │  2. Embed the question
  │                              │  3. FAISS finds top 4 similar chunks
  │                              │  4. Build prompt with chunks + question
  │                              │  5. Send to LLM (via Groq API)
  │                              │  6. Strip <think> tags from response
  │                              │  7. Return answer + source snippets
  │◀─────────────────────────────│
  │  Show answer bubble          │
  │  Show collapsible sources    │
```

### Flow 4: Session Cleanup
```
Browser                        Server
  │                              │
  │  (user closes tab)           │
  │  DELETE /session/{id}        │
  │─────────────────────────────▶│  Remove FAISS store from memory
  │                              │
```

---

## 6. Backend Deep Dive

### 6.1 main.py — The API Server

This is the entry point. It sets up FastAPI and defines all the HTTP routes.

**Key Concepts:**

```python
# Load environment variables FIRST, before any other imports need them
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
```
This reads your `.env` file so `GROQ_API_KEY` is available via `os.environ`.

```python
# CORS middleware — allows the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow any origin (for development)
    allow_methods=["*"],      # Allow GET, POST, DELETE, etc.
    allow_headers=["*"],      # Allow any headers
)
```
Without CORS, browsers block requests from a webpage to a different origin.

**Routes Summary:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/llms` | List available LLM models |
| POST | `/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question about the uploaded PDF |
| DELETE | `/session/{id}` | Clean up session data |

**Pydantic Models** validate request/response data:
```python
class AskRequest(BaseModel):
    question: str          # Must be a string
    session_id: str        # Must be a string
    llm_id: str           # Must be a string
# If someone sends {"question": 123}, FastAPI auto-returns a 422 error
```

**Static File Serving:**
```python
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
```
This serves `frontend/index.html` at the root URL, so no separate web server is needed.

---

### 6.2 config.py — Configuration Loader

Reads `llms.json` once, caches it, and provides helper functions.

**Key Pattern — Lazy Loading with Caching:**
```python
_CONFIG = None  # Module-level cache

def _load_config():
    global _CONFIG
    if _CONFIG is None:                    # Only read file once
        with open(_CONFIG_PATH) as f:
            _CONFIG = json.load(f)
    return _CONFIG
```
This is called the **Singleton Pattern** — the config is loaded on first access and reused after.

**Functions:**
- `get_all_llms()` → Returns `[{"id": "Groq-Llama", "display_name": "qwen/qwen3-32b"}, ...]`
- `get_llm_config("Groq-Llama")` → Returns the full config dict for that LLM
- `get_default_llm_id()` → Returns `"Groq-Llama"` (from the `managerLLM` key)

---

### 6.3 llm_factory.py — Dynamic LLM Loading

This module dynamically creates LLM instances from the JSON config using Python's `importlib`.

**Why Dynamic Loading?**
Instead of hardcoding `from langchain_groq import ChatGroq`, we read the module and class names from JSON. This means you can add new LLMs just by editing `llms.json` — no code changes needed.

**How it works step by step:**

```python
# 1. Read config: import_module="langchain_groq", import_class="ChatGroq"
# 2. Dynamic import (equivalent to: from langchain_groq import ChatGroq)
module = importlib.import_module("langchain_groq")
cls = getattr(module, "ChatGroq")

# 3. Resolve environment variables
#    "ENV:GROQ_API_KEY" → os.environ["GROQ_API_KEY"] → "gsk_abc123..."
constructor_kwargs = _resolve_env_vars(config)

# 4. Convert string booleans: "false" → False
constructor_kwargs = _clean_config(constructor_kwargs)

# 5. Create instance (equivalent to: ChatGroq(model_name="...", api_key="...", ...))
instance = cls(**constructor_kwargs)
```

**Caching:**
```python
_LLM_CACHE = {}  # {"Groq-Llama": <ChatGroq instance>}

def get_llm_instance(llm_id):
    if llm_id in _LLM_CACHE:
        return _LLM_CACHE[llm_id]    # Return cached instance
    # ... create new instance ...
    _LLM_CACHE[llm_id] = instance    # Cache for next time
    return instance
```

---

### 6.4 rag_pipeline.py — The RAG Brain

This is the core of the application. It handles all 4 stages of RAG.

#### Stage 1: Load & Chunk PDF

```python
def load_and_chunk_pdf(file_path):
    # Load: PyPDFLoader reads each page as a Document object
    loader = PyPDFLoader(file_path)
    pages = loader.load()             # [Document(page_content="...", metadata={"page": 0}), ...]

    # Chunk: Split long pages into smaller overlapping pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,               # Each chunk ≤ 800 characters
        chunk_overlap=100,            # 100 chars overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # Try to split at natural boundaries
    )
    chunks = splitter.split_documents(pages)
    return chunks
```

**Why Chunk?**
- LLMs have limited context windows
- Smaller chunks = more precise retrieval
- Overlap ensures we don't cut sentences in half

**Why These Separators?**
The splitter tries `\n\n` (paragraph breaks) first, then `\n` (line breaks), then `. ` (sentences), etc. This produces natural, readable chunks.

#### Stage 2: Build Vector Store (Embedding + Indexing)

```python
def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
```

What happens inside:
1. Each chunk's text is converted to a 384-dimensional vector by `all-MiniLM-L6-v2`
2. All vectors are inserted into a FAISS index
3. FAISS organizes them for fast similarity search

**The `all-MiniLM-L6-v2` model:**
- Trained by Microsoft
- 22M parameters (tiny — runs on CPU)
- Produces 384-dimensional embeddings
- No API key needed

#### Stage 3: Retrieve Context

```python
def retrieve_context(vector_store, query, k=4):
    results = vector_store.similarity_search(query, k=k)
    return results
```

1. The user's question is also embedded into a 384-dim vector
2. FAISS finds the 4 stored vectors closest to this query vector
3. Returns the corresponding text chunks

#### Stage 4: Generate Answer

```python
def generate_answer(llm, query, context_docs):
    # Build numbered context blocks
    for i, doc in enumerate(context_docs, 1):
        context_blocks.append(f"[Context {i}] (Page {page}):\n{doc.page_content}")

    prompt = f"""You are a helpful document Q&A assistant. 
    Answer based ONLY on the provided context. 
    Cite which context block(s) you used.
    
    --- CONTEXT ---
    {context_str}
    
    --- QUESTION ---
    {query}"""

    response = llm.invoke(prompt)
    
    # Clean up: remove <think> tags from reasoning models
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer
```

**The Prompt Engineering:**
- "Answer based ONLY on the provided context" — prevents hallucination
- "Cite which context block(s)" — makes answers verifiable
- Numbered context blocks — makes citations unambiguous

**The `<think>` tag stripping:**
Some models (like Qwen) include their reasoning process in `<think>...</think>` tags. We strip these so users only see the final answer.

---

### 6.5 vector_store.py — Session Memory

A simple in-memory dictionary mapping session IDs to FAISS indexes.

```python
_STORES = {}
# Example: {"a1b2c3d4-...": <FAISS index for user A's PDF>,
#            "e5f6g7h8-...": <FAISS index for user B's PDF>}
```

**Why sessions?**
Each user/tab gets a unique `session_id`. This lets multiple users upload different PDFs simultaneously without conflicts.

**Trade-off:** In-memory means all data is lost if the server restarts. For a production app, you'd persist FAISS indexes to disk.

---

### 6.6 llms.json — LLM Configuration File

```json
{
  "managerLLM": "Groq-Llama",          // Default LLM to use
  "Groq-Llama": {                       // LLM ID (key)
    "import_module": "langchain_groq",  // Python module to import
    "import_class": "ChatGroq",         // Class to instantiate
    "display_name": "qwen/qwen3-32b",  // Shown in the UI dropdown
    "max_input_chars": 25000,           // Max context length
    "config": {                         // Constructor arguments
      "model_name": "qwen/qwen3-32b",
      "api_key": "ENV:GROQ_API_KEY",   // "ENV:" prefix → read from environment
      "temperature": 0.3,              // Lower = more deterministic
      "max_tokens": 2048,              // Max output length
      "streaming": "false"
    }
  }
}
```

**Adding a new model is easy — just add another JSON block. No code changes needed.**

---

## 7. Frontend Deep Dive

### 7.1 index.html — Page Structure

The HTML is organized into 3 main areas:

```
┌────────────────────────────────────────────────┐
│  NAVBAR (fixed top)        [DocTalk]   [≡]     │
├──────────┬─────────────────────────────────────┤
│          │                                      │
│ SIDEBAR  │          CHAT PANEL                  │
│ (280px)  │                                      │
│          │   ┌──────────────────────────────┐   │
│ [LLM ▼]  │   │    User bubble (right)    ◀──│   │
│          │   │                               │   │
│ [Upload] │   │  ◀── Assistant bubble (left)  │   │
│          │   │       ▶ Sources (4)           │   │
│ [Status] │   │                               │   │
│          │   └──────────────────────────────┘   │
│ [Badge]  │                                      │
│          │   ┌──────────────────────┬────┐      │
│ © Shivam │   │  Ask a question...   │ ➤  │      │
│          │   └──────────────────────┴────┘      │
└──────────┴─────────────────────────────────────┘
```

**Key HTML elements:**
- `<select>` for LLM dropdown
- `<div class="drop-zone">` for drag-and-drop file upload
- `<div class="chat-messages">` as scrollable message container
- `<details>` for collapsible source citations

### 7.2 style.css — Styling & Animations

**CSS Custom Properties (Variables):**
```css
:root {
  --accent: #4f6ef7;        /* Primary blue */
  --sidebar-bg: #f0f4ff;    /* Light blue sidebar */
  --bubble-user: #4f6ef7;   /* User message = blue */
  --bubble-assistant: #f1f5f9; /* Assistant = light gray */
}
```
Changing `--accent` updates the entire color scheme!

**Chat Bubble Animation:**
```css
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(10px); }  /* Start: invisible, 10px below */
  to   { opacity: 1; transform: translateY(0); }     /* End: visible, normal position */
}
.chat-bubble { animation: fadeSlideUp .3s ease both; }
```

**Typing Indicator (3 bouncing dots):**
```css
.typing-indicator span {
  animation: typingBounce .6s infinite alternate;
}
.typing-indicator span:nth-child(2) { animation-delay: .15s; }  /* Staggered */
.typing-indicator span:nth-child(3) { animation-delay: .3s; }
```

**Responsive Design:**
```css
@media (max-width: 768px) {
  .sidebar {
    position: fixed;
    transform: translateX(-100%);  /* Hidden off-screen */
  }
  .sidebar.open {
    transform: translateX(0);      /* Slide in when toggled */
  }
  .sidebar-toggle { display: block; }  /* Show hamburger menu */
}
```

### 7.3 app.js — Application Logic

The entire app runs as an **IIFE** (Immediately Invoked Function Expression):
```javascript
(function () {
  "use strict";
  // All code here — no global variables leak
})();
```

**Session Management:**
```javascript
let sessionId = crypto.randomUUID();  
// Generates something like: "3b241101-e2bb-4d7e-8b52-5d6a0c9c1f23"
// Each browser tab gets its own unique session
```

**Key Flows:**

1. **Fetch LLMs on page load** → populates the `<select>` dropdown
2. **File drag-and-drop** → validates it's a PDF, shows filename
3. **Upload** → sends `FormData` with file + session_id, shows spinner
4. **Ask** → sends JSON, shows typing indicator, then renders answer + sources
5. **Cleanup on tab close:**
```javascript
window.addEventListener("beforeunload", () => {
  navigator.sendBeacon(`/session/${sessionId}`, "");
  // sendBeacon is reliable even during page unload
});
```

**Source Citations Display:**
```javascript
// Uses <details> for collapsible sections
const details = document.createElement("details");
const summary = document.createElement("summary");
summary.textContent = `Sources (${sources.length})`;
// Page numbers are 0-indexed from PyPDF, so we add 1:
`Page ${parseInt(src.page) + 1}`
```

---

## 8. Key Concepts for Beginners

### What is an API?
An API (Application Programming Interface) is a set of rules for how programs communicate. In DocTalk, the frontend sends HTTP requests to the backend API, and the backend responds with JSON data.

### What is CORS?
Cross-Origin Resource Sharing. Browsers block requests between different domains for security. CORS headers tell the browser "it's okay, this server accepts requests from other origins."

### What is a Vector Database?
A specialized database that stores numerical vectors and supports fast "nearest neighbor" search. FAISS (Facebook AI Similarity Search) is one of the most popular, optimized for speed.

### What is an Embedding Model?
A neural network that converts text into fixed-size numerical vectors where similar meanings are close together in vector space. `all-MiniLM-L6-v2` is a small, efficient model that runs locally.

### What is LangChain?
A framework that provides standardized interfaces for working with LLMs, embeddings, vector stores, document loaders, and text splitters. It's like an "adapter layer" that lets you swap components easily.

### What is Pydantic?
A Python library for data validation. You define a class with type hints, and Pydantic automatically validates incoming data, converts types, and raises clear errors.

### What is a Session?
A way to associate data with a specific user/tab. DocTalk uses UUIDs (Universally Unique Identifiers) generated in the browser. Each session has its own vector store.

---

## 9. Setup Guide

### Prerequisites
- Python 3.12 installed
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/ShivamKumarIND/rag-project.git
cd rag-project

# 2. Create a virtual environment with Python 3.12
py -3.12 -m venv venv

# 3. Activate it
# Windows CMD:
venv\Scripts\activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r backend\requirements.txt

# 5. Create .env file with your API key
#    Create a file called .env in the project root:
#    GROQ_API_KEY=gsk_your_key_here

# 6. Run the server
cd backend
uvicorn main:app --reload

# 7. Open in browser
# http://127.0.0.1:8000
```

---

## 10. API Reference

### GET /llms
Returns all available LLMs and the default selection.

**Response:**
```json
{
  "llms": [
    { "id": "Groq-Llama", "display_name": "qwen/qwen3-32b" }
  ],
  "default": "Groq-Llama"
}
```

### POST /upload
Upload a PDF file for indexing.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| file | File | PDF file to upload |
| session_id | string | Unique session identifier |

**Response:**
```json
{
  "message": "PDF uploaded and indexed",
  "session_id": "3b241101-e2bb-...",
  "chunk_count": 137,
  "filename": "RAMAYANA.pdf"
}
```

### POST /ask
Ask a question about the uploaded document.

**Request:** `application/json`
```json
{
  "question": "Who is Ram?",
  "session_id": "3b241101-e2bb-...",
  "llm_id": "Groq-Llama"
}
```

**Response:**
```json
{
  "answer": "Ram (Rama) is the central protagonist of the Ramayana...",
  "sources": [
    { "page": 2, "snippet": "Rama was born in Ayodhya..." },
    { "page": 5, "snippet": "The prince Rama was exiled..." }
  ]
}
```

### DELETE /session/{session_id}
Clean up the session's vector store.

**Response:**
```json
{ "message": "Session cleared" }
```

---

## 11. Common Issues & Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| "Could not load LLM list" toast | Server not running | Run `uvicorn main:app --reload` in backend/ |
| "Environment variable 'GROQ_API_KEY' is not set" | Missing .env file | Create `.env` in project root with your key |
| Upload hangs for a long time | First-time embedding model download | Wait — `all-MiniLM-L6-v2` (~80MB) downloads on first use |
| "Failed to load LLM" | Invalid API key | Check your Groq key at console.groq.com |
| Empty/bad answers | PDF has scanned images, not text | Use a text-based PDF; scanned PDFs need OCR |
| `<think>` tags in response | Model includes reasoning | Already handled — `re.sub` strips these tags |

---

## 12. What You Can Learn From This Project

| Skill | Where in the Project |
|---|---|
| **Building REST APIs** | main.py — FastAPI routes, status codes, error handling |
| **File uploads** | main.py — multipart form handling, temp file management |
| **Data validation** | main.py — Pydantic models (AskRequest, AskResponse) |
| **Dynamic imports** | llm_factory.py — `importlib.import_module()` |
| **Environment variables** | llm_factory.py — `os.environ`, dotenv |
| **Design patterns** | config.py — Singleton/caching; llm_factory.py — Factory pattern |
| **NLP & Embeddings** | rag_pipeline.py — text splitting, vector embeddings |
| **Vector search** | rag_pipeline.py — FAISS similarity search |
| **Prompt engineering** | rag_pipeline.py — structured prompts with context + citations |
| **Frontend API calls** | app.js — fetch(), FormData, async/await |
| **Drag & drop** | app.js — dragover, drop events |
| **CSS animations** | style.css — @keyframes, transitions |
| **Responsive design** | style.css — @media queries, sidebar toggle |
| **Session management** | app.js — crypto.randomUUID(), sendBeacon cleanup |

---

*Built with care by Shivam Kumar. All Rights Reserved.*
