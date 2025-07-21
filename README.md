# RayBot: RAG‑Powered AI Agent

RayBot is a lightweight, modular Retrieval‑Augmented Generation (RAG) chatbot and document assistant designed to run on small VMs. It integrates Google Gemini for generative capabilities, ChromaDB for vector search, and FastAPI for a simple HTTP interface.

---

## 🚀 Features

* **Retrieval‑Augmented Generation**: Ingest PDFs, DOCXs, and plain text to create a custom knowledge base.
* **Google Gemini Integration**: Zero‑shot and context‑aware prompting via `gemini-1.5-flash`.
* **Vector Search**: Semantic search with `all-MiniLM-L6-v2` embeddings in ChromaDB.
* **Modular Skill Registry**: Easily add new capabilities like summarization, code generation, math solving, entity extraction, and more.
* **Memory Bank**: Persistent conversation memory with SQLite and FIFO eviction.
* **Task Manager**: Tracks asynchronous tasks, dependencies, and priorities.
* **RESTful API**: Exposed via FastAPI with endpoints for chat, document upload, status, and health checks.
* **Caching (Optional)**: Redis support for response caching.

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/raybot.git
   cd raybot
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (see below)

5. **Run the server**

   ```bash
   uvicorn backend.fastapi_backend:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## ⚙️ Configuration

Place configuration values in environment variables or a `.env` file:

| Variable          | Default                  | Description                           |
| ----------------- | ------------------------ | ------------------------------------- |
| `GEMINI_API_KEY`  | `LOL`                    | Your Google Gemini API key            |
| `GEMINI_MODEL`    | `gemini-1.5-flash`       | Gemini model name                     |
| `REDIS_URL`       | `redis://localhost:6379` | Redis connection URL for caching      |
| `USE_REDIS`       | `false`                  | Enable Redis caching (`true`/`false`) |
| `MAX_UPLOAD_SIZE` | `10485760` (10 MB)       | Maximum upload size for documents     |

---

## 🗂️ Directory Structure

```
raybot/
├─ backend/
│  ├─ aiAgent.py          # Core RAG agent implementation
│  ├─ fastapi_backend.py  # FastAPI app exposing endpoints
│  └─ requirements.txt    # Python dependencies
├─ chroma_db/            # ChromaDB persistence directory
├─ .env                  # Environment variables (optional)
└─ README.md             # This documentation
```

---

## 🚧 Usage

### 1. Chat Endpoint

**POST** `/api/process`

* **Request**

  ```json
  { "input": "Hello, RayBot!" }
  ```
* **Response**

  ```json
  {
    "response": "Hi there! How can I help?",
    "type": "general"
  }
  ```

### 2. Upload Document

**POST** `/api/upload` (multipart/form-data)

* **Form Field**: `file` – PDF, DOCX, TXT up to 10 MB
* **Response**

  ```json
  {
    "success": true,
    "message": "Successfully ingested 12 chunks from example.pdf",
    "chunks": 12
  }
  ```

### 3. Status & Health

* **GET** `/api/status` – Returns agent state, active tasks, memory count, etc.
* **GET** `/health` – Returns `{ "status": "healthy", "agent": "Raybot" }`

---

## 🔧 Extending RayBot

1. **Add a new skill** in `aiAgent.py`:

   ```python
   def translate(self, text: str) -> str:
       prompt = f"Translate to French:\n{text}"
       return self.gemini.generate_response(prompt)
   ```
2. **Register the skill**:

   ```python
   self.skill_registry.register("translate", self.translate)
   ```
3. **Update intent analysis** to return `action: "translate"` when appropriate.


