# 🏭 AI Industrial Chatbot

A document-grounded industrial chatbot built using a **Retrieval-Augmented Generation (RAG)** pipeline. The system retrieves relevant information from industrial PDF documents and uses a local Hugging Face language model to generate concise answers based only on the retrieved document context.

The application provides a **Streamlit chat interface**, displays the retrieved source documents and pages, and shows similarity scores for the retrieved chunks.

---

## 🚀 Features

- 📄 **PDF-based industrial knowledge base**
- ✂️ **Text cleaning and overlapping chunking**
- 🧠 **Semantic embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
- ⚡ **FAISS vector similarity search**
- 🤖 **Local LLM inference** using `Qwen/Qwen2.5-1.5B-Instruct`
- 🔎 **Top-K document retrieval**
- 📚 **Source and page references** for retrieved information
- 🛡️ **Document-grounded prompting** to reduce unsupported answers
- 💻 **CPU/GPU support** through PyTorch
- 🎈 **Interactive Streamlit chatbot UI**
- 💬 **Chat history and clear-chat functionality**
- 📖 Knowledge-base display for industrial documents

---

## 🏗️ System Architecture

```text
                 INDUSTRIAL PDF DOCUMENTS
                           │
                           ▼
                    PDF Text Extraction
                           │
                           ▼
                     Text Cleaning
                           │
                           ▼
                  Chunking (900 / 150)
                           │
                           ▼
              Sentence Transformer Embeddings
                           │
                           ▼
                    FAISS Vector Index
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 │                   │
             USER QUESTION       Metadata JSON
                 │
                 ▼
          Question Embedding
                 │
                 ▼
           FAISS Top-K Search
                 │
                 ▼
        Relevant Document Chunks
                 │
                 ▼
          Context + User Query
                 │
                 ▼
       Qwen2.5-1.5B-Instruct LLM
                 │
                 ▼
            Generated Answer
                 │
                 ▼
        Streamlit Chat Interface
                 │
                 └── Retrieved Sources
```

---

## 🔄 RAG Pipeline

The project is divided into two main stages.

### 1. Indexing Stage

`build_index.py` creates the searchable knowledge base.

1. Reads all PDF files from the `data/` directory.
2. Extracts text page by page using `pypdf`.
3. Normalizes unnecessary whitespace.
4. Splits the text into chunks:
   - Chunk size: **900**
   - Chunk overlap: **150**
5. Generates embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

6. Normalizes the embeddings.
7. Builds a FAISS `IndexFlatIP` index.
8. Saves:
   - `vector_db/index.faiss`
   - `vector_db/metadata.json`

The embeddings are normalized, so inner-product search corresponds to cosine similarity.

### 2. Question Answering Stage

`rag.py` handles the complete question-to-answer pipeline.

1. Converts the user's question into an embedding.
2. Searches the FAISS index.
3. Retrieves the **top 4** matching chunks by default.
4. Builds a prompt containing:
   - Source document
   - Page number
   - Retrieved content
   - User question
5. Passes the prompt to:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

6. Generates the answer.
7. Returns the answer together with source metadata and similarity scores.

The prompt explicitly instructs the model to use only the provided industrial documents and to state when the required information cannot be found.

---

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Streamlit | Web-based chatbot interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text and question embeddings |
| FAISS | Vector similarity search |
| Hugging Face Transformers | LLM loading and inference |
| Qwen 2.5 1.5B Instruct | Answer generation |
| PyTorch | Model execution and device detection |
| JSON | Document metadata storage |

---

## 📁 Project Structure

```text
AI-Industrial-Chatbot/
│
├── app.py
├── rag.py
├── build_index.py
│
├── data/
│   ├── industrial_document_1.pdf
│   ├── industrial_document_2.pdf
│   └── ...
│
├── vector_db/
│   ├── index.faiss
│   └── metadata.json
│
└── README.md
```

> Add the industrial PDFs that should form the knowledge base inside the `data/` folder.

---

## 📚 Example Knowledge Base

The current Streamlit interface presents these industrial document categories:

- ⚙️ IM-45 Induction Motor Maintenance
- 🔧 PLC-200 Operations & Troubleshooting
- 🏭 CNC-300 Operations & Safety
- ⚡ Industrial Electrical Safety Guide

These names are displayed by the interface as the intended knowledge-base documents.

---

## ⚙️ Installation

### 1. Clone or copy the project

```bash
git clone <your-repository-url>
cd AI-Industrial-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit faiss-cpu pypdf sentence-transformers transformers torch
```

> For a GPU-specific FAISS/PyTorch setup, install the appropriate packages for your hardware and environment.

---

## 📄 Add Documents

Create a `data/` directory in the project root:

```bash
mkdir data
```

Place the industrial PDF documents inside it:

```text
data/
├── motor_maintenance.pdf
├── plc_troubleshooting.pdf
├── cnc_safety.pdf
└── electrical_safety.pdf
```

The indexing script searches for `*.pdf` files in this directory.

---

## 🧠 Build the Vector Database

Before starting the chatbot, create the FAISS index:

```bash
python build_index.py
```

This generates:

```text
vector_db/
├── index.faiss
└── metadata.json
```

Run this again whenever the source PDF knowledge base is changed.

---

## ▶️ Run the Streamlit Application

Start the chatbot with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 💬 Example Usage

Example questions could include:

```text
What are the maintenance procedures for an induction motor?

How can a PLC fault be troubleshooted?

What safety precautions should be followed during CNC operation?

What are the electrical safety guidelines mentioned in the documents?
```

The chatbot retrieves relevant chunks and generates an answer from those chunks.

---

## 🔍 Retrieval Configuration

The main RAG configuration is:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K = 4

MAX_NEW_TOKENS = 384
```

The document indexing configuration is:

```python
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
```

---

## 🛡️ Grounded Answering

The RAG prompt is designed to keep the chatbot grounded in the provided documents.

The model is instructed to:

- Answer using only the retrieved industrial document context.
- Avoid inventing technical information.
- Avoid making up specifications, procedures, measurements, or values.
- Clearly state when the requested information is not available.
- Mention the source document and page when possible.

This makes the system more suitable for document-based industrial assistance than a general-purpose chatbot.

---

## 🖥️ User Interface

The Streamlit application includes:

- Industrial-themed dark UI
- Animated chatbot/industrial header
- Welcome panel
- Chat history
- Chat input
- AI thinking animation
- Retrieved-source expandable section
- Source document name
- Page number
- Similarity score
- Knowledge-base sidebar
- Technology information
- Clear-chat button

---

## 🧪 CLI Mode

`rag.py` can also be executed directly without Streamlit:

```bash
python rag.py
```

This starts a command-line chatbot where questions can be entered interactively.

Type:

```text
exit
```

to close the CLI chatbot.

---

## ⚡ CPU and GPU Support

The RAG implementation checks whether CUDA is available.

```text
CUDA available → GPU
CUDA unavailable → CPU
```

For GPU execution, the Qwen model is loaded using FP16 and automatic device mapping. For CPU execution, the model is loaded using FP32.

---

## 📊 Source Traceability

For each retrieved result, the application can display:

```text
Source document
Page number
Similarity score
Chunk ID
```

This allows the user to inspect where the retrieved information originated.

---

## ⚠️ Current Limitations

- The knowledge base is based on the PDFs placed in the `data/` directory.
- PDF pages without extractable text are skipped by the indexing script.
- The current retrieval system uses dense semantic similarity with FAISS.
- The chatbot is constrained by the information available in the indexed documents.
- Generated answers depend on the retrieved context and the local Qwen model.
- Rebuilding the index is required when source documents are changed.
- The current implementation does not describe an external cloud LLM/API dependency; the LLM is loaded locally through Hugging Face Transformers.

---

## 🔮 Possible Future Improvements

- Add PDF upload directly from the Streamlit interface.
- Add document deletion and re-indexing from the UI.
- Add configurable `TOP_K`, chunk size, and similarity thresholds.
- Add hybrid keyword + semantic retrieval.
- Add reranking for improved retrieval accuracy.
- Add conversation-aware retrieval for follow-up questions.
- Add authentication and role-based access.
- Add evaluation metrics such as retrieval recall and answer faithfulness.
- Add logging and monitoring for industrial deployments.
- Add support for additional industrial document formats.

---

## 🎯 Project Objective

The objective of this project is to develop an **AI-powered industrial assistant that can answer technical questions using a controlled industrial document knowledge base**.

Instead of relying only on the language model's pretrained knowledge, the system follows a RAG approach:

```text
Retrieve relevant industrial information
                ↓
Provide that information as context
                ↓
Generate a grounded response
```

This architecture is intended to make industrial question answering more **document-specific, traceable, and controlled**.

---

## 👨‍💻 Project Components

### `build_index.py`
Responsible for:

- Reading industrial PDFs
- Cleaning text
- Creating chunks
- Generating embeddings
- Building the FAISS index
- Saving document metadata

### `rag.py`
Responsible for:

- Loading FAISS
- Loading metadata
- Creating question embeddings
- Retrieving relevant chunks
- Building the grounded prompt
- Running the Qwen LLM
- Returning answers and sources

### `app.py`
Responsible for:

- Streamlit application
- Chat interface
- Session state
- Displaying answers
- Displaying retrieved sources
- Industrial UI styling

---

## 📜 License

Add your preferred project license here, for example:

```text
MIT License
```

---

## ⭐ Acknowledgements

This project uses open-source models and libraries from the Python, FAISS, Sentence Transformers, PyTorch, Hugging Face, and Streamlit ecosystems.
