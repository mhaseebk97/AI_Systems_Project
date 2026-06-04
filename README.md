# MyGPT Safety Assistant

## Overview
MyGPT Safety Assistant is an offline, AI-powered industrial safety assistant designed to help workers and supervisors access Occupational Health and Safety (OHS) information quickly and accurately. The system combines Retrieval-Augmented Generation (RAG), semantic search, keyword search, and deterministic safety rules to deliver grounded, verifiable answers from technical manuals and regulations.

## Key Features

- Offline deployment for privacy and data sovereignty
- Retrieval-Augmented Generation (RAG)
- Hybrid Search:
  - Vector Search (FAISS)
  - Keyword Search (SQLite FTS5)
- Llama 3 (8B) local inference through Ollama
- Citation-based responses
- Heading-aware document chunking
- Symbolic Knowledge Injection for safety-critical facts
- Deterministic mathematical calculations using SymPy
- Session-based conversational memory
- FastAPI backend with streaming responses

---

## Problem Statement

Industrial safety documentation is often large, technical, and difficult to navigate during time-sensitive situations. Traditional search methods struggle with:

- Semantic overload from lengthy manuals
- Slow information retrieval
- Difficulty verifying compliance requirements
- AI hallucination risks in safety-critical environments

This project addresses these challenges by grounding AI responses in verified documentation and enforcing strict retrieval-based answer generation.

---

## System Architecture

### Components

1. **Embedding Model**
   - `nomic-embed-text`
   - Converts text into vector embeddings for semantic retrieval

2. **Inference Model**
   - `Llama 3 (8B)`
   - Generates final user-facing answers

3. **Hybrid Retrieval Engine**
   - FAISS vector similarity search
   - SQLite FTS5 keyword search
   - Combined ranking strategy for improved accuracy

4. **Symbolic Tools**
   - Safety signage rule base
   - SymPy mathematical engine

### High-Level Flow

```text
User Query
    │
    ▼
Hybrid Search
(FAISS + FTS5)
    │
    ▼
Relevant Document Chunks
    │
    ▼
Prompt Construction
    │
    ▼
Llama 3 via Ollama
    │
    ▼
Grounded Response + Citations
```

---

## Dataset

The assistant is trained on industrial safety documentation including:

- Risk assessment manuals
- Construction site safety guidelines
- Italian OHS regulations (D.Lgs 81/08)
- Safety training and compliance documents

Topics include:

- Noise exposure
- Vibrations
- Ergonomics
- PPE requirements
- Construction hazards
- Emergency procedures
- Employer and supervisor responsibilities

---

## Preprocessing Pipeline

### Normalization

- Whitespace cleanup
- Lowercasing
- Character sanitization

### Heading-Aware Chunking

Document sections are segmented while preserving heading context.

Example:

```text
[PROTECTIVE EYEWEAR]
Wear protective eyewear at all times.
```

Benefits:

- Improved retrieval accuracy
- Better contextual understanding
- More reliable citations

### Tag Extraction

Important safety concepts are automatically identified and indexed for retrieval.

---

## Symbolic Knowledge Injection

The system injects deterministic rules into the AI workflow to reduce hallucinations.

### Safety Signage Standards

Examples:

| Color | Meaning |
|---------|---------|
| Red | Prohibition |
| Blue | Mandatory Action |
| Yellow | Warning |
| Green | Emergency / Rescue |

### Mathematical Reasoning

Calculations are delegated to SymPy rather than relying on LLM-generated arithmetic.

---

## Technology Stack

### Backend

- Python 3.9+
- FastAPI
- SQLite
- FAISS
- HTTPX
- SymPy

### AI Infrastructure

- Ollama
- Llama 3 (8B)
- nomic-embed-text

### Frontend

- HTML
- CSS
- Vanilla JavaScript
- Server-Sent Events (SSE)

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns model and service status.

### Chat

```http
POST /chat
```

Request:

```json
{
  "session": "default",
  "message": "What PPE is required for excavation work?"
}
```

### RAG Debug

```http
POST /rag_debug
```

Returns retrieved document snippets for debugging and evaluation.

---

## Project Structure

```text
app/
├── api/
│   └── server.py
├── core/
│   ├── rag.py
│   ├── llm_engine.py
│   ├── tools.py
│   └── memory.py
├── web/
│   └── index.html
├── data/
│   ├── db/
│   └── indexes/
└── docs/
```

---

## Evaluation

The system was evaluated using:

### Retrieval Accuracy
Measures whether the correct document sections are retrieved.

### Faithfulness
Ensures responses remain grounded in source documentation.

### Technical Reliability
Includes testing and validation of FAISS, SQLite, and retrieval pipelines.

---

## Future Enhancements

### Multimodal RAG
Analyze safety signs and equipment from images.

### Voice Interface
Hands-free interaction for workers in the field.

### Large-Scale Knowledge Bases
Support thousands of manuals without performance degradation.

---

## Safety Focus

Unlike general-purpose chatbots, MyGPT Safety Assistant prioritizes:

- Evidence-based responses
- Citation-backed answers
- Reduced hallucinations
- Regulatory compliance
- Worker safety

---

## Authors

- Muhammad Haseeb Khalid
- Siraj Ahmed

## Supervisor

Professor Roberto Pietrantuono

---

## License

This project is intended for academic and research purposes. Add an appropriate open-source license before public release.
