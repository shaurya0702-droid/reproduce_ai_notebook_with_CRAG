# ReproduceAI — V0 Notebook (Prototype)

Research Paper Assistant that uses RAG to answer questions, summarize papers, and extract key research insights

This notebook is the original proof-of-concept for ReproduceAI, built before the project was refactored into a modular Python application (`core.py`, `features.py`, `prompts.py`, `main.py`, `app.py`).

It is kept here for historical reference and learning purposes only. **This is not the active codebase.** For the current, working version of ReproduceAI, see the root of the repository.

---

## Purpose

This notebook was used to prototype the core Retrieval-Augmented Generation (RAG) pipeline and validate each feature end-to-end in a single linear script before splitting the logic into reusable modules.

It answers one question: *"Does this pipeline actually work?"* — before any concern for structure, reusability, or deployment.

---

## What This Notebook Does

Run top to bottom, it:

1. Loads a research paper PDF (`paper.pdf`) using `PyMuPDFLoader`
2. Splits it into overlapping chunks with `RecursiveCharacterTextSplitter`
3. Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
4. Stores embeddings in a persistent ChromaDB vector store
5. Builds a retriever and runs a basic question-answering chain
6. Generates a **Paper Overview** (title, authors, conference, year, domain, description)
7. Extracts **Structured Metadata** (dataset, architecture, optimizer, learning rate, batch size, epochs, metrics) field-by-field, validated with a Pydantic model, along with page citations
8. Generates a **Paper Summary** (short summary, key bullet points, main contribution)
9. Generates an **Implementation Plan** from the extracted metadata
10. Generates a **Risk Analysis** (missing info, reproduction difficulty, risks, suggestions)
11. Generates an **Executive Summary** combining all of the above
12. Assembles everything into a single `research_report` dictionary

---

## How to Run

1. Place a research paper PDF in the same directory as this notebook, named `paper.pdf`.
2. Create a `.env` file with:
GROQ_API_KEY=your_groq_api_key
3. Install dependencies (see root `requirements.txt`).
4. Run all cells top to bottom in order — later cells depend on variables defined earlier (`docs`, `retriever`, `metadata`, etc.), so cells cannot be run out of order.

---

## Known Limitations of This Version

These are exactly the issues that motivated the refactor into the current architecture, listed here intentionally so the evolution of the project is visible:

- **Everything lives in one script.** No separation between infrastructure (loading, chunking, embeddings), prompts, and feature logic — hard to test or reuse individual pieces.
- **Persistent Chroma storage.** Uses `persist_directory="./chroma_db"` with the default collection, meaning running the notebook again on a different paper appends to the same vector store instead of starting fresh. The production version fixes this with a unique collection per run.
- **14+ sequential LLM calls per paper.** Metadata is extracted one field at a time (`title`, `dataset`, `architecture`, ...), each a separate retrieval + LLM call. This is slow and burns through API rate limits quickly. The production version consolidates this.
- **No error handling.** A failed API call, rate limit, or malformed PDF will simply crash the notebook.
- **No UI.** Output is only visible via `print()` statements.
- **Hardcoded values.** Model names, chunk sizes, and file paths are inlined directly rather than centralized in a config.

---

## Relationship to the Current Codebase

| Notebook Section | Became |
|---|---|
| PDF loading, splitting, embeddings, vectorstore, retriever, LLM setup | `core.py` |
| All `PromptTemplate` definitions + `FIELDS` dict | `prompts.py` |
| `PaperMetadata`, overview/metadata/summary/planner/risk/executive-summary functions | `features.py` |
| Top-to-bottom script execution | `main.py` (`run_pipeline()`) |
| `print()` statements | `app.py` (Streamlit UI) |

---

## Should You Edit This Notebook?

No. This notebook is considered **complete and frozen**. All new feature work happens in the modular codebase at the project root. This file stays untouched as a record of where the project started.
