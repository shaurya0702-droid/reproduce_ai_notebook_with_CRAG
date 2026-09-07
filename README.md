# ReproduceAI — v0 (Prototype Archive)

> **Status: 🗄️ Archived / Frozen Prototype Baseline**
> This repository holds the original proof-of-concept notebooks for **ReproduceAI**, built before the project was refactored into a modular Python application (`core.py`, `features.py`, `prompts.py`, `main.py`, `app.py`). It is kept here for historical reference and learning purposes only — **this is not the active codebase.** For the current, working version of ReproduceAI, see the production repository at the project root.

**ReproduceAI** is a research paper assistant that uses Retrieval-Augmented Generation (RAG) to answer questions about a paper, summarize it, and extract structured reproduction metadata — with a Corrective RAG (CRAG) layer added on top to catch and fix bad retrievals.

---

## Purpose

This notebook set was used to prototype the core RAG pipeline and validate each feature end-to-end in a single linear script, before splitting the logic into reusable modules. It answers one question — **"Does this pipeline actually work?"** — before any concern for structure, reusability, or deployment.

---

## Repository Layout

```text
reproduce_ai_version0/
├── Reproduce_AI/
│   ├── version_0_notebook/       # Earlier archived notebook revisions
│   ├── ReproduceAI_V0.ipynb      # Core prototype: PDF ingestion -> RAG -> Structured Report
│   ├── rag.ipynb                 # Baseline RAG + Corrective RAG (CRAG) benchmark
│   └── paper.pdf                 # Sample input paper: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
├── .gitignore
└── README.md
```

---

## Notebook 1 — Core Pipeline (`ReproduceAI_V0.ipynb`)

### What This Notebook Does

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

### Pipeline

```text
paper.pdf
   │
   ▼
PyMuPDFLoader (ingestion)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
   │
   ▼
sentence-transformers/all-MiniLM-L6-v2 (embeddings)
   │
   ▼
ChromaDB  (./chroma_db)
   │
   ▼
ChatGroq  (openai/gpt-oss-safeguard-20b)
   │
   ▼
Paper Overview → Structured Metadata (Pydantic) → Summary → Implementation Plan
   → Risk Analysis → Executive Summary → research_report{}
```

### Stack

| Component | Choice |
|---|---|
| PDF Loader | `PyMuPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (size=500, overlap=50) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB, persisted at `./chroma_db` |
| LLM | `ChatGroq` — `openai/gpt-oss-safeguard-20b` |
| Structured extraction | Pydantic model (`PaperMetadata`) |

### How to Run

1. Place a research paper PDF in the same directory as this notebook, named `paper.pdf` (the repo ships with `paper.pdf` — the CRAG paper — as a working sample).
2. Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```
3. Install dependencies (see root `requirements.txt`, or the install block below).
4. Run all cells top to bottom in order — later cells depend on variables defined earlier (`docs`, `retriever`, `metadata`, etc.), so cells **cannot** be run out of order.

### Known Limitations of This Version

These are exactly the issues that motivated the refactor into the current architecture, listed here intentionally so the evolution of the project is visible:

- **Everything lives in one script.** No separation between infrastructure (loading, chunking, embeddings), prompts, and feature logic — hard to test or reuse individual pieces.
- **Persistent Chroma storage.** Uses `persist_directory="./chroma_db"` with the default collection, meaning running the notebook again on a different paper **appends** to the same vector store instead of starting fresh. The production version fixes this with a unique collection per run.
- **14+ sequential LLM calls per paper.** Metadata is extracted one field at a time (title, dataset, architecture, ...), each a separate retrieval + LLM call. This is slow and burns through API rate limits quickly. The production version consolidates this.
- **No error handling.** A failed API call, rate limit, or malformed PDF will simply crash the notebook.
- **No UI.** Output is only visible via `print()` statements.
- **Hardcoded values.** Model names, chunk sizes, and file paths are inlined directly rather than centralized in a config.

### Relationship to the Current Codebase

| Notebook Section | Became |
|---|---|
| PDF loading, splitting, embeddings, vectorstore, retriever, LLM setup | `core.py` |
| All `PromptTemplate` definitions + `FIELDS` dict | `prompts.py` |
| `PaperMetadata`, overview/metadata/summary/planner/risk/executive-summary functions | `features.py` |
| Top-to-bottom script execution | `main.py` (`run_pipeline()`) |
| `print()` statements | `app.py` (Streamlit UI) |

---

## Notebook 2 — Baseline RAG + Corrective RAG (`rag.ipynb`)

This notebook builds a second, independent RAG chain over the same `paper.pdf` and then layers **Corrective RAG (CRAG)** on top of it — an evaluator that grades retrieved chunks and reroutes to web search when retrieval quality is poor.

### Setup

```bash
pip install -q langchain-community langchain-text-splitters langchain-huggingface \
                langchain-chroma langchain-groq tavily-python pymupdf python-dotenv
```

### Ingestion (actual run stats)

```python
loader = PyMuPDFLoader("paper.pdf")
docs = loader.load()
```

On the bundled sample paper ("Corrective Retrieval Augmented Generation," Yan, Gu, Zhu & Ling), this notebook run produced:

- **16 pages** loaded
- **144 chunks** after `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
- Retriever: `vectorstore.as_retriever(search_kwargs={"k": 4})` — top-4 chunks per query

### Baseline RAG Chain

```python
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-safeguard-20b",
    temperature=0
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

**Example run** — `"How does CRAG correct an incorrect retrieval result?"` retrieved 4 chunks (evaluator/architecture discussion, Self-CRAG/Self-RAG comparison tables, etc.) and produced a baseline answer describing a T5-based "correction module" that rewrites low-confidence retrievals — reasonably close to the paper's actual mechanism, since the relevant explanatory text happened to be well-retrieved for this particular question.

### The CRAG Layer

A structured evaluator grades retrieved documents before they're used to answer:

```python
class CRAGEvaluation(BaseModel):
    action: str    # "CORRECT", "AMBIGUOUS", or "INCORRECT"
    score: float   # retrieval quality, 0.0-1.0
    reason: str    # short explanation

evaluator = llm.with_structured_output(CRAGEvaluation)
```

Two supporting functions:

- **`evaluate_retrieval(question)`** — retrieves docs, runs the evaluator over them *jointly* against the question, returns `(docs, CRAGEvaluation)`
- **`refine_documents(docs, question)`** — re-evaluates each document *individually* and keeps only those scoring `>= 0.5`

### Routing Logic (`crag_answer`)

```text
                 ┌───────────────┐
   question ────▶│  Retriever    │  (k=4, Chroma)
                 └───────┬───────┘
                         ▼
                 ┌───────────────┐
                 │ CRAGEvaluation│  action + score + reason
                 └───────┬───────┘
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
       CORRECT       AMBIGUOUS      INCORRECT
           │             │             │
   refine_documents  refine_documents  Tavily search
   (score ≥ 0.5)     (score ≥ 0.5)     (max_results=5)
           │        + Tavily search        │
           │          (max_results=3)      │
           │             │                 │
           └─────────────┴─────────────────┘
                         ▼
              final answer generation (same prompt/llm)
```

| Grade | Action |
|---|---|
| `CORRECT` | Refine internal documents (drop low-score chunks), use as context |
| `INCORRECT` | Discard internal documents entirely; answer purely from **Tavily** web search |
| `AMBIGUOUS` | Blend refined internal documents with Tavily web search results |

**Example run** — same question as the baseline, `"How does CRAG correct an incorrect retrieval result?"`, through `crag_answer()`:

```
ACTION: AMBIGUOUS
SCORE:  0.6
REASON: The retrieved documents contain partial information about CRAG's
        correction mechanism—specifically that a retrieval evaluator
        estimates confidence and triggers actions such as Correct,
        Incorrect, or Ambiguous. However, they lack a detailed description
        of how the system actually corrects an irrelevant retrieval.
```

With `AMBIGUOUS` routing, the notebook blended the refined internal chunks with 3 Tavily web results, producing a more complete three-step answer (detect → discard faulty documents → trigger corrective web search → replace/augment context) than the baseline chain alone gave for the same question.

### Benchmark Experiment — Baseline RAG vs. CRAG

This is the core stress-test in the notebook: a question specific and technical enough that the internal paper chunks alone are insufficient, to see whether CRAG actually catches a bad retrieval and corrects it.

**Test question:**
> "In the 2024 CRAG paper, what specific learning rate, optimizer, and base model were used to train the lightweight retrieval evaluator?"

**Ground truth (given to the grader):**
> The retrieval evaluator is initialized from T5-large (Raffel et al., 2020) and fine-tuned with AdamW using a cosine learning rate schedule starting at 2e-5.

A separate strict 1–10 LLM grader was used, explicitly penalizing base-model hallucinations:

| Score band | Meaning |
|---|---|
| 8–9 | Accurate on core facts (T5-large + ~2e-5 schedule) |
| 4–6 | Mostly correct, misses optimizer or exact schedule |
| 2–3 | Hallucinates the wrong base model (e.g. MiniLM, BERT) |
| 1 | Completely wrong, irrelevant, or fabricated |

**Actual results from this run:**

| Approach | Evaluator Decision | Answer (base model / optimizer / LR) | Score |
|---|---|---|---|
| Normal RAG | — (no grading step) | Guessed **BERT-base-uncased**, AdamW, LR **1e-5** — wrong base model | **1 / 10** |
| CRAG | `INCORRECT` (confidence 0.1) → Tavily fallback | Correctly identified **T5-large**, AdamW, cosine schedule starting at **2e-5** | **9 / 10** |

```
Evaluator Decision: INCORRECT (Confidence: 0.1)
Normal RAG Score  : 1/10
C-RAG Score       : 9/10
Net Improvement   : +8
```

This is the clearest demonstration in the notebook of why the CRAG layer matters: on a question the internal paper chunks genuinely can't answer (the evaluator's own training details aren't well-covered in the retrieved pages), plain RAG doesn't know it's wrong and hallucinates a plausible-sounding but incorrect base model. CRAG's evaluator correctly flags the retrieval as `INCORRECT` with very low confidence (0.1), discards it, and pulls the correct facts from Tavily web search instead.

---

## Known Limitations (Notebook 2)

- Same monolithic, single-script structure as Notebook 1 — no separation between evaluator, router, and generation logic.
- `refine_documents` makes one evaluator call **per retrieved chunk**, on top of the joint evaluation call — for `k=4` this is up to 5 sequential LLM calls per query before an answer is even generated.
- Shares the same `./chroma_db` persistence issue as Notebook 1 (no per-paper isolation).
- No caching — the same question re-run will re-issue the full retrieval + evaluation + (possibly) web search chain from scratch.

---

## Migration & Lineage Matrix

| Prototype Component | Production Target |
|---|---|
| PDF loading, splitting, embeddings, vectorstore, retriever, LLM setup | `core.py` |
| All `PromptTemplate` definitions + `FIELDS` dict | `prompts.py` |
| `PaperMetadata`, overview/metadata/summary/planner/risk/executive-summary functions | `features.py` |
| CRAG evaluator (`CRAGEvaluation`), `refine_documents`, Tavily routing | `retrieval/crag_router.py` |
| Top-to-bottom script execution | `main.py` (`run_pipeline()`) |
| `print()` statements | `app.py` (Streamlit UI) |

---

## Setup & Execution

### Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Installation

```bash
# Clone the repo
git clone https://github.com/shaurya0702-droid/reproduce_ai_version0.git
cd reproduce_ai_version0/Reproduce_AI

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -q langchain-community langchain-text-splitters langchain-huggingface \
                langchain-chroma langchain-groq tavily-python pymupdf python-dotenv \
                jupyter pydantic

# Set environment variables (see above)
jupyter notebook ReproduceAI_V0.ipynb    # core prototype
jupyter notebook rag.ipynb               # baseline RAG + CRAG benchmark
```

---

## Should You Edit These Notebooks?

No. Both notebooks are considered complete and frozen. All new feature work happens in the modular codebase at the project root. These files stay untouched as a record of where the project started.

## Where to Go Instead

For the current, actively developed version of ReproduceAI — modular codebase, isolated per-paper Chroma collections, batched LLM calls, and a Streamlit chat interface — see the production repository at the project root.
