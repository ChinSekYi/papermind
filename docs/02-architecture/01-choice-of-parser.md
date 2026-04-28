# ADR 01: Choice of PDF Parsing Engine
- Parsing strategies:
    - marker-pdf - very slow parsing time.
    - PyMuPDF4LLM - much faster than marker-pdf. Preserves tables and formulas.
    - Docling - 
    - MinerU

## Context
Academic PDFs from ArXiv contain multi-column layouts, complex tables, and LaTeX formulas. We need a parser that preserves structural hierarchy for RAG accuracy.


##########EDIT BELOW.BELOW IS A SAMPLE ########
## Options Considered
1. **PyMuPDF (fitz):** Fast, but loses table structure (converts tables to plain text).
2. **Docling (IBM):** Slower, but uses layout-analysis to preserve tables as Markdown.
3. **Marker:** High fidelity, but heavy resource requirements.

## Evaluation Results (from MLflow)
| Parser | Avg Latency | Table Integrity | Formula Rendering |
| :--- | :--- | :--- | :--- |
| PyMuPDF | ~0.5s | Poor | Scrambled |
| Docling | ~8.0s | Excellent | Near-perfect |

## Decision
We will use **Docling** as the primary ingestion engine.

## Rationale
While Docling is significantly slower than PyMuPDF, the downstream "cost" of hallucination due to broken tables is too high. In an AI Engineer role, **data quality beats speed** at the ingestion stage.

## Consequences
- Need to implement an asynchronous queue (Redis/Celery) for multi-file uploads.
- Vector search will need to handle larger chunks due to detailed Markdown tables.


-----

