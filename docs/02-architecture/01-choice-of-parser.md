# ADR 01: Choice of PDF Parsing Engine

## Status
Decided

## Context
Academic PDFs from ArXiv contain multi-column layouts, complex tables, and LaTeX formulas. Papermind's goal is to help junior ML engineers understand difficult research papers — so the parser must preserve structural hierarchy (sections, equations, tables, figures) accurately enough for a downstream RAG pipeline to produce reliable explanations.

We benchmarked three parsers on two representative papers:
- **P1:** *Relaxation-Informed Training of Neural Network Surrogate Models* (35 pages, math-heavy — equations, proofs, optimization formulas)
- **P2:** *Long-Tail Internet Photo Reconstruction* (16 pages, figure/table-heavy — results tables, architecture diagrams)

## Options Considered

### 1. pymupdf4llm
Fast text extraction built on PyMuPDF.

### 2. Docling (IBM)
Layout-aware parser using document understanding models.

### 3. MinerU
VLM/OCR-based extraction using a hybrid local pipeline (requires `torch`).

## Evaluation Results

### Equations (P1 — math-heavy)
| Parser | Equation Handling |
| :--- | :--- |
| MinerU | ✅ Full LaTeX in `$$...$$` blocks — fully readable and renderable |
| Docling | ❌ `<!-- formula-not-decoded -->` — detected but could not render |
| pymupdf4llm | ❌ `==> picture intentionally omitted <==` — treats equations as images, drops them entirely |

### Tables (P2 — results-heavy)
| Parser | Table Handling |
| :--- | :--- |
| MinerU | ✅ Clean markdown tables, properly structured |
| Docling | ✅ Best formatting — cleanest column alignment, correct headers |
| pymupdf4llm | ❌ Merges entire table content into single cells, duplicates content, unreadable |

### Images / Figures
| Parser | Image Handling |
| :--- | :--- |
| MinerU | ✅ Extracts and references images as local files `![](images/hash.jpg)` |
| Docling | ❌ No image extraction |
| pymupdf4llm | ❌ Notes figure dimensions but omits content |

### Speed (on Apple M3 Pro, CPU/MPS)
| Parser | Speed |
| :--- | :--- |
| pymupdf4llm | Seconds |
| Docling | Minutes |
| MinerU | Hours (~4hrs for a 35-page paper) |

### Weighted Rubric Scores (rough, 0-10 per criterion)

| Criteria (weight) | MinerU | Docling | pymupdf4llm |
| :--- | :---: | :---: | :---: |
| Structure Preservation (40%) | 9 | 7 | 2 |
| Content Accuracy (30%) | 9 | 6 | 3 |
| Downstream Utility (20%) | 8 | 8 | 3 |
| Latency / Cost (10%) | 1 | 7 | 10 |
| **Weighted Total** | **8.0** | **6.9** | **3.3** |

Note: by the weighted rubric, MinerU scores highest overall (8.0 vs 6.9) — its
quality advantage outweighs the 10% latency penalty in the formula. However,
~4hrs/paper is a hard practical blocker for an interactive MVP that a 10% weight
doesn't capture. **Docling was chosen as a deliberate override of the raw weighted
score**, because latency is a binary go/no-go constraint for this use case, not a
smoothly tradeable one. This is a good talking point: the rubric informs the
decision but doesn't mechanically determine it — judgment about hard constraints
matters too.

### Summary Scorecard
| Criteria | MinerU | Docling | pymupdf4llm |
| :--- | :---: | :---: | :---: |
| Equations | ✅ Best | ❌ Fails | ❌ Drops |
| Tables | ✅ Good | ✅ Best | ❌ Broken |
| Images | ✅ Only one | ❌ | ❌ |
| Speed | ❌ Hours | ⚠️ Minutes | ✅ Seconds |

## Decision
**Docling** is selected as the primary parser for the MVP pipeline, with MinerU reserved for offline/async pre-processing where quality is critical.

## Rationale
MinerU produces the highest quality output — it is the only parser that correctly renders LaTeX equations and extracts images. However, its processing speed (~400s/page on local hardware) makes it impractical for any real-time or interactive use case.

Docling offers the best speed/quality tradeoff:
- Tables are cleanly extracted and readable
- Text structure and section hierarchy are well preserved
- Processing time is in the order of minutes, not hours
- The equation limitation is a known gap, but for MVP purposes, inline math notation (e.g. `ℓ1`, `ℓ2`) is still partially preserved as text

pymupdf4llm is too lossy for ML papers — broken tables and dropped equations would directly harm explanation quality downstream.

## Consequences
- Docling will be the default parser in the ingestion pipeline
- Equations will be partially lost for heavily math-heavy papers — this is an acceptable MVP tradeoff, to be revisited in a future phase
- MinerU can be used as an optional high-quality mode (e.g. async background job) for users who need full equation fidelity
- Need to account for Docling's multi-minute latency in the API design (async processing recommended)
