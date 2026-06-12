# Prev Evaluation Framework (Interview Reference)

This is the original planned evaluation design referenced on the resume
("LLM-as-a-judge evaluation framework", "weighted MCDM rubric tracked via MLflow").
Only partially implemented — the actual MVP used manual scoring against this
rubric (see [docs/02-architecture/01-choice-of-parser.md](docs/02-architecture/01-choice-of-parser.md)).
This doc is for defending the design in an interview if asked "how would this work".

## Weighted MCDM Rubric

Criteria and weights (from [docs/04-evaluation/evaluation_framework.md](docs/04-evaluation/evaluation_framework.md)):

- Structural Fidelity — 40%
- Factual Faithfulness — 30%
- Downstream Usefulness — 20%
- Efficiency — 10%

### Alternative naming (industry-standard terms)

| Term used here | Industry equivalent | Meaning |
|---|---|---|
| Structural Fidelity | Layout/Structure Preservation (Document Structure Accuracy) | Does the parser correctly identify and preserve headings, reading order, tables, figures, equations, columns — the document's *shape*, not its content. |
| Factual Faithfulness | Content Accuracy / Extraction Fidelity (OCR: char/word error rate) | Does the extracted text match the source verbatim — no dropped, garbled, duplicated, or hallucinated content. |
| Downstream Usefulness | Task Utility / Pipeline Compatibility | Is the output in a form downstream components (chunkers, embedders, LLM prompts) can consume well — clean markdown, sensible chunk boundaries. |
| Efficiency | Latency / Throughput (Compute Cost) | Speed and resource cost of parsing — wall-clock time, GPU/CPU usage. |

These map to a standard "quality vs. cost" split used in doc-AI benchmarks
(e.g. Unstructured.io, LlamaParse comparisons): **accuracy metrics**
(structure + content) vs. **operational metrics** (utility + cost). For the
interview, "structure preservation, content accuracy, downstream utility,
and latency/cost" reads as more standard phrasing than "fidelity" / "MCDM".

```python
from pydantic import BaseModel

class ParserScore(BaseModel):
    structural_fidelity: float  # 0-10
    factual_faithfulness: float
    downstream_usefulness: float
    efficiency: float

WEIGHTS = {
    "structural_fidelity": 0.4,
    "factual_faithfulness": 0.3,
    "downstream_usefulness": 0.2,
    "efficiency": 0.1,
}

def weighted_score(score: ParserScore) -> float:
    return sum(getattr(score, k) * w for k, w in WEIGHTS.items())
```

Each parser is scored per paper, scores are averaged across the corpus, and
parsers are ranked by total weighted score. This is "MCDM" in the sense that
each criterion is scored independently before being combined — nothing
fancier than a weighted sum, but principled because the criteria are kept
separate until the final combination.

## LLM-as-a-Judge

Structural Fidelity and Efficiency are scored with automated/pattern-based
checks (regex for headings/tables/equations, wall-clock timing) — no LLM
needed there, which also keeps cost down.

Factual Faithfulness and Downstream Usefulness are scored by prompting an
LLM with the source text and parser output side by side:

```python
JUDGE_PROMPT = """You are evaluating a PDF parser's output against the source document.

SOURCE (ground truth excerpt):
{source_text}

PARSED OUTPUT:
{parsed_text}

Score 0-10 on each, with brief justification:
1. Factual Faithfulness: Does the parsed text match the source exactly,
   without hallucinated, dropped, or corrupted content?
2. Downstream Usefulness: Could a chunker segment this cleanly, and could
   a summarization prompt extract objectives/methods/results accurately?

Return JSON: {{"factual_faithfulness": int, "factual_reasoning": str,
"downstream_usefulness": int, "downstream_reasoning": str}}"""
```

The JSON response is parsed into `ParserScore`-compatible fields and fed
into `weighted_score`.

## Interview Framing

"Automated checks for objective criteria (structure, speed), LLM-judge for
subjective/semantic criteria (faithfulness, usefulness), combined via the
weighted rubric." Defensible architecture even though only partially
implemented — and a reasonable answer if asked to sketch or code part of it
live.
