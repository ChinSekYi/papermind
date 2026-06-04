# Evaluation Framework for MVP

## Evaluation Methods

1. **LLM-as-a-judge** — Structured comparison of parser outputs
2. **Automated checks** — Pattern matching for obvious failures (missing headings, broken equations, lost tables)
3. **Manual human evaluation** — Visual inspection on representative paper set
4. **End-to-end testing** — Which parser produces the best final paper summaries downstream

## Evaluation Layer Design

1. **Format and completeness** — Structure preservation
2. **Factual faithfulness** — Content accuracy
3. **Decision usefulness** — Downstream pipeline utility

## Scoring Rubric

### Structural Fidelity (40%)
- Preserves section headings, reading order, paragraphs
- Correctly extracts tables, figures, captions, equations, references
- Handles two-column layouts and mixed formatting

### Factual Faithfulness (30%)
- Extracted text matches source PDF
- No hallucinated or corrupted content

### Downstream Usefulness (20%)
- Chunker can segment correctly
- Prompt chain extracts objectives, methods, insights accurately

### Efficiency (10%)
- Parsing speed
- Manual cleanup burden

## Evaluation Workflow

1. Run 3 parsers (PyMuPDF4LLM, Docling, MinerU) on each paper
2. Score manually using the rubric above
3. Apply pattern checks for structural failures
4. Track results in MLflow with manual scores and artifacts
5. Run end-to-end pipeline test with top performer