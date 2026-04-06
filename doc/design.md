# System Design

# Overview

## Components
- Parser
- LLM processor
- Output formatter

## Data Flow
PDF -> Text -> Chunk -> LLM extract information -> LLM generates insights -> JSON formatting

## Key Decisions
- Which parsing strategy to use?
    - marker-pdf - very slow parsing time. >6min for "attention is all you need.pdf"
    - PyMuPDF4LLM - much faster than marker-pdf. Preserves tables and formulas too.
- What chucking strategy to use?
    - Read: https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d#b123
    - Fixed sized chunking, Recursive chunking, Document Based Chunking, Semantic Chunking, Agentic Chunking
- What prompting strategy to use?
- Why JSON output? 


