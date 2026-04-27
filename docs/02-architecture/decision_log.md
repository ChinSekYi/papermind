# Design log
To track major choices, alternatives considered, and why I picked what I picked.
# Overview

## Components
- Parser
- LLM processor
- Output formatter

## Data Flow
PDF -> Text -> Chunk -> LLM extract information -> LLM generates insights -> JSON formatting

## Key Decisions

#### Chunking
- What chunking strategy to use?
    - Read: https://medium.com/@anuragmishra_27746/five-levels-of-chunking-strategies-in-rag-notes-from-gregs-video-7b735895694d#b123
        - Examples: Fixed sized chunking, Recursive chunking, Document Based Chunking, Semantic Chunking, Agentic Chunking
    - Research papers contain structured, hierarchical sections. So, we need to retain the relation between the chunks.
    - Within section context window. Content in each section might be large. Split by paragraphs
    - Chunk-level information extraction - For each chunk, pull structured fields like objectives, methods, key insights, references.


- How to ensure each chunk is related to each other? Since each section have a subsection. 
- Doing chunking by sections vs use fixed size chunking + LLM to find structured JSON.





- What prompting strategy to use?
- Why JSON output? 



## Current limitations
- Parser can only parse text and not images. Tables are parsed into a weird format.

## Inspiration (existing tools)
- https://docanalyzer.ai/ - limitations: does not parse image
