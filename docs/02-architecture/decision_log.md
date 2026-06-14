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

#### Chunk size merging (feat/phase2)
- Problem found: `section_extractor` originally made one chunk per paragraph (split on blank lines). On a real paper this produced 2224 chunks — way too many LLM calls in `llm_process` (one per chunk), too slow/costly.
- Fix: added `_merge_paragraphs`, which greedily concatenates consecutive paragraphs within a section up to a `max_chunk_size` char budget (default 2000 chars), starting a new chunk when the budget would be exceeded.
- Result: same paper now produces 184 chunks instead of 2224.
- Mistakes/notes:
    - Chose a simple char-length budget instead of token counting — good enough for now, but token-based budgeting would be more accurate for LLM context limits.
    - Still paragraph-granularity within a chunk — a single huge paragraph longer than `max_chunk_size` is not split further (acceptable edge case for now, but worth a TODO).
    - Still no `###` subsection awareness — merging happens within `##` sections only, so a section with many subsections gets all subsections' paragraphs merged together without subsection boundaries preserved.



## Current limitations
- Parser can only parse text and not images. Tables are parsed into a weird format.

## Inspiration (existing tools)
- https://docanalyzer.ai/ - limitations: does not parse image
