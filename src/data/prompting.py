from .chunking import Chunk

def build_prompt(chunk: Chunk) -> str:
    return f"""You are a research paper analyser for ML Engineers.
        Extract only what is present in this chunk.

        Return valid JSON only, with this shape:
        {{
            "chunk_id": {chunk.chunk_id},
            "section_title": "{chunk.section_title}",
            "chunk_type": "abstract|introduction|method|experiment|result|limitation|conclusion|reference|other",
            "chunk_summary": "string or null",
            "key_claims": ["string"],
            "evidence_quote": "exact short quote or null",
            "entities": {{
                "methods": ["name"],
                "datasets": ["name"],
                "metrics": ["name"],
                "models": ["name"]
            }},
            "experiment_info": {{
                "dataset": "string or null",
                "task": "string or null",
                "metric": "string or null",
                "baseline": "string or null",
                "result": "string or null",
                "improvement": "string or null"
            }},
            "limitation": "string or null",
            "confidence": 0.0
        }}

        Rules:
        - Use null for missing fields.
        - Keep values complete.
        - Do not invent information.
        - Use an exact quote only when the chunk clearly states it.
        - confidence: a float 0-1 indicating how directly chunk_summary and
          key_claims are stated in the text vs. inferred. 1.0 = explicitly
          stated, lower = paraphrased or uncertain.

        TEXT:
        {chunk.content}
    """

def chain_prompts(prev_text, new_text) -> str:
    if not prev_text:
        return f""" ===CHUNK_START=== \n\n
        {new_text} \n\n
         ===CHUNK_END===\n\n """
    else:
        return f"""{prev_text} \n\n
        ===CHUNK_START=== \n\n
        {new_text}, \n\n
        ===CHUNK_END===\n\n """


def build_final_prompt(summaries) -> str:
    return f"""
        You are combining chunk-level JSON into one final paper JSON.

        Return JSON only.

        Final JSON shape:
        {{
            "title": "string or null",
            "authors": ["string"],
            "year": "string or null",
            "problem": "string or null",
            "method": "string or null",
            "dataset": ["string"],
            "metrics": ["string"],
            "key_contributions": ["string"],
            "prerequisites": ["string"],
            "limitations": ["string"],
            "when_to_use": ["string"],
            "when_not_to_use": ["string"],
            "final_summary": ["string"],
            "confidence": 0.0
        }}

        Rules:
        - Use only the chunk JSON below.
        - Do not invent missing details.
        - Remove duplicates.
        - Keep values complete.
        - If something is unclear, use null or an empty list.
        - confidence: a float 0-1, the average of the confidence values across
          the chunk JSONs below, indicating overall confidence in this summary.
        - prerequisites: list background concepts, techniques, or prior papers a
          reader would need to understand before this paper, based only on what
          is referenced or assumed in the chunk JSON.

        Chunk JSON:
        {summaries}
        """