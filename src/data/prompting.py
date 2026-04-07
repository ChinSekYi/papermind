from .chunking import Chunk

def build_prompt(chunk: Chunk) -> str:
    return f"""You are a research paper analyser for ML Engineers.
    
    Summarise the following sections clearly.

    TEXT: {chunk.content}

    Return: 
    - Key insights
    - Methodology
    - Objectives
    - References
    """

def build_final_prompt(summaries):
    return f"""
Combine these summaries into one coherent research summary:

{summaries}
"""