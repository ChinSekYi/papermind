"""
input: markdown text (from paper_url_to_text)
output: list[Chunk]
"""
import re
from .chunking import Chunk
from .input_ingestion import paper_url_to_text


def _merge_paragraphs(paragraphs: list[str], max_chunk_size: int) -> list[str]:
    merged = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_chunk_size:
            current = f"{current}\n\n{para}"
        else:
            merged.append(current)
            current = para
    if current:
        merged.append(current)
    return merged


def section_extractor(text: str, max_chunk_size: int = 2000) -> list[Chunk]:
    # TODO: clean section name, remove irrelevant details, ensure system reliability
    # limitations: doesnt differentiate between ## and ### sections.

    output = []

    # Split on markdown headings, keeping the heading text via capture group
    parts = re.split(r'^##\s*(.+)$', text, flags=re.MULTILINE)

    # parts[0] is any text before the first heading (discarded)
    # remaining parts come in (title, content) pairs
    for title, content in zip(parts[1::2], parts[2::2]):
        title = title.strip()
        paragraphs = [c.strip() for c in re.split(r'\n\s*\n', content.strip()) if c.strip()]
        chunks = _merge_paragraphs(paragraphs, max_chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_data = {"section_title": title, "chunk_id": i, "content": chunk}
            output.append(Chunk(**chunk_data))

    return output


if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    text = paper_url_to_text(config["paper_url"])
    output = section_extractor(text)
    print(output)
