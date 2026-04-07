from pydantic import BaseModel

class Chunk(BaseModel):
    section_title: str
    chunk_id: int
    content: str