from fastapi import FastAPI
from src.data.section_extractor import section_extractor
from src.data.model import llm_process


app = FastAPI(title="Papermind API")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/get_paper_info")
def get_paper_info_endpoint():
    output = section_extractor("./data/processed/attention-is-all-you-need.md")
    summaries = llm_process(output)
    return summaries

