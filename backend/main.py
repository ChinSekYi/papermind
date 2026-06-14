import yaml
from fastapi import FastAPI
from src.data.section_extractor import section_extractor
from src.data.model import llm_process

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Papermind API")


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/get_paper_info")
def get_paper_info_endpoint():
    with open(config["demo_input_path"]) as f:
        text = f.read()
    output = section_extractor(text)
    return llm_process(output)

