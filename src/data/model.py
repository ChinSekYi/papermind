from .prompting import build_prompt, build_final_prompt, chain_prompts
from .section_extractor import section_extractor

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class FinalOutput(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    problem: str | None = None
    method: str | None = None
    dataset: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    key_contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    when_to_use: list[str] = Field(default_factory=list)
    when_not_to_use: list[str] = Field(default_factory=list)
    final_summary: list[str] = Field(default_factory=list)
    confidence: float = 0.0


llm = ChatOllama(
        model="gemma4",
        temperature=0.1
    )

final_output_llm = llm.with_structured_output(FinalOutput)

def llm_process(chunks):
    summaries = ""
    prev_text = ""
    for i, chunk in enumerate(chunks):
        print(f"**** chunk cur anlaysed: {chunk}")
        res = llm.invoke(build_prompt(chunk))
        print(f"        {res}")
        print("\n")
        prev_text = chain_prompts(prev_text, res.content)

    print("==" * 100)
    print(summaries)
    return final_output_llm.invoke(build_final_prompt(prev_text))

import yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    output = section_extractor(config["filepath"])
    final_output = llm_process(output)
    print(final_output)