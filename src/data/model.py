from .prompting import build_prompt, build_final_prompt, chain_prompts
from .section_extractor import section_extractor

from pydantic import BaseModel
from langchain_ollama import ChatOllama

class Summary(BaseModel):
    key_idea: str
    methods: str
    results: str


llm = ChatOllama(
        model="gemma4",
        temperature=0.1
    )

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
    final_summary = llm.invoke(build_final_prompt(prev_text))
    #output = Summary()
    return final_summary.content



if __name__ == "__main__":
    output = section_extractor("./data/processed/attention-is-all-you-need.md")
    summaries = llm_process(output)
    print(summaries)