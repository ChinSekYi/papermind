"""Quick end-to-end demo on a small local paper (fast, no scraping)."""
import json

import yaml

from .section_extractor import section_extractor
from .model import llm_process

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    text = open(config["demo_input_path"]).read()
    chunks = section_extractor(text)
    final_output = llm_process(chunks)

    with open(config["demo_output_path"], "w") as f:
        json.dump(final_output.model_dump(), f, indent=2)

    print(final_output)
