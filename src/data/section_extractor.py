"""
input: .md file
output: JSON file
    [{"section": "method", 
    "raw_title": "## Proposed Methods",
    "chunks": [..,..,,,]],
    ,..
    }]
"""
import json
import re
from .chunking import Chunk

def section_extractor(filepath) -> json:
    # TODO: clean section name, remove irrelevant details, ensure system reliability
    # limitations: doesnt differentiate between ## and ### sections. 

    output = []

    with open(filepath, 'r') as f:
        content = f.read()

    sections = re.split(r'^#{2,}', content, flags=re.MULTILINE)

    # for each section, save the title and content. 
    for section in sections:
        title_content = re.split(r'\*\*(.*?)\*\*', section.strip(), maxsplit=1)
        title_content = title_content[1:] if len(title_content) > 2 else title_content

        # Happy path
        # TODO: what if section dont have matching pattern?
        if len(title_content) == 2: 
            content = title_content[1]
            chunks = re.split(r'\n\s*\n', content.strip())
            for i, chunk in enumerate(chunks):
                chunk_data = {"section_title": title_content[0], "chunk_id":i, "content":chunk}
                output.append(Chunk(**chunk_data))
    return output
        
import yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    output = section_extractor(config["filepath"])
    print(output)