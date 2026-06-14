
import requests
from bs4 import BeautifulSoup

def paper_url_to_text(paper_url: str) -> str:
    response = requests.get(paper_url, timeout=30)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "nav", "footer", "figure", "table"]):
        tag.decompose()

    blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            continue
        if tag.name in {"h1", "h2", "h3"}:
            blocks.append(f"## {text}")
        else:
            blocks.append(text)

    return "\n\n".join(blocks)


if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    res = paper_url_to_text(config["paper_url"])
    print(res)