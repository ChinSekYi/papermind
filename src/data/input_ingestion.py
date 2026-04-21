
import requests
from bs4 import BeautifulSoup

arxiv_id = "2604.18580v1"

def paper_url_to_text(paper_url: str) -> str:
    txt_url = f"https://arxiv.org/html/{arxiv_id}"
    response = requests.get(txt_url, timeout=30)

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
    res = paper_url_to_text(arxiv_id)
    print(res)