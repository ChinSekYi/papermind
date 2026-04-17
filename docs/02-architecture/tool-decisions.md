# Details to consider

- Which python version to use?
    - Avoid latest and old releases
    - Read the change log on official python website for the python version you are planing to use. (https://docs.python.org/release/3.8.12/whatsnew/changelog.html#changelog)
    - Check Dependencies. Eg, if you are planning to use Pytorch, check which Python version is compatible with Pytorch. 
- Which virtual env to use?
    - uv - modern Python package and project manager written in Rust.
    To enter uv environment: 
    ```
    uv sync
    source .venv/bin/activate
    uv pip install -r requirements.txt
    uv pip list
    ```
    
