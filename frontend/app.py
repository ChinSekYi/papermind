import requests
import streamlit as st
import pandas as pd
import json 

API_URL = "http://127.0.0.1:8000"

# Set main page
st.set_page_config(page_title="Papermind MVP", layout="wide")
st.title("Papermind MVP")
st.write("Demo")

if "papers" not in st.session_state:
    st.session_state.papers = []
    
def normalize_value(v):
    if isinstance(v, list):
        return "\n".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return v

if st.button('run'):
    with st.spinner('Model is running...'):
        response = requests.get(f"{API_URL}/get_paper_info", timeout=180)
    
    if response.status_code == 200:
        st.success(f"Done")
        data = response.json()
        st.session_state.papers.append({k: normalize_value(v) for k, v in data.items()})
    else:
        st.error(f"Error {response.status_code}: {response.text}")

if st.session_state.papers:
    # Build as papers x fields, then transpose to fields x papers
    df = pd.DataFrame(st.session_state.papers)
    df.index = [f"Paper {i+1}" for i in range(len(df))]
    compare_df = df.T
    st.dataframe(compare_df, use_container_width=True)

