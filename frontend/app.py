import os

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# Set main page
st.set_page_config(page_title="Papermind MVP", layout="wide")
st.title("Papermind MVP")
st.write("Demo")

if st.button('run'):
    with st.spinner('Model is running...'):
        response = requests.get(f"{API_URL}/get_paper_info")
    if response.status_code == 200:
        st.success(f"Done")
        st.write(response.json())
    else:
        st.error(f"Error {response.status_code}: {response.text}")