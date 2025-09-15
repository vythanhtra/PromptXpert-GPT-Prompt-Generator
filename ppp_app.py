import streamlit as st
from pathlib import Path

st.set_page_config(page_title="PPP Financial Model", layout="wide")

# Read HTML content
html_file = Path("ppp_model.html")
if html_file.exists():
    html_content = html_file.read_text(encoding="utf-8")
    st.components.v1.html(html_content, height=1000, scrolling=True)
else:
    st.error("ppp_model.html not found. Please ensure the file exists in the project root.")
