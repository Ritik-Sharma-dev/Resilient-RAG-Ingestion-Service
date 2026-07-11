import requests
import streamlit as st

st.set_page_config(page_title="Resilient RAG Ingestion Service", page_icon="📄", layout="wide")

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")

    try:
        requests.get(f"{api_base_url}/openapi.json", timeout=2)
        st.success("Backend reachable")
    except requests.exceptions.RequestException:
        st.error("Backend not reachable — is uvicorn running?")

st.title("Resilient RAG Ingestion Service")
st.caption("Upload a PDF to ingest it. Per-step timing (extract/chunk/embed/upsert) is traced in Langfuse.")

st.subheader("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if st.button("Ingest", type="primary", disabled=uploaded_file is None):
    with st.spinner("Extracting, chunking, embedding, and upserting..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{api_base_url}/ingest", files=files, timeout=180)
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            if response.status_code == 200:
                body = response.json()
                if body["status"] == "already_ingested":
                    st.info(f"Already ingested as `{body['doc_id']}` ({body['chunk_count']} chunks) — skipped re-embedding.")
                else:
                    st.success(f"Ingested as `{body['doc_id']}` — {body['chunk_count']} chunks.")
            else:
                detail = (
                    response.json().get("detail", response.text)
                    if response.headers.get("content-type", "").startswith("application/json")
                    else response.text
                )
                st.error(f"Ingest failed ({response.status_code}): {detail}")

st.divider()

with st.expander("Documents already ingested", expanded=True):
    if st.button("Refresh"):
        st.rerun()

    try:
        docs_response = requests.get(f"{api_base_url}/documents", timeout=10)
        if docs_response.status_code == 200:
            documents = docs_response.json()
        else:
            st.error(f"Could not load document list ({docs_response.status_code}): {docs_response.text}")
            documents = []
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not load document list: {exc}")
        documents = []

    if documents:
        st.dataframe(
            [
                {
                    "Filename": d["filename"],
                    "Doc ID": d["doc_id"],
                    "Chunks": d["chunk_count"],
                    "Ingested At (UTC)": d["ingested_at"],
                }
                for d in documents
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No documents ingested yet.")
