"""Streamlit interface for the CrediTrust complaint analyst."""

from pathlib import Path

import streamlit as st

from src.rag import ComplaintRetriever, answer_question

st.set_page_config(page_title="CrediTrust Complaint Analyst", page_icon="💬", layout="wide")
st.title("CrediTrust Complaint Analyst")
st.caption("Evidence-backed insights from customer complaint narratives")


@st.cache_resource
def load_retriever() -> ComplaintRetriever:
    return ComplaintRetriever(Path("vector_store"))


with st.sidebar:
    st.header("Search settings")
    product_label = st.selectbox(
        "Product",
        ["All products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"],
    )
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=5)
    st.info("Answers are grounded in retrieved excerpts. Open each source to verify the evidence.")

if "history" not in st.session_state:
    st.session_state.history = []

try:
    retriever = load_retriever()
except (FileNotFoundError, ValueError) as error:
    st.error("Vector store not found or invalid. Run `python -m src.index_complaints` first.")
    st.code(str(error))
    st.stop()

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        with st.expander(f"View {len(item['sources'])} sources"):
            for number, source in enumerate(item["sources"], 1):
                st.markdown(f"**Source {number} — {source.citation}** · similarity `{source.score:.3f}`")
                st.write(source.text)

question = st.chat_input("Ask about complaint themes, causes, or product differences")
if question:
    product = None if product_label == "All products" else product_label
    with st.spinner("Searching complaints and synthesizing evidence..."):
        answer, sources = answer_question(question, retriever, top_k=top_k, product=product)
    st.session_state.history.append({"question": question, "answer": answer, "sources": sources})
    st.rerun()

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.history = []
    st.rerun()

