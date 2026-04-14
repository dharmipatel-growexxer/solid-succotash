"""
Temporary Streamlit dashboard for the Government Schemes RAG chatbot.
"""
from typing import Dict, List

import streamlit as st

from chain import RAGChain


def _init_state() -> None:
    if "chain" not in st.session_state:
        st.session_state.chain = RAGChain()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = []


def _extract_unique_scheme_links(docs: List[Dict], max_links: int = 10) -> List[Dict[str, str]]:
    seen = set()
    links: List[Dict[str, str]] = []

    for doc in docs:
        meta = doc.get("metadata", {})
        name = meta.get("scheme_name", "Unknown Scheme")
        url = (meta.get("scheme_url") or "").strip()
        if not url:
            continue
        key = (name, url)
        if key in seen:
            continue
        seen.add(key)
        links.append({"scheme_name": name, "scheme_url": url})
        if len(links) >= max_links:
            break

    return links


def _render_sidebar() -> None:
    st.sidebar.title("Sarathi Dashboard")
    st.sidebar.caption("Temporary Streamlit interface")

    if st.sidebar.button("Reset Conversation"):
        st.session_state.chain.reset()
        st.session_state.messages = []
        st.session_state.last_docs = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Detected User Profile")
    profile = st.session_state.chain.get_user_profile()
    has_profile = False
    for key, value in profile.items():
        if value:
            has_profile = True
            pretty_key = key.replace("_", " ").title()
            st.sidebar.write(f"**{pretty_key}:** {value}")
    if not has_profile:
        st.sidebar.write("No profile info detected yet.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieved Scheme Links")
    links = _extract_unique_scheme_links(st.session_state.last_docs)
    if links:
        for item in links:
            st.sidebar.markdown(f"- [{item['scheme_name']}]({item['scheme_url']})")
    else:
        st.sidebar.write("No links yet. Ask a question.")


def main() -> None:
    st.set_page_config(page_title="Sarathi - Govt Schemes", page_icon="S", layout="wide")
    _init_state()

    st.title("Sarathi - Government Schemes Assistant")
    st.caption("Ask about Indian government schemes and get scheme URLs with recommendations.")

    _render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Example: I am 19 years old college student in Gujarat. Suggest schemes.")

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, docs, _ = st.session_state.chain.query(user_input)
            except Exception as exc:
                response = f"Error: {exc}"
                docs = []

        st.markdown(response)

        links = _extract_unique_scheme_links(docs)
        if links:
            st.markdown("**Quick Links**")
            for item in links:
                st.markdown(f"- [{item['scheme_name']}]({item['scheme_url']})")

    st.session_state.last_docs = docs
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
