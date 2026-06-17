"""
Streamlit Web-UI fuer den PDF-Chatbot.

Starten mit:
    streamlit run app.py
"""

import os
import tempfile
from typing import Any

import streamlit as st

from pdf_rag_chatbot import RAGEngine


Message = dict[str, Any]


def initialize_session_state() -> None:
    """Initialisiert alle Streamlit-Session-Werte an einer Stelle."""
    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine()

    defaults = {
        "messages": [],
        "pdf_loaded": False,
        "pdf_name": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state() -> None:
    st.session_state.pdf_loaded = False
    st.session_state.pdf_name = None
    st.session_state.messages = []


def save_uploaded_pdf(uploaded_file) -> str:
    """Speichert Streamlits Upload temporaer und gibt den Dateipfad zurueck."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def ingest_uploaded_pdf(uploaded_file, engine: RAGEngine) -> None:
    if uploaded_file.name == st.session_state.pdf_name:
        return

    with st.spinner("PDF wird verarbeitet..."):
        tmp_path = save_uploaded_pdf(uploaded_file)
        try:
            chunk_count = engine.ingest_pdf(tmp_path)
            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            st.success(f"{chunk_count} Chunks indexiert")
        except Exception as exc:
            st.error(f"Fehler beim Verarbeiten: {exc}")
        finally:
            os.unlink(tmp_path)


def render_sidebar(engine: RAGEngine) -> None:
    with st.sidebar:
        st.header("Dokument")

        uploaded_file = st.file_uploader(
            "PDF hochladen",
            type=["pdf"],
            help="Maximale Dateigroesse: ca. 20 MB empfohlen",
        )
        if uploaded_file is not None:
            ingest_uploaded_pdf(uploaded_file, engine)

        if st.session_state.pdf_loaded:
            st.info(f"**Aktives Dokument:**\n{st.session_state.pdf_name}")
        else:
            st.warning("Kein Dokument geladen")

        st.divider()

        if st.button("Datenbank leeren", disabled=not st.session_state.pdf_loaded):
            engine.clear_database()
            reset_document_state()
            st.rerun()

        with st.expander("Technischer Stack"):
            st.markdown(
                f"""
                - **LLM:** Ollama / {engine.config.chat_model}
                - **Embeddings:** Ollama / {engine.config.embedding_model}
                - **Vektordatenbank:** ChromaDB (lokal)
                - **Framework:** LangChain
                - **UI:** Streamlit
                - **Chunk-Groesse:** 1.000 Zeichen
                - **Retrieval:** Top-{engine.config.retrieval_results} per Cosine Similarity
                """
            )


def render_sources(sources: list[dict[str, Any]], title: str = "Quellen") -> None:
    if not sources:
        return

    with st.expander(title):
        for source in sources:
            st.caption(f"**Seite {source['page']}:** {source['snippet']}...")


def render_message(message: Message) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def render_chat_history() -> None:
    for message in st.session_state.messages:
        render_message(message)


def append_message(role: str, content: str, sources: list[dict[str, Any]] | None = None) -> None:
    message: Message = {"role": role, "content": content}
    if sources is not None:
        message["sources"] = sources

    st.session_state.messages.append(message)


def render_chat_input(engine: RAGEngine) -> None:
    prompt = st.chat_input(
        "Stelle eine Frage zu deinem PDF...",
        disabled=not st.session_state.pdf_loaded,
    )
    if not prompt:
        return

    append_message("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Antwort wird generiert..."):
            result = engine.ask(prompt)

        st.write(result["answer"])
        render_sources(result["sources"], "Quellen aus dem Dokument")

    append_message("assistant", result["answer"], result["sources"])


def render_empty_state() -> None:
    if st.session_state.pdf_loaded or st.session_state.messages:
        return

    st.info("Lade links ein PDF hoch um zu starten.")
    st.subheader("Was kann dieser Chatbot?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Beispiel-Fragen:**
            - "Fasse das Dokument in 3 Saetzen zusammen."
            - "Was sind die wichtigsten Kernaussagen?"
            - "Was steht auf Seite 5?"
            """
        )
    with col2:
        st.markdown(
            """
            **Wie es funktioniert:**
            1. PDF -> Text extrahieren
            2. Text -> Chunks
            3. Chunks -> Vektoren
            4. Frage -> aehnliche Chunks finden
            5. Chunks + Frage -> Ollama
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        page_icon="📄",
        layout="wide",
    )
    st.title("PDF RAG Chatbot")
    st.caption(
        "Lade ein PDF hoch und stelle Fragen dazu. "
        "Antworten basieren ausschliesslich auf dem Inhalt deines Dokuments."
    )

    initialize_session_state()
    engine: RAGEngine = st.session_state.engine

    render_sidebar(engine)
    render_chat_history()
    render_chat_input(engine)
    render_empty_state()


if __name__ == "__main__":
    main()
