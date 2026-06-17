"""Prompt templates used by the RAG pipeline."""

PROMPT_TEMPLATE = """Du bist ein hilfreicher Assistent, der Fragen zu den
hochgeladenen PDF-Dokumenten beantwortet.

Benutze AUSSCHLIESSLICH den folgenden Kontext aus den Dokumenten.
Wenn du die Antwort im Kontext nicht findest, sage:
"Diese Information ist in den hochgeladenen Dokumenten nicht vorhanden."

Kontext:
{context}

Frage: {question}

Antwort:"""
