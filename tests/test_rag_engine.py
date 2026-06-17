import unittest
from types import SimpleNamespace

from pdf_rag_chatbot.rag_engine import RAGEngine, format_sources


class RAGEngineTest(unittest.TestCase):
    def test_ask_without_loaded_pdf_returns_user_message(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.qa_chain = None

        result = engine.ask("Was steht im Dokument?")

        self.assertEqual(
            result,
            {
                "answer": "Bitte zuerst ein PDF hochladen.",
                "sources": [],
            },
        )

    def test_format_sources_adds_one_based_pages_and_deduplicates(self):
        docs = [
            SimpleNamespace(metadata={"page": 0}, page_content="Erster Abschnitt\nmit Umbruch"),
            SimpleNamespace(metadata={"page": 0}, page_content="Erster Abschnitt\nmit Umbruch"),
            SimpleNamespace(metadata={"page": 2}, page_content="Anderer Abschnitt"),
        ]

        self.assertEqual(
            format_sources(docs),
            [
                {"page": 1, "snippet": "Erster Abschnitt mit Umbruch"},
                {"page": 3, "snippet": "Anderer Abschnitt"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
