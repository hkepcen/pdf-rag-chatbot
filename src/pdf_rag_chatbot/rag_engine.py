"""
Das Herzstück des PDF-Chatbots.

Ablauf (RAG-Pipeline):
  1. PDF laden         -> Text extrahieren
  2. Text splitten     -> in kleine Chunks zerlegen
  3. Embeddings        -> jeden Chunk in einen Zahlenvektor umwandeln
  4. ChromaDB          -> Vektoren speichern (persistiert auf Disk)
  5. Retriever         -> bei einer Frage die 4 ähnlichsten Chunks finden
  6. LLM               -> Frage + Chunks zusammen ans Modell schicken
"""

from typing import Any, Protocol, TypedDict

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import DEFAULT_CONFIG, RAGConfig
from .prompts import PROMPT_TEMPLATE


class Source(TypedDict):
    page: int
    snippet: str


class Answer(TypedDict):
    answer: str
    sources: list[Source]


class SourceDocument(Protocol):
    metadata: dict[str, Any]
    page_content: str


def format_sources(source_documents: list[SourceDocument]) -> list[Source]:
    """Bereitet LangChain-Quelldokumente fuer die UI auf."""
    sources: list[Source] = []
    seen: set[tuple[int, str]] = set()

    for doc in source_documents:
        page = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content[:200].replace("\n", " ")
        key = (page, snippet[:50])
        if key not in seen:
            seen.add(key)
            sources.append({"page": page, "snippet": snippet})

    return sources


class RAGEngine:
    """
    Kapselt die gesamte RAG-Logik.

    Verwendung:
        engine = RAGEngine()
        engine.ingest_pdf("mein_dokument.pdf")
        antwort = engine.ask("Was ist der Hauptinhalt?")
    """

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=config.chat_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.vectorstore = None
        self.qa_chain = None

        self._load_existing_db()

    def _load_existing_db(self):
        """Laedt eine bereits gespeicherte ChromaDB, falls vorhanden."""
        if self.config.chroma_dir.exists() and any(self.config.chroma_dir.iterdir()):
            self.vectorstore = Chroma(
                persist_directory=str(self.config.chroma_dir),
                embedding_function=self.embeddings,
            )
            self._build_qa_chain()

    def ingest_pdf(self, pdf_path: str) -> int:
        """
        Liest ein PDF ein und speichert es in ChromaDB.

        Gibt die Anzahl der erstellten Chunks zurueck.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.config.chroma_dir),
        )
        self._build_qa_chain()

        return len(chunks)

    def _build_qa_chain(self):
        """Baut die RetrievalQA-Chain auf."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.retrieval_results},
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def ask(self, question: str) -> Answer:
        """
        Stellt eine Frage an die RAG-Pipeline.

        Gibt ein Dict zurueck:
            {
                "answer": str,
                "sources": [{"page": int, "snippet": str}, ...]
            }
        """
        if not self.qa_chain:
            return {
                "answer": "Bitte zuerst ein PDF hochladen.",
                "sources": [],
            }

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": format_sources(result.get("source_documents", [])),
        }

    def clear_database(self):
        """Loescht die gesamte ChromaDB."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
        self.vectorstore = None
        self.qa_chain = None

    def is_ready(self) -> bool:
        """Gibt zurueck, ob ein PDF geladen und die Pipeline bereit ist."""
        return self.qa_chain is not None
