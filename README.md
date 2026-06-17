# PDF RAG Chatbot

Ein lokal laufender Chatbot, der Fragen zu beliebigen PDF-Dokumenten beantwortet –
basierend auf Retrieval-Augmented Generation (RAG).

## Architektur

```
PDF-Datei
    │
    ▼
PyPDFLoader          ← Text aus PDF extrahieren
    │
    ▼
RecursiveCharacterTextSplitter   ← in ~1.000-Zeichen-Chunks aufteilen
    │
    ▼
Ollama Embeddings    ← jeden Chunk in einen Vektor umwandeln
    │
    ▼
ChromaDB             ← Vektoren persistent auf Disk speichern
    │
    ▼  (bei Frage)
Cosine Similarity    ← Top-4 ähnlichste Chunks zur Frage finden
    │
    ▼
Ollama LLM           ← Frage + Chunks → Antwort generieren
    │
    ▼
Streamlit UI         ← Antwort + Quellenangaben anzeigen
```

## Tech Stack

| Komponente | Technologie                  |
|---|------------------------------|
| LLM | Ollama / llama3.2            |
| Embeddings | Ollama / nomic-embed-text    |
| Vektordatenbank | ChromaDB (lokal, persistent) |
| RAG-Framework | LangChain                    |
| Web-UI | Streamlit                    |
| PDF-Parsing | PyPDF                        |

## Lokale Installation

```bash
# 1. Repository klonen
git clone https://github.com/dein-name/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. Virtuelle Umgebung erstellen
# Empfohlen: Python 3.12. Python 3.14 kann bei gepinnten Dependencies Build-Probleme verursachen.
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Dependencies installieren
pip install -r requirements.txt
pip install -e .

# 4. Ollama starten und Modelle herunterladen
# Falls Ollama noch nicht laeuft:
ollama serve

# In einem zweiten Terminal:
ollama pull llama3.2
ollama pull nomic-embed-text

# 5. Optionale Konfiguration setzen
cp .env.example .env
# Defaults funktionieren, wenn Ollama lokal auf http://localhost:11434 laeuft

# 6. App starten
streamlit run app.py
```

Die App öffnet sich automatisch unter http://localhost:8501

Wichtig: Wenn du vorher mit OpenAI-Embeddings getestet hast, nutze fuer Ollama eine neue ChromaDB. Standardmaessig verwendet das Projekt jetzt `chroma_db_ollama`.

## Projektstruktur

```text
pdf-rag-chatbot/
├── app.py                         # Streamlit-Einstiegspunkt
├── src/
│   └── pdf_rag_chatbot/
│       ├── __init__.py
│       ├── config.py              # Modelle, Chunking und Retrieval-Konstanten
│       ├── prompts.py             # Prompt-Templates
│       └── rag_engine.py          # RAG-Pipeline
├── tests/
│   └── test_rag_engine.py
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Tests

Die Tests prüfen aktuell die Logik, die ohne Ollama-Aufruf lokal ausführbar ist.

```bash
source .venv/bin/activate
python -m unittest discover tests
```

## Wichtige Design-Entscheidungen

**Chunk-Größe (1.000 Zeichen, 200 Überlappung)**
Kleinere Chunks erhöhen die Retrieval-Präzision, verlieren aber Kontext.
Größere Chunks behalten Kontext, senken aber die Ähnlichkeits-Genauigkeit.
1.000/200 ist ein bewährter Ausgangspunkt für technische Dokumente.

**temperature=0 beim LLM**
Für faktische Q&A ist Determinismus wichtiger als Kreativität.
Höhere temperature würde "kreativere", aber weniger zuverlässige Antworten erzeugen.

**Prompt-Engineering**
Das System-Prompt weist das LLM explizit an, nur den gegebenen Kontext zu nutzen.
Das verhindert Halluzinationen – das LLM soll keine Fakten erfinden.

## Mögliche Erweiterungen

- [ ] Multi-PDF Support (mehrere Dokumente gleichzeitig)
- [ ] Hybrid Search (Vektorsuche + Keyword-Suche kombinieren)
- [ ] MLflow für Retrieval-Qualitäts-Tracking
- [ ] Deployment auf AWS (EC2 + Docker)
- [ ] Evaluation mit RAGAS (Retrieval-Metriken)

## Kosten

Die App nutzt Ollama lokal. Dadurch entstehen keine API-Kosten. Die Antwortgeschwindigkeit haengt von deinem Rechner und dem gewaehlten Modell ab.
