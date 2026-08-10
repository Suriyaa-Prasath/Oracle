"""Oracle — a multi-agent RAG system over personal resume/portfolio documents.

Package layout
--------------
- `config`    : all tunables (models, paths, chunk sizes) in one place
- `ingest`    : load + chunk + embed documents into ChromaDB
- `retrieve`  : query ChromaDB for the top-k relevant chunks
- `agents`    : LangGraph graph — router, research, tool, synthesis nodes
- `tools`     : callable tools (web search, calculator, Wikipedia)
- `memory`    : conversation history handling for multi-turn chat
"""

__version__ = "0.1.0"
