# RECITALS Test Document

RECITALS LLm inference provides a natural-language interface for interacting with platform
documentation.

The user-facing chat interface is provided by Onyx. Onyx also provides the API
used by external clients.

The generation model is Mistral 7B Instruct, running through Ollama on a
dedicated GCP GPU virtual machine.

The RECITALS LLM inference / RAG / retrieval layer uses LlamaIndex for document chunking, embedding,
retrieval, and grounded context construction.

Qdrant is the vector database. It stores document chunks, vector embeddings,
and metadata required for retrieval and source attribution.

All RECITALS LLM inference services are deployed through Docker Compose.
