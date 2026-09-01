import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


SUPPORTED_SUFFIXES = {".docx", ".md", ".markdown", ".txt"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def integer_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc

    if value < 0:
        raise RuntimeError(f"{name} must not be negative")

    return value


def document_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def extract_text(path: Path) -> str:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".docx":
        input_document = InputDocument(
            path_or_stream=path,
            format=InputFormat.DOCX,
            backend=MsWordDocumentBackend,
        )
        backend = input_document._backend

        if backend is None:
            raise RuntimeError("Docling did not initialize the DOCX backend")

        return backend.convert().export_to_markdown().strip()

    raise ValueError(f"Unsupported document type: {path.suffix or '<none>'}")


def discover_documents(directory: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    )
    supported = [
        candidate
        for candidate in files
        if candidate.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    unsupported = [
        candidate
        for candidate in files
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES
    ]
    return supported, unsupported


def ingest_directory(directory: Path) -> None:
    documents, unsupported = discover_documents(directory)

    if not documents and not unsupported:
        raise RuntimeError(f"No documents found in directory: {directory}")

    failures: list[tuple[str, str]] = []
    ingestion_failures = 0

    for path in documents:
        document_id = path.relative_to(directory).as_posix()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "app.ingest",
                str(path),
                "--document-id",
                document_id,
            ],
            check=False,
        )

        if result.returncode != 0:
            ingestion_failures += 1
            failures.append((document_id, f"exit code {result.returncode}"))

    for path in unsupported:
        relative_path = path.relative_to(directory).as_posix()
        failures.append((relative_path, "unsupported file type"))
        print(f"Unsupported document: {relative_path}", file=sys.stderr)

    print(
        f"Batch summary: discovered={len(documents) + len(unsupported)} "
        f"indexed={len(documents) - ingestion_failures} "
        f"failed={len(failures)}"
    )

    if failures:
        for document_id, reason in failures:
            print(f"FAILED: {document_id}: {reason}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index a supported document or directory tree into RECITALS Qdrant."
        )
    )
    parser.add_argument(
        "path",
        help="Path to a DOCX, Markdown or text document, or a directory.",
    )
    parser.add_argument(
        "--document-id",
        help="Stable document identifier. Defaults to the filename.",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()

    if path.is_dir():
        if args.document_id:
            raise ValueError("--document-id cannot be used with a directory")
        ingest_directory(path)
        return

    if not path.is_file():
        raise FileNotFoundError(f"Document does not exist: {path}")

    if path.is_symlink():
        raise ValueError(f"Symbolic links are not accepted: {path}")

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            "Ingestion supports only .docx, .md, .markdown and .txt files."
        )

    max_document_bytes = integer_env("MAX_DOCUMENT_BYTES", 25 * 1024 * 1024)
    document_size = path.stat().st_size

    if document_size > max_document_bytes:
        raise ValueError(
            f"Document exceeds MAX_DOCUMENT_BYTES ({max_document_bytes}): {path}"
        )

    text = extract_text(path)
    if not text:
        raise ValueError(f"Document contains no text: {path}")

    document_id = args.document_id or path.name
    content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    qdrant_url = required_env("QDRANT_URL")
    qdrant_api_key = required_env("QDRANT_API_KEY")
    collection_name = os.getenv(
        "QDRANT_COLLECTION",
        "recitals_documents",
    )

    ollama_base_url = required_env("OLLAMA_BASE_URL")
    embedding_model_name = required_env("OLLAMA_EMBED_MODEL")

    chunk_size = integer_env("CHUNK_SIZE", 400)
    chunk_overlap = integer_env("CHUNK_OVERLAP", 50)

    if chunk_size < 50:
        raise RuntimeError("CHUNK_SIZE must be at least 50")

    if chunk_overlap >= chunk_size:
        raise RuntimeError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=30,
    )

    embed_model = OllamaEmbedding(
        model_name=embedding_model_name,
        base_url=ollama_base_url,
        request_timeout=180.0,
    )

    # Obtain the dimension from the real configured embedding model.
    probe_embedding = embed_model.get_text_embedding(
        "RECITALS embedding dimension probe"
    )
    vector_size = len(probe_embedding)

    if vector_size <= 0:
        raise RuntimeError("Embedding model returned an empty vector")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(
            f"Created Qdrant collection {collection_name!r} "
            f"with vector size {vector_size}."
        )

    # Idempotent MVP re-index:
    # remove prior chunks for this stable document ID before inserting new ones.
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=document_filter(document_id)
        ),
        wait=True,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            embed_model,
        ],
        vector_store=vector_store,
    )

    document = Document(
        id_=document_id,
        text=text,
        metadata={
            "document_id": document_id,
            "file_name": path.name,
            "source_type": (
                "sftpgo"
                if path.is_relative_to(Path("/app/documents"))
                else "manual"
            ),
            "source_path": str(path),
            "content_sha256": content_sha256,
        },
    )

    nodes = pipeline.run(
        documents=[document],
        show_progress=True,
    )

    indexed_count = client.count(
        collection_name=collection_name,
        count_filter=document_filter(document_id),
        exact=True,
    ).count

    print(f"Document ID:       {document_id}")
    print(f"Content SHA256:    {content_sha256}")
    print(f"Embedding model:   {embedding_model_name}")
    print(f"Embedding size:    {vector_size}")
    print(f"Generated nodes:   {len(nodes)}")
    print(f"Indexed points:    {indexed_count}")


if __name__ == "__main__":
    main()
