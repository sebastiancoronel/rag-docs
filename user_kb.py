"""Gestión de bases de conocimiento efímeras subidas por el usuario.

Cada visitante de la demo obtiene un `namespace` (UUID) con su propia
colección Chroma en `config.USER_VECTORDIR`. Las colecciones se eliminan
al borrar los datos o automáticamente cuando superan el TTL.
"""

import tempfile
import time
import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

BATCH_SIZE = 100

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)


class QuotaExceeded(ValueError):
    """Se superó un límite de la demo (archivos, tamaño o caracteres)."""


def new_namespace():
    return uuid.uuid4().hex[:12]


def _client():
    return chromadb.PersistentClient(path=config.USER_VECTORDIR)


def _collection_names(client):
    cols = client.list_collections()
    return [c if isinstance(c, str) else c.name for c in cols]


def parse_upload(filename, data):
    """Convierte un archivo subido (nombre + bytes) en una lista de Documents.

    Soporta .pdf, .txt y .md. Lanza ValueError en formatos no soportados.
    """
    suffix = Path(filename).suffix.lower()
    fuente = f"UPLOAD-{Path(filename).name}"

    if suffix == ".pdf":
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            docs = PyPDFLoader(tmp.name).load()
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    elif suffix in (".txt", ".md"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        docs = [Document(page_content=text)]
    else:
        raise ValueError(f"Unsupported format: {suffix} (use PDF, TXT or MD)")

    for doc in docs:
        doc.metadata["FUENTE"] = fuente
    return docs


def detect_scanned(docs):
    """True si un PDF parece escaneado: casi sin capa de texto extraíble."""
    return sum(len(d.page_content.strip()) for d in docs) < config.SCANNED_MIN_CHARS


def check_quota(current_files, new_files, current_chars, new_chars):
    """Valida los límites acumulados de la sesión. Lanza QuotaExceeded."""
    if new_files <= 0 and new_chars <= 0:
        raise ValueError("There is no content to index.")
    if current_files + new_files > config.MAX_FILES:
        raise QuotaExceeded(
            f"Maximum of {config.MAX_FILES} files per session "
            f"(you already indexed {current_files})."
        )
    if current_chars + new_chars > config.MAX_TOTAL_CHARS:
        disponible = config.MAX_TOTAL_CHARS - current_chars
        raise QuotaExceeded(
            f"Session limit of {config.MAX_TOTAL_CHARS} characters exceeded "
            f"({max(disponible, 0)} remaining)."
        )


def get_user_vectorstore(namespace, embeddings):
    """Colección Chroma del usuario. Registra created_at solo al crearla."""
    client = _client()
    name = f"user_{namespace}"
    meta = None
    if name not in _collection_names(client):
        meta = {"created_at": str(time.time())}
    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        client=client,
        collection_metadata=meta,
    )


def index_user_docs(namespace, documents, embeddings):
    """Divide e indexa documentos en la colección del usuario.

    Cada chunk se antecede con su código de fuente para que el modelo
    pueda citarlo. Devuelve la cantidad de chunks indexados.
    """
    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("The content was empty after processing.")

    for split in splits:
        fuente = (split.metadata or {}).get("FUENTE")
        if fuente and not split.page_content.startswith("FUENTE:"):
            split.page_content = f"FUENTE: {fuente}\n\n{split.page_content}"

    vectorstore = get_user_vectorstore(namespace, embeddings)
    for i in range(0, len(splits), BATCH_SIZE):
        vectorstore.add_documents(splits[i:i + BATCH_SIZE])
    return len(splits)


def delete_user_data(namespace):
    """Elimina por completo la colección del usuario (si existe)."""
    client = _client()
    try:
        client.delete_collection(f"user_{namespace}")
    except Exception:
        pass  # no existía


def sweep_expired(ttl_hours=config.SESSION_TTL_HOURS):
    """Elimina colecciones user_* con antigüedad mayor al TTL.

    Devuelve la cantidad de colecciones eliminadas. Colecciones sin
    metadata created_at se conservan (comportamiento seguro por defecto).
    """
    client = _client()
    removed = 0
    now = time.time()
    for name in _collection_names(client):
        if not name.startswith("user_"):
            continue
        try:
            col = client.get_collection(name)
            created_at = float((col.metadata or {}).get("created_at", now))
        except Exception:
            continue
        if now - created_at > ttl_hours * 3600:
            try:
                client.delete_collection(name)
                removed += 1
            except Exception:
                pass
    return removed
