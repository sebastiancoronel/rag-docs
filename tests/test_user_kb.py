"""Tests de user_kb: parseo, cuotas, indexado efímero y limpieza TTL."""

import time

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config
import user_kb


class FakeEmbeddings(Embeddings):
    """Embeddings deterministas de dimensión fija (no requieren red ni keys)."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.5]


@pytest.fixture
def vectordir(tmp_path, monkeypatch):
    path = tmp_path / "vectordb_users"
    monkeypatch.setattr(config, "USER_VECTORDIR", str(path))
    return str(path)


def test_new_namespace_unico():
    ns1 = user_kb.new_namespace()
    ns2 = user_kb.new_namespace()
    assert ns1 != ns2
    assert len(ns1) == 12


def test_parse_txt_utf8():
    docs = user_kb.parse_upload("notas.txt", "hola café".encode("utf-8"))
    assert len(docs) == 1
    assert "café" in docs[0].page_content
    assert docs[0].metadata["FUENTE"] == "UPLOAD-notas.txt"


def test_parse_md():
    docs = user_kb.parse_upload("guia.md", "# Título\ncontenido".encode("utf-8"))
    assert "Título" in docs[0].page_content
    assert docs[0].metadata["FUENTE"] == "UPLOAD-guia.md"


def test_parse_txt_latin1_fallback():
    docs = user_kb.parse_upload("notas.txt", "año".encode("latin-1"))
    assert "año" in docs[0].page_content


def test_parse_formato_invalido():
    with pytest.raises(ValueError):
        user_kb.parse_upload("foto.png", b"\x89PNG")


def test_detect_scanned():
    assert user_kb.detect_scanned([Document(page_content="")]) is True
    assert user_kb.detect_scanned(
        [Document(page_content="x" * config.SCANNED_MIN_CHARS)]
    ) is False


def test_quota_max_archivos():
    with pytest.raises(user_kb.QuotaExceeded):
        user_kb.check_quota(
            current_files=config.MAX_FILES,
            new_files=1,
            current_chars=0,
            new_chars=10,
        )


def test_quota_max_caracteres():
    with pytest.raises(user_kb.QuotaExceeded):
        user_kb.check_quota(
            current_files=0,
            new_files=1,
            current_chars=config.MAX_TOTAL_CHARS - 100,
            new_chars=200,
        )


def test_quota_sin_contenido():
    with pytest.raises(ValueError):
        user_kb.check_quota(0, 0, 0, 0)


def test_quota_ok():
    user_kb.check_quota(
        current_files=1,
        new_files=2,
        current_chars=10_000,
        new_chars=20_000,
    )


def test_index_y_delete_roundtrip(vectordir):
    docs = [
        Document(
            page_content=(
                "El procedimiento de reset requiere reiniciar el equipo y "
                "esperar dos minutos antes de volver a encenderlo."
            ),
            metadata={"FUENTE": "UPLOAD-manual.md"},
        )
    ]

    chunks = user_kb.index_user_docs("abc123", docs, FakeEmbeddings())
    assert chunks >= 1

    client = user_kb._client()
    assert "user_abc123" in user_kb._collection_names(client)

    stored = client.get_collection("user_abc123").get()
    assert len(stored["ids"]) == chunks
    for content in stored["documents"]:
        assert content.startswith("FUENTE: UPLOAD-manual.md")

    user_kb.delete_user_data("abc123")
    assert "user_abc123" not in user_kb._collection_names(client)


def test_sweep_expired_elimina_solo_vencidas(vectordir):
    old_doc = [Document(page_content="contenido antiguo",
                        metadata={"FUENTE": "UPLOAD-viejo.txt"})]
    new_doc = [Document(page_content="contenido reciente",
                        metadata={"FUENTE": "UPLOAD-nuevo.txt"})]
    user_kb.index_user_docs("viejo", old_doc, FakeEmbeddings())
    user_kb.index_user_docs("nuevo", new_doc, FakeEmbeddings())

    client = user_kb._client()
    client.get_collection("user_viejo").modify(
        metadata={"created_at": str(time.time() - (config.SESSION_TTL_HOURS + 1) * 3600)}
    )

    removed = user_kb.sweep_expired(ttl_hours=config.SESSION_TTL_HOURS)

    assert removed == 1
    names = user_kb._collection_names(client)
    assert "user_viejo" not in names
    assert "user_nuevo" in names
