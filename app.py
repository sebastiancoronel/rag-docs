import streamlit as st
from langchain_core.documents import Document

import config
import user_kb
from rag import ai_with_sources, build_embeddings, get_user_retriever

st.set_page_config(page_title="RAG-Docs - Probá con tus propios datos", layout="wide")


# ---------------- Estado de sesión ----------------
def init_state():
    defaults = {
        "namespace": user_kb.new_namespace(),
        "provider": config.DEFAULT_PROVIDER,
        "indexed": False,
        "files": [],
        "chunks": 0,
        "total_chars": 0,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# Limpieza de sesiones abandonadas (una vez por visita)
if "swept" not in st.session_state:
    user_kb.sweep_expired()
    st.session_state.swept = True


def reset_user_data(new_namespace=True):
    """Borra la colección efímera y el estado asociado."""
    user_kb.delete_user_data(st.session_state.namespace)
    if new_namespace:
        st.session_state.namespace = user_kb.new_namespace()
    st.session_state.indexed = False
    st.session_state.files = []
    st.session_state.chunks = 0
    st.session_state.total_chars = 0
    st.session_state.messages = []


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Configuración")

    provider = st.selectbox(
        "Proveedor de IA",
        options=list(config.PROVIDERS),
        format_func=lambda p: config.PROVIDERS[p]["label"],
        index=list(config.PROVIDERS).index(st.session_state.provider),
    )
    if provider != st.session_state.provider:
        had_data = st.session_state.indexed
        reset_user_data()
        st.session_state.provider = provider
        if had_data:
            st.toast(
                "Cambiaste de proveedor: los vectores eran incompatibles "
                "y se borraron. Volvé a indexar tus documentos."
            )

    provider_info = config.PROVIDERS[st.session_state.provider]
    api_key = st.text_input(
        f"API Key de {provider_info['label']}",
        type="password",
    )
    st.markdown(f"[Consigue tu API key gratis]({provider_info['key_url']})")

    st.divider()
    st.subheader("Tus documentos")

    uploaded_files = st.file_uploader(
        "Sube archivos (PDF o texto)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    pasted_text = st.text_area("…o pega tu texto aquí", height=120)

    if st.button("Indexar mis documentos", use_container_width=True):
        errors = []
        warnings_list = []

        if not api_key:
            errors.append("Primero ingresá tu API key.")

        files = uploaded_files or []
        if len(files) > config.MAX_FILES:
            errors.append(
                f"Máximo {config.MAX_FILES} archivos por subida."
            )
        oversized = [
            f.name for f in files if f.size > config.MAX_FILE_MB * 1024 * 1024
        ]
        if oversized:
            errors.append(
                f"Archivo(s) mayores a {config.MAX_FILE_MB} MB: "
                f"{', '.join(oversized)}"
            )

        documents = []
        valid_files = []
        new_chars = 0

        if not errors:
            for f in files:
                try:
                    docs = user_kb.parse_upload(f.name, f.getvalue())
                except Exception as exc:
                    errors.append(f"{f.name}: {exc}")
                    continue
                if user_kb.detect_scanned(docs):
                    warnings_list.append(
                        f"⚠️ `{f.name}` parece escaneado (contiene imágenes sin "
                        "texto). Probá con un PDF de texto o pegá su contenido."
                    )
                    continue
                documents.extend(docs)
                valid_files.append(f.name)
                new_chars += sum(len(d.page_content) for d in docs)

            if pasted_text.strip():
                documents.append(Document(
                    page_content=pasted_text.strip(),
                    metadata={"FUENTE": "UPLOAD-texto_pegado"},
                ))
                new_chars += len(pasted_text.strip())

            try:
                user_kb.check_quota(
                    current_files=len(st.session_state.files),
                    new_files=len(valid_files) + (1 if pasted_text.strip() else 0),
                    current_chars=st.session_state.total_chars,
                    new_chars=new_chars,
                )
                chunk_count = user_kb.index_user_docs(
                    st.session_state.namespace,
                    documents,
                    build_embeddings(st.session_state.provider, api_key),
                )
            except user_kb.QuotaExceeded as exc:
                errors.append(str(exc))
            except ValueError as exc:
                errors.append(str(exc))
            else:
                st.session_state.files.extend(valid_files)
                if pasted_text.strip():
                    st.session_state.files.append("texto pegado")
                st.session_state.chunks += chunk_count
                st.session_state.total_chars += new_chars
                st.session_state.indexed = True
                st.success(f"Listo: {chunk_count} fragmentos indexados.")
                for warning in warnings_list:
                    st.markdown(warning)

        for error in errors:
            st.error(error)
        if not errors:
            for warning in warnings_list:
                st.markdown(warning)

    st.divider()
    st.subheader("Estado de la sesión")
    if st.session_state.indexed:
        st.write(f"📄 Archivos: {len(st.session_state.files)}")
        st.write(f"🧩 Fragmentos: {st.session_state.chunks}")
        st.write(f"🔤 Caracteres: {st.session_state.total_chars:,}")
        if st.button("🗑️ Borrar mis datos", use_container_width=True):
            reset_user_data()
            st.rerun()
    else:
        st.caption("Todavía no indexaste documentos.")

# ---------------- Chat principal ----------------
st.title("Probá el RAG con tus propios datos")
st.caption(
    "Subí PDFs o texto en la barra lateral, indexalos y preguntale lo que "
    "quieras: responde únicamente con tu información, citando fuentes."
)

if not st.session_state.indexed:
    st.info("👈 Sube tus documentos y presiona **Indexar mis documentos** para empezar.")


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Ver fuentes"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")


for msg in st.session_state.messages:
    render_message(msg)

question = st.chat_input(
    "Hacé tu pregunta sobre tus documentos...",
    disabled=not st.session_state.indexed,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    with st.chat_message("assistant"):
        if not api_key:
            assistant_msg = {
                "role": "assistant",
                "content": (
                    f"No se configuró la API key de "
                    f"{config.PROVIDERS[st.session_state.provider]['label']}. "
                    "Agregala en la barra lateral."
                ),
                "sources": [],
            }
            st.warning(assistant_msg["content"])
        else:
            with st.spinner("Buscando en tus documentos..."):
                try:
                    retriever = get_user_retriever(
                        st.session_state.namespace,
                        st.session_state.provider,
                        api_key,
                    )
                    result = ai_with_sources(
                        question,
                        provider=st.session_state.provider,
                        api_key=api_key,
                        retriever=retriever,
                    )
                    assistant_msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Ver fuentes"):
                            for src in result["sources"]:
                                st.markdown(f"- `{src}`")
                except Exception as exc:
                    message = str(exc)
                    lowered = message.lower()
                    if any(token in lowered for token in (
                        "api key", "api_key", "401", "403",
                        "unauthorized", "invalid_api_key",
                    )):
                        friendly = (
                            "Tu API key parece inválida o sin permisos. "
                            "Revisala en la barra lateral."
                        )
                    elif "429" in lowered or "quota" in lowered or "rate" in lowered:
                        friendly = (
                            "Se alcanzó el límite de uso de tu API key. "
                            "Esperá un momento e intentá de nuevo."
                        )
                    else:
                        friendly = f"Ocurrió un error: {message}"
                    st.error(friendly)
                    assistant_msg = {
                        "role": "assistant",
                        "content": friendly,
                        "sources": [],
                    }
        st.session_state.messages.append(assistant_msg)
