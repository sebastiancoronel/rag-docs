import streamlit as st
from langchain_core.documents import Document

import config
import user_kb
from rag import ai_with_sources, build_embeddings, get_user_retriever

st.set_page_config(page_title="RAG-Docs - Try it with your own data", layout="wide")


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
    st.title("Configuration")

    provider = st.selectbox(
        "AI provider",
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
                "You switched providers: the vectors were incompatible and were "
                "deleted. Re-index your documents."
            )

    provider_info = config.PROVIDERS[st.session_state.provider]
    api_key = st.text_input(
        f"{provider_info['label']} API key",
        type="password",
    )
    st.markdown(f"[Get a free API key]({provider_info['key_url']})")

    st.divider()
    st.subheader("Your documents")

    uploaded_files = st.file_uploader(
        "Upload files (PDF or text)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    pasted_text = st.text_area("…or paste your text here", height=120)

    if st.button("Index my documents", use_container_width=True):
        errors = []
        warnings_list = []

        if not api_key:
            errors.append("Enter your API key first.")

        files = uploaded_files or []
        if len(files) > config.MAX_FILES:
            errors.append(
                f"Maximum of {config.MAX_FILES} files per upload."
            )
        oversized = [
            f.name for f in files if f.size > config.MAX_FILE_MB * 1024 * 1024
        ]
        if oversized:
            errors.append(
                f"File(s) larger than {config.MAX_FILE_MB} MB: "
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
                        f"⚠️ `{f.name}` looks scanned (it contains images with no "
                        "text). Try a text PDF or paste its content."
                    )
                    continue
                documents.extend(docs)
                valid_files.append(f.name)
                new_chars += sum(len(d.page_content) for d in docs)

            if pasted_text.strip():
                documents.append(Document(
                    page_content=pasted_text.strip(),
                    metadata={"FUENTE": "UPLOAD-pasted_text"},
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
                    st.session_state.files.append("pasted text")
                st.session_state.chunks += chunk_count
                st.session_state.total_chars += new_chars
                st.session_state.indexed = True
                st.success(f"Done: {chunk_count} chunks indexed.")
                for warning in warnings_list:
                    st.markdown(warning)

        for error in errors:
            st.error(error)
        if not errors:
            for warning in warnings_list:
                st.markdown(warning)

    st.divider()
    st.subheader("Session status")
    if st.session_state.indexed:
        st.write(f"📄 Files: {len(st.session_state.files)}")
        st.write(f"🧩 Chunks: {st.session_state.chunks}")
        st.write(f"🔤 Characters: {st.session_state.total_chars:,}")
        if st.button("🗑️ Delete my data", use_container_width=True):
            reset_user_data()
            st.rerun()
    else:
        st.caption("You haven't indexed any documents yet.")

# ---------------- Chat principal ----------------
st.title("Try RAG with your own data")
st.caption(
    "Upload PDFs or text in the sidebar, index them and ask whatever you "
    "want: it answers using only your information, citing sources."
)

if not st.session_state.indexed:
    st.info("👈 Upload your documents and press **Index my documents** to get started.")


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")


for msg in st.session_state.messages:
    render_message(msg)

question = st.chat_input(
    "Ask a question about your documents...",
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
                    f"No API key configured for "
                    f"{config.PROVIDERS[st.session_state.provider]['label']}. "
                    "Add it in the sidebar."
                ),
                "sources": [],
            }
            st.warning(assistant_msg["content"])
        else:
            with st.spinner("Searching your documents..."):
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
                        with st.expander("View sources"):
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
                            "Your API key looks invalid or lacks permissions. "
                            "Check it in the sidebar."
                        )
                    elif "429" in lowered or "quota" in lowered or "rate" in lowered:
                        friendly = (
                            "Your API key has reached its usage limit. "
                            "Wait a moment and try again."
                        )
                    else:
                        friendly = f"An error occurred: {message}"
                    st.error(friendly)
                    assistant_msg = {
                        "role": "assistant",
                        "content": friendly,
                        "sources": [],
                    }
        st.session_state.messages.append(assistant_msg)
