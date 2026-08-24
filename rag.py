import re
import warnings

import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)


def build_llm(provider, api_key):
    if provider not in config.PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider}")
    p = config.PROVIDERS[provider]
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=p["chat_model"],
            temperature=0,
            google_api_key=api_key,
        )
    if provider == "openai":
        return ChatOpenAI(
            model=p["chat_model"],
            temperature=0,
            api_key=api_key,
        )
    raise ValueError(f"Proveedor no soportado: {provider}")


def build_embeddings(provider, api_key):
    if provider not in config.PROVIDERS:
        raise ValueError(f"Proveedor no soportado: {provider}")
    p = config.PROVIDERS[provider]
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=p["embed_model"],
            google_api_key=api_key,
        )
    if provider == "openai":
        return OpenAIEmbeddings(
            model=p["embed_model"],
            api_key=api_key,
        )
    raise ValueError(f"Proveedor no soportado: {provider}")


def get_user_retriever(namespace, provider, api_key, k=5):
    """Retriever sobre la colección efímera del usuario."""
    from user_kb import get_user_vectorstore

    vectorstore = get_user_vectorstore(namespace, build_embeddings(provider, api_key))
    return vectorstore.as_retriever(search_kwargs={"k": k})


def prompt(texto):
    return ChatPromptTemplate.from_messages(
        [
            ("system", texto),
            ("human", "{input}"),
        ])


def respuesta(pregunta, llm, retriever, prompt):
    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, chain)
    return rag.invoke({"input": pregunta})


def ai_with_sources(question, provider=config.DEFAULT_PROVIDER, api_key=None, retriever=None):
    label = config.PROVIDERS.get(provider, {}).get("label", provider)
    if not api_key:
        return {
            'answer': f'No se configuró la API key de {label}. Agregala en la barra lateral.',
            'sources': [],
        }
    if retriever is None:
        raise ValueError(
            "Falta el retriever: la demo funciona con colecciones efímeras por usuario "
            "(usá get_user_retriever(namespace, ...))."
        )
    llm = build_llm(provider, api_key)
    response = respuesta(question, llm, retriever, prompt(config.texto))
    answer = response.get('answer', '')
    sources = []
    for doc in response.get('context', []):
        fuente = (doc.metadata or {}).get('FUENTE')
        if not fuente:
            match = re.search(r'FUENTE:\s*(\S+)', doc.page_content)
            fuente = match.group(1) if match else None
        if fuente and fuente not in sources:
            sources.append(fuente)
    return {'answer': answer, 'sources': sources}


def ai(question, **kwargs):
    return ai_with_sources(question, **kwargs)['answer']
