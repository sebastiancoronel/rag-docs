import os

from dotenv import load_dotenv

load_dotenv()

model = 'gemini-3-flash-preview'

google_api_key = os.getenv('GOOGLE_API_KEY', '')

# ---------------- Límites de la demo ----------------
MAX_FILE_MB = 5
MAX_FILES = 3
MAX_TOTAL_CHARS = 50_000
SESSION_TTL_HOURS = 4
SCANNED_MIN_CHARS = 50

# Carpeta de colecciones efímeras de usuarios (separada de la base principal)
USER_VECTORDIR = "./vectordb_users"

DEFAULT_PROVIDER = "gemini"

PROVIDERS = {
    "gemini": {
        "label": "Google Gemini",
        "chat_model": model,
        "embed_model": "models/gemini-embedding-001",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "label": "OpenAI",
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "key_url": "https://platform.openai.com/api-keys",
    },
}

# Generalized prompt: the assistant answers only from the documents the user
# uploaded in this session (portfolio demo).
texto = (
    "You are an assistant that answers the user's questions using ONLY the "
    "information contained in the context retrieved from the documents that "
    "the user uploaded in this demo.\n\n"
    "Rules:\n"
    "- Use only the information from the context. Do not make things up or use external knowledge.\n"
    "- Each context chunk begins with its source code (for example "
    "FUENTE: UPLOAD-my_file.pdf). Cite the sources you used at the end of your answer, using that format.\n"
    "- If there is no relevant information in the context to answer the question, say so clearly: "
    "\"I couldn't find information about this in the provided documents.\" Do not invent answers.\n"
    "- Answer in the same language the user uses to ask the question, in a professional, clear and concise tone.\n\n"
    "Retrieved context:\n"
    "{context}"
)
