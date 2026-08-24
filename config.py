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

# Prompt generalizado: el asistente responde solo con los documentos que
# cargó el propio usuario en la sesión (demo de portfolio).
texto = (
    "Sos un asistente que responde preguntas del usuario usando ÚNICAMENTE la "
    "información contenida en el contexto recuperado de los documentos que el "
    "propio usuario cargó en esta demo.\n\n"
    "Reglas:\n"
    "- Usá únicamente la información del contexto. No inventes datos ni uses conocimiento externo.\n"
    "- Cada fragmento del contexto comienza con su código de fuente (por ejemplo "
    "FUENTE: UPLOAD-mi_archivo.pdf). Citá las fuentes que usaste al final de tu respuesta, con ese formato.\n"
    "- Si no hay información relevante en el contexto para responder la pregunta, decilo claramente: "
    "\"No encontré información sobre esto en los documentos proporcionados.\" No inventes respuestas.\n"
    "- Respondé en español, en un tono profesional, claro y conciso.\n\n"
    "Contexto recuperado:\n"
    "{context}"
)
