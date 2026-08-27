# RAG-Docs — Probá RAG con tus propios documentos (Gemini/OpenAI + Streamlit)

Aplicación web de Retrieval-Augmented Generation (RAG) que permite consultar documentos propios (PDF, TXT, MD) usando inteligencia artificial. Subís tus archivos y el sistema crea un asistente que responde preguntas basándose exclusivamente en su contenido, citando la fuente. Usa ChromaDB para almacenamiento vectorial, embeddings de Google Gemini/OpenAI y un pipeline de chunking semántico para recuperar contexto relevante. Desplegada en Streamlit Community Cloud con soporte multi-proveedor de LLM. Sin registro: elegís tu proveedor y usás tu propia API key.

## Arquitectura

```
Tus archivos (PDF/TXT/MD o texto pegado)
      │
      ▼
┌──────────────────┐   chunks + FUENTE    ┌─────────────────────────────┐
│  user_kb.py      │ ───────────────────► │  ChromaDB efímera           │
│  parseo + cuotas │   Embeddings por     │  ./vectordb_users/user_{uuid}│
└──────────────────┘   API del proveedor  └─────────────────────────────┘
                                                    │ retriever k=5
                                                    ▼
                                          ┌─────────────────────────────┐
                                          │  LLM del proveedor elegido  │
                                          │  Gemini | GPT-4o mini       │
                                          └─────────────────────────────┘
                                                    │
                                                    ▼
                                        Respuesta + fuentes (UPLOAD-*)
```

Cada visitante obtiene una colección vectorial **aislada por sesión** (UUID), con límites anti-abuso y limpieza automática (TTL 4 h). El detalle completo está en [ARCHITECTURE.md](ARCHITECTURE.md).

## Stack

- **Streamlit** — interfaz de chat + sidebar de carga
- **ChromaDB** — base vectorial efímera por sesión (`./vectordb_users/`)
- **Embeddings por API** — `gemini-embedding-001` o `text-embedding-3-small` (según el proveedor que elijas)
- **LLM** — `ChatGoogleGenerativeAI` o `gpt-4o-mini`
- **LangChain** — pipeline RAG (loader, splitter, retriever, chains)
- **FastAPI** (opcional) — endpoint `/rag-api/get-response`

## Requisitos

- Python 3.9+
- Una API key propia:
  - Google Gemini (gratis): https://aistudio.google.com/apikey
  - OpenAI: https://platform.openai.com/api-keys

> No se necesita MySQL ni ninguna otra base: la demo funciona solo con tus archivos.

## Instalación

```bash
# 1. Crear y activar el entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (Opcional) configurar .env con GOOGLE_API_KEY como default local
cp .env.example .env         # Windows: copy .env.example .env
```

## Ejecutar

```bash
streamlit run app.py
```

1. En la **sidebar**: elegí proveedor, pegá tu API key (hay link para conseguirla gratis).
2. Subí hasta **3 archivos** (PDF/TXT/MD, máx. 5 MB c/u) **o pegá texto** y presioná **"Indexar mis documentos"**.
3. En el **chat**: preguntá lo que quieras sobre tus documentos. Expandí **"Ver fuentes"** para ver qué archivo usó el RAG.
4. Al terminar, botón **"🗑️ Borrar mis datos"**: tu colección desaparece del disco.

### Límites de la demo

| Límite | Valor |
| --- | --- |
| Tamaño por archivo | 5 MB |
| Archivos por sesión | 3 |
| Caracteres totales | 50.000 |
| Vida útil de la sesión | 4 horas (limpieza automática) |

> ⚠️ Si un PDF parece escaneado (sin capa de texto), la app lo detecta y avisa: probá con un PDF de texto o pegá su contenido manualmente.
>
> ℹ️ Si cambiás de proveedor después de indexar, la colección se borra automáticamente (los vectores son incompatibles entre modelos): volvé a indexar.

### API (opcional)

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/rag-api/get-response \
  -H "Content-Type: application/json" \
  -d '{"question": {"question": "De que trata mi documento?"}, "namespace": "tu_namespace", "provider": "gemini", "api_key": "TU_KEY"}'
```

El `namespace` identifica la colección efímera del usuario (en Streamlit es un UUID por sesión).

## Preguntas sugeridas para la demo

Sube cualquier documento propio (un manual, un reglamento, un CV) y probá:

- "¿De qué trata este documento? Hacé un resumen."
- Una pregunta específica cuyo answer esté en el archivo → respuesta citando `UPLOAD-*`.
- Una pregunta que NO esté en el documento → *"No encontré información sobre esto en los documentos proporcionados."* ← caso honestidad del RAG.

## Estructura del proyecto

```
RAG-Docs/
├── app.py               # UI Streamlit (proveedor, uploads, chat)
├── config.py            # PROVIDERS (modelos), límites de demo, prompt
├── rag.py               # Pipeline RAG (factories LLM/embeddings, retriever, chains)
├── user_kb.py           # Ingesta de archivos, cuotas, colecciones efímeras + TTL
├── main.py              # API FastAPI (colecciones efímeras por namespace)
├── tests/
│   └── test_user_kb.py  # Tests pytest (parseo, cuotas, índice, TTL)
├── database.py          # [LEGADO] acceso MySQL de la versión anterior
├── seed.sql             # [LEGADO] datos de ejemplo de la versión anterior
├── vectordb_users/      # Colecciones efímeras (generado en runtime, gitignored)
├── requirements.txt
├── .env.example         # Template de configuración
└── .gitignore
```

## Notas de confiabilidad

- **Sin key**: el chat no arranca hasta que ingreses una key válida; errores comunes (key inválida, cuota agotada) muestran mensajes claros.
- **Respuestas inventadas**: el system prompt restringe al contexto recuperado y las fuentes `UPLOAD-*` se muestran bajo cada respuesta.
- **Aislamiento**: cada sesión tiene su propia colección; nadie consulta datos de otro visitante.
- **Limpieza garantizada**: borrado manual + sweep automático de colecciones vencidas al arrancar la app.
- **Seguridad de keys**: la API key viaja solo en memoria de sesión del servidor; nunca se persiste ni loguea.

## Tests

```bash
pytest tests -q
```
