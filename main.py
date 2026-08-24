from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag import ai_with_sources, get_user_retriever

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/rag-api/get-response")
async def get_response(request: Request):
    payload = await request.json()

    question_data = payload.get('question')
    if isinstance(question_data, dict):
        question = question_data.get('question')
    else:
        question = question_data

    provider = payload.get('provider', 'gemini')
    api_key = payload.get('api_key') or ''
    namespace = payload.get('namespace')

    if not question:
        raise HTTPException(status_code=400, detail="Falta 'question' en el body.")
    if not namespace:
        raise HTTPException(
            status_code=400,
            detail=(
                "La demo funciona con datos propios: enviá 'namespace' "
                "(colección efímera del usuario), 'provider' y 'api_key'."
            ),
        )

    retriever = get_user_retriever(namespace, provider, api_key)
    result = ai_with_sources(question, provider=provider, api_key=api_key, retriever=retriever)
    return JSONResponse(content=jsonable_encoder(result))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
