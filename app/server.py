from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/test")
async def HelloThere():
    return {"message": "Hello World"}

class ModeloLinguagem:

    def __init__(self):
        pass
    
    def get_input_schema(self):
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    
if __name__ == "__main__":

    modelo_linguagem = ModeloLinguagem()
    add_routes(app, modelo_linguagem)
    
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
