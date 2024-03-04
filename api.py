
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gemini_ai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
async def read_root():
    return {"message": "Hello World"}

@app.post("/api/gemini-fine-tuned/conclusion")
def gemini(request_body: dict):
    query = request_body.get("query", "")
    return gemini_ai.generate_conclusion(query)

@app.post("/api/gemma-fine-tuned/conclusion")
def gemini(request_body: dict):
    query = request_body.get("query", "")
    return gemini_ai.generate_conclusion(query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
