from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Simple Code Assistant API")

class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"
    context: Optional[List[str]] = None

@app.post("/generate")
async def generate_code(request: CodeRequest):
    try:
        # Simple response for testing
        return {
            "code": f"# Generated code for: {request.prompt}\n# Language: {request.language}\n\ndef example():\n    return 'This is a test response'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_code(request: CodeRequest):
    try:
        return {
            "analysis": "This is a test analysis response",
            "language": request.language,
            "timestamp": "2024-03-19"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_code(request: CodeRequest):
    try:
        return {
            "explanation": "This is a test explanation response"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080) 