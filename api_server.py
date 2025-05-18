from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from code_assistant import CodeAssistant
import uvicorn

app = FastAPI(title="Personal Code Assistant API")
assistant = CodeAssistant()

class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"
    context: Optional[List[str]] = None

class CodeAnalysisRequest(BaseModel):
    code: str
    language: str = "python"

@app.post("/generate")
async def generate_code(request: CodeRequest):
    try:
        code = assistant.generate_code(
            prompt=request.prompt,
            language=request.language,
            context=request.context
        )
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    try:
        analysis = assistant.analyze_code(
            code=request.code,
            language=request.language
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain_code(request: CodeAnalysisRequest):
    try:
        explanation = assistant.explain_code(
            code=request.code,
            language=request.language
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True) 