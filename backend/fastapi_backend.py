from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import traceback
import aiofiles
import os
import tempfile

# Import the advanced agent and enums + Config from aiAgent.py
from backend.aiAgent import RAGAIAgent as AIAgent, AgentState, Priority, Config

# Initialize FastAPI app
app = FastAPI()

# Initialize agent singleton
agent = AIAgent("Raybot")

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ProcessRequest(BaseModel):
    input: str

class ProcessResponse(BaseModel):
    response: str
    widget: Optional[Dict] = None
    type: str = "general"

@app.post("/api/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    try:
        if asyncio.iscoroutinefunction(agent.process_input):
            result = await agent.process_input(req.input)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, agent.process_input, req.input)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, asyncio.Task):
            result = await result
        if isinstance(result, dict):
            response_data = {
                "response": result.get("response", str(result)),
                "type": result.get("type", "general")
            }
            if "widget" in result:
                response_data["widget"] = result["widget"]
            return ProcessResponse(**response_data)
        return ProcessResponse(response=str(result), type="general")
    except Exception as e:
        traceback.print_exc()
        return ProcessResponse(response=f"Error: {e}", type="error")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base"""
    tmp_dir = tempfile.gettempdir()
    temp_path = os.path.join(tmp_dir, file.filename)

    # Save file temporarily
    async with aiofiles.open(temp_path, 'wb') as out:
        content = await file.read()
        await out.write(content)

    try:
        result = await agent.ingest_document(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

    # Clean up
    try:
        os.remove(temp_path)
    except OSError:
        pass

    # Return result
    if result.get("success"):
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=500, detail=result.get("message", "Ingestion failed"))

@app.get("/api/status")
async def status():
    try:
        return agent.get_status()
    except Exception as e:
        traceback.print_exc()
        return {"state": "error", "active_tasks": 0, "memories": 0, "error": str(e)}

@app.get("/api/debug/process-test")
async def test_process():
    try:
        test_input = "Hello, how are you?"
        is_async = asyncio.iscoroutinefunction(agent.process_input)
        if is_async:
            result = await agent.process_input(test_input)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, agent.process_input, test_input)
        return {"success": True, "input": test_input, "result": str(result), "result_type": type(result).__name__, "is_async": is_async}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": agent.name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

@app.get("/api/debug/memory")
async def debug_memory():
    return agent.debug_memory()
