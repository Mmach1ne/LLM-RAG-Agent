from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
# Remove all inline agent, memory, task, and skill class definitions
# Import the advanced agent and enums from aiAgent.py
from backend.aiAgent import AIAgent, AgentState, Priority

# Initialize FastAPI app and agent singleton
app = FastAPI()
agent = AIAgent("Atlas")

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

class StatusResponse(BaseModel):
    state: str
    active_tasks: int
    memories: int

@app.post("/api/process", response_model=ProcessResponse)
def process(req: ProcessRequest) -> ProcessResponse:
    try:
        result = agent.process_input(req.input)
        # If the agent returns a dict, unpack it; else, treat as text
        if isinstance(result, dict):
            return ProcessResponse(**result)
        return ProcessResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    st = agent.get_status()
    # Only return the fields expected by StatusResponse
    return StatusResponse(
        state=st.get("state", "idle"),
        active_tasks=st.get("active_tasks", 0),
        memories=st.get("memories", 0)
    )

# To run: uvicorn fastapi_backend:app --reload
