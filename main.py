import os
from dotenv import load_dotenv
from functools import lru_cache
from crewai import LLM, Agent, Task, Crew, Process
from datetime import datetime
import requests
import json
from crewai.tools import tool
from crewai_tools import SerperDevTool
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import time
from passlib.context import CryptContext

# Import your database collections and travel_chatbot functions
from database import users_collection, chats_collection
from travel_chatbot import (
    create_setup_crew,
    invoke_agent,
    extract_json_from_response,
    human_input_tool,
)

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

app = FastAPI(title="Travel Chatbot API")

# CORS Middleware
origins = ["http://localhost:8080", "http://127.0.0.1:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Pydantic Models ---
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ChatMessage(BaseModel):
    session_id: str
    user_email: str
    content: str
    sender: str

class ChatbotRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class ChatbotResponse(BaseModel):
    session_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    requires_input: bool = False
    input_question: Optional[str] = None

class HumanInputRequest(BaseModel):
    session_id: str
    response: str

# In-memory session storage for active chats
sessions: Dict[str, Dict[str, Any]] = {}

# --- Authentication Endpoints ---
@app.post("/auth/signup")
async def signup(user: UserCreate):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(user.password)
    user_data = user.model_dump()
    user_data["hashed_password"] = hashed_password
    del user_data["password"]
    users_collection.insert_one(user_data)
    return {"message": "User created successfully"}

@app.post("/auth/login")
async def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not pwd_context.verify(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return {"message": "Login successful", "user": {"name": db_user["name"], "email": db_user["email"]}}

# --- Chat Message Saving Endpoint ---
@app.post("/chats/messages")
async def save_chat_message(message: ChatMessage):
    message_data = message.model_dump()
    message_data["timestamp"] = datetime.utcnow()
    chats_collection.insert_one(message_data)
    return {"message": "Message saved successfully"}

# --- Background Crew Task ---
def run_crew_task(session_id: str, initial_prompt: str):
    try:
        def get_human_input_for_session(question: str) -> str:
            sessions[session_id]["pending_input"] = question
            sessions[session_id]["status"] = "awaiting_input"
            while sessions[session_id].get("human_response") is None:
                time.sleep(1)
            response = sessions[session_id].pop("human_response")
            sessions[session_id]["pending_input"] = None
            sessions[session_id]["status"] = "in_progress"
            return response

        human_input_tool.func = get_human_input_for_session
        setup_crew = create_setup_crew(initial_prompt)
        sessions[session_id]["status"] = "in_progress"
        trip_details_output = setup_crew.kickoff()
        trip_details = extract_json_from_response(trip_details_output.raw)
        sessions[session_id]["trip_details"] = trip_details
        sessions[session_id]["status"] = "setup_complete"
        
        result_object = invoke_agent(**trip_details)
        raw_result = result_object.raw if hasattr(result_object, 'raw') else str(result_object)

        # --- THIS IS THE FIX ---
        # Clean the raw markdown output to remove code fences
        cleaned_result = re.sub(r'^```markdown\n', '', raw_result)
        cleaned_result = re.sub(r'```$', '', cleaned_result)
        cleaned_result = cleaned_result.strip()
        # --- END OF FIX ---

        sessions[session_id]["result"] = cleaned_result # Store the cleaned result
        sessions[session_id]["status"] = "completed"
        
    except Exception as e:
        print(f"Error in background task for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)

# --- Chatbot Core Endpoints ---
@app.post("/chatbot/start", response_model=ChatbotResponse)
async def start_chatbot(request: ChatbotRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"status": "initializing", "initial_prompt": request.prompt}
    background_tasks.add_task(run_crew_task, session_id, request.prompt)
    return ChatbotResponse(session_id=session_id, status="in_progress", message="Chatbot processing started.")

@app.post("/chatbot/input", response_model=ChatbotResponse)
async def provide_human_input(request: HumanInputRequest):
    session_id = request.session_id
    if session_id not in sessions or sessions[session_id].get("status") != "awaiting_input":
        raise HTTPException(status_code=400, detail="Not awaiting input.")
    sessions[session_id]["human_response"] = request.response
    return ChatbotResponse(session_id=session_id, status="in_progress", message="Input received.")

@app.get("/chatbot/status/{session_id}", response_model=ChatbotResponse)
async def get_session_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    status = session.get("status", "error")
    response_data = {"session_id": session_id, "status": status, "message": f"Session status: {status}"}
    if status == "awaiting_input":
        response_data.update({"requires_input": True, "input_question": session.get("pending_input")})
    elif status == "completed":
        response_data["data"] = {"result": session.get("result")}
    elif status == "error":
        response_data["data"] = {"error": session.get("error")}
    return ChatbotResponse(**response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
