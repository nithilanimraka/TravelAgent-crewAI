import os
from dotenv import load_dotenv
from functools import lru_cache
from crewai import LLM, Agent, Task, Crew, Process
from datetime import datetime
import requests
import json
from crewai.tools import tool
from crewai_tools import SerperDevTool
from IPython.display import Markdown, display
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from passlib.context import CryptContext # For hashing passwords

# Import the collections from your database module
from database import users_collection

# Load environment variables
load_dotenv()

# Set environment variables from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI1_API_KEY = os.getenv("GEMINI1_API_KEY")
GEMINIPRO_API_KEY = os.getenv("GEMINIPRO_API_KEY")
OPENROUTER_API_KEY3 = os.getenv("OPENROUTER_API_KEY3")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

print("API Keys loaded successfully.")

# (Your LLM initializations and tool definitions remain the same)
@lru_cache(maxsize=1)
def initialize_llm():
    return LLM(
        model="openrouter/z-ai/glm-4.5-air:free",
        api_key=OPENROUTER_API_KEY3,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        temperature=0.4,
    )

# ... (other LLM initializations, tools, and helper functions are unchanged) ...
# Your existing travel_chatbot functions (create_setup_crew, invoke_agent, etc.)
# should be available here. For this example, I'll assume they are in scope.
# Make sure to import them correctly as you did before.
from travel_chatbot import (
    create_setup_crew,
    invoke_agent,
    extract_json_from_response,
    human_input_tool,
)


app = FastAPI(title="Travel Chatbot API")

# Add CORSMiddleware
origins = ["http://localhost:8080", "http://127.0.0.1:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- NEW: Pydantic Models for User Auth ---
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserInDB(UserCreate):
    hashed_password: str

 # --- NEW: Pydantic Model for Login ---
class UserLogin(BaseModel):
    email: str
    password: str   

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Pydantic Models
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

 # --- NEW: Signup Endpoint ---
@app.post("/auth/signup")
async def signup(user: UserCreate):
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = pwd_context.hash(user.password)
    
    # Create user document
    user_data = user.dict()
    user_data["hashed_password"] = hashed_password
    del user_data["password"] # Do not store plain password
    
    # Insert new user into the database
    users_collection.insert_one(user_data)
    
    return {"message": "User created successfully"}

 # --- NEW: Login Endpoint ---
@app.post("/auth/login")
async def login(user: UserLogin):
    # Find user in the database
    db_user = users_collection.find_one({"email": user.email})
    
    # Check if user exists and password is correct
    if not db_user or not pwd_context.verify(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # On successful login, return user info (excluding password)
    return {
        "message": "Login successful",
        "user": {
            "name": db_user["name"],
            "email": db_user["email"],
        }
    }      

# This function will run in the background
def run_crew_task(session_id: str, initial_prompt: str):
    """Runs the full crew process, from setup to final result."""
    try:
        # --- This is the key change for handling human input ---
        # We define a function that will be called by our tool
        def get_human_input_for_session(question: str) -> str:
            sessions[session_id]["pending_input"] = question
            sessions[session_id]["status"] = "awaiting_input"
            # This is a blocking part, but it's in a background thread, so it's okay.
            # The frontend will poll and see the "awaiting_input" status.
            while sessions[session_id].get("human_response") is None:
                time.sleep(1) # prevent busy-waiting
            
            response = sessions[session_id]["human_response"]
            # Clear the response so it's not used again
            sessions[session_id]["human_response"] = None 
            sessions[session_id]["pending_input"] = None
            sessions[session_id]["status"] = "in_progress" # Back to processing
            return response

        # Monkey-patch the function that the tool executes
        # The tool object itself remains a valid `BaseTool` instance
        human_input_tool.func = get_human_input_for_session
        
        # Now, create the crew with the patched tool
        setup_crew = create_setup_crew(initial_prompt)
        
        sessions[session_id]["status"] = "in_progress"
        trip_details_output = setup_crew.kickoff()
        
        trip_details_json_str = trip_details_output.raw
        trip_details = extract_json_from_response(trip_details_json_str)
        
        sessions[session_id]["trip_details"] = trip_details
        sessions[session_id]["status"] = "setup_complete"
        
        # Run the main agent
        result_object = invoke_agent(
            location=trip_details['location'],
            interests=trip_details['interests'],
            budget=trip_details['budget'],
            num_people=trip_details['num_people'],
            travel_dates=trip_details['travel_dates'],
            preferred_currency=trip_details['preferred_currency']
        )
        
        # Store the raw string in the session
        sessions[session_id]["result"] = result_object.raw if hasattr(result_object, 'raw') else str(result_object)
        sessions[session_id]["status"] = "completed"
        
    except Exception as e:
        print(f"Error in background task for session {session_id}: {e}")
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)


@app.post("/chatbot/start", response_model=ChatbotResponse)
async def start_chatbot(request: ChatbotRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "status": "initializing",
        "initial_prompt": request.prompt,
    }
    
    # Add the long-running agent task to the background
    background_tasks.add_task(run_crew_task, session_id, request.prompt)
    
    # Immediately return a response to the client
    return ChatbotResponse(
        session_id=session_id,
        status="in_progress",
        message="Chatbot processing started. Please wait.",
    )


@app.post("/chatbot/input", response_model=ChatbotResponse)
async def provide_human_input(request: HumanInputRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    if session.get("status") != "awaiting_input":
        raise HTTPException(status_code=400, detail="Not awaiting input.")

    # Store the user's response where the background thread can find it
    session["human_response"] = request.response
    
    return ChatbotResponse(
        session_id=session_id,
        status="in_progress",
        message="Input received. Resuming process.",
    )


@app.get("/chatbot/status/{session_id}", response_model=ChatbotResponse)
async def get_session_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    status = session.get("status", "error")
    
    response_data = {
        "session_id": session_id,
        "status": status,
        "message": f"Session status: {status}",
        "requires_input": False,
    }

    if status == "awaiting_input":
        response_data["requires_input"] = True
        response_data["input_question"] = session.get("pending_input")
    elif status == "completed":
        response_data["data"] = {"result": session.get("result")}
    elif status == "error":
        response_data["data"] = {"error": session.get("error")}
        
    return ChatbotResponse(**response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)