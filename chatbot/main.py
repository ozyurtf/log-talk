from uuid import uuid4
from pathlib import Path
from models import *
from process import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi import Header
from dotenv import load_dotenv
from chainlit.utils import mount_chainlit
from typing import Dict
import asyncio

app = FastAPI(title = "Drone Log API", description = "API for processing drone flight logs", version = "1.0.0")

executor = ThreadPoolExecutor(max_workers=4)

load_dotenv()

upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)
flight_data_store: Dict[str, Dict[str, dict]] = {}

app.add_middleware(CORSMiddleware,
                   allow_origins = ["http://localhost:3000", "http://localhost:8080", "*"], 
                   allow_credentials = True,
                   allow_methods = ["GET", "POST", "DELETE"],
                   allow_headers = ["*"])

async def process_file_background(file_id: str, file_path: str, user_id: str):
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(executor, read_data, str(file_path), ["GPS"])
        if user_id in flight_data_store and file_id in flight_data_store[user_id]:
            flight_data_store[user_id][file_id]["content"] = content
        
    except Exception as e:
        print(f"Error processing file {file_id}: {str(e)}")

@app.post("/api/files/{file_id}", response_model = FileReceiveResponse, status_code = 201, description = "Upload and process a drone flight log file")
async def receive_file(file_id: str, file: UploadFile = File(...), user_id: str = Header(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(('.bin', '.log')):
        raise HTTPException(status_code=400, detail="Only .bin and .log files are supported")
    
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 100MB")
    
    file_path = upload_dir / f"{file_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)    
        
        if user_id not in flight_data_store:
            flight_data_store[user_id] = {}
            
        file_data = {"file_id": file_id,
                     "file_path": str(file_path),
                     "filename": file.filename,
                     "content": ""}                 
        
        flight_data_store[user_id][file_id] = file_data
        if background_tasks:
            background_tasks.add_task(process_file_background, file_id, file_path, user_id)
        else:
            asyncio.create_task(process_file_background(file_id, file_path, user_id))
        
        return FileReceiveResponse(**file_data)
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code = 500, detail = f"Failed to upload file: {str(e)}")

@app.get("/api/files/{file_id}/status", description = "Get the status and summary of an uploaded file")
async def get_file_status(file_id: str, user_id: str = Header(...)):
    if user_id not in flight_data_store or file_id not in flight_data_store[user_id]:
        raise HTTPException(status_code = 404, detail="File not found")
    
    file_data = flight_data_store[user_id][file_id]
    
    status_summary = {"user_id": user_id,
                      "has_file": True,
                      "file_id": file_id,
                      "filename": file_data["filename"],
                      "content": file_data["content"]}    
    
    return status_summary

@app.get("/api/files/", description = "Get a list of all uploaded files")
async def list_files():
    files = []
    for user_id in flight_data_store.keys(): 
        user_data = flight_data_store[user_id]
        for file_id in user_data.keys():
            data = user_data[file_id]
            file_summary = {"file_id": file_id, 
                            "filename": data["filename"]}
            files.append(file_summary)
    return files

@app.post("/api/files/{file_id}/chat", description = "Ask questions about a specific uploaded file")
async def chat_with_file_data(request: ChatMessage, file_id: str, user_id: str = Header(...)):
    if user_id not in flight_data_store or file_id not in flight_data_store[user_id]:
        raise HTTPException(status_code = 404, detail="File not found")
    
    file_data = flight_data_store[user_id][file_id]
        
    prompt = f"""
    The user uploaded drone flight data from file: {file_data['filename']}
    The flight data has been processed.
    User question: {request.message}
    Answer the question based on the processed flight data: {file_data["content"]}
    """
    
    return {"response": f"Based on your flight data from {file_data['filename']}: {request.message}",
            "prompt": prompt,
            "filename": file_data['filename']}

@app.delete("/api/files/{file_id}", description = "Delete an uploaded file and its data")
async def delete_file(file_id: str, user_id: str = Header(...)):
    if user_id not in flight_data_store:
        raise HTTPException(status_code=404, detail="User not found")

    if file_id not in flight_data_store[user_id]: 
        raise HTTPException(status_code=404, detail="File not found")
        
    file_data = flight_data_store[user_id][file_id]
    file_path = Path(file_data['file_path'])
    if file_path.exists():
        file_path.unlink()
        
    del flight_data_store[user_id][file_id]
    return {"message": f"File {file_data['filename']} deleted successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# @app.lifespan("shutdown")
# async def shutdown_event():
#     executor.shutdown(wait=True)

# mount_chainlit(app=app, target="app.py", path="/chainlit")