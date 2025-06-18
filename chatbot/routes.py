import uuid
import os
from fastapi import FastAPI, UploadFile, File
from uuid import uuid4
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from models import *
from process import *

app = FastAPI()
upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)
flight_data_cache = {}
current_file_data = None
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])
    
@app.post("/api/files/", 
         response_model = FileUploadResponse,
         status_code = 200, 
         tags = ["Files"],
         summary = "Get uploaded files",
         description = "Receive the uploaded sfile from the UI")
async def receive_file(file: UploadFile = File(...)): 
    global current_file_data
    file_id = str(uuid4())
    file_path = upload_dir / f"{file_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)    
    gps_summary = generate_gps_summary(str(file_path))
    flight_data_cache[file_id] = gps_summary
    current_file_data = {"file_id": file_id,
                         "filename": file.filename,
                         "summary": gps_summary,
                         "message": f"The file {file.filename} was uploaded and processed."}
    return current_file_data
    
@app.get("/api/file-status")
async def get_file_status():
    if current_file_data:
        return {"has_file": True,
                "filename": current_file_data['filename'],
                "summary": str(current_file_data['summary'])}
    return {"has_file": False}    

    
@app.post("/api/chat/")
async def chat_with_processed_data(request: ChatRequest):
    global current_file_data
    
    if not current_file_data:
        return {"response": "No file uploaded. Please upload a file first."}
    
    prompt = f"""
              The user uploaded drone flight data. Here's the summary:
              {current_file_data['summary']}

              User question: {request.message}

              Answer the question based on the flight data above.
              """
    
    return {"response": f"Based on your flight data: {request.message}",
            "prompt": prompt,
            "flight_data": current_file_data['summary']}