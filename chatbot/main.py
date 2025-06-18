## Ideas:
# Add authentication
# Store chat history 
# Add like and dislike buttons
# Process the data much faster 
# Allow multiple users to use the chatbot at the same time in a fast manner
# Create database (maybe relational database in Azure or AWS or vector database) to store data
# Visual (Multimodal) RAG 
# RAG for both webpage and file
# Incremental update to vector store for each file 
# Utilizing plots when answering questions
# OpenAI's embedding for RAG
# OpenAI's other services for different components
# LangChain's tools for parsing webpage
# Downloading the chat history 
# Downloading the plots 
# Structuring the answers in a structured data structure, clicking in a button, and sending it into a database
# Add command for "train yourself" that fine-tune the model in the background with the user feedbacks
# How to choose the vector database ? How to make that decision ? What are the pros and cons of different options ? 
# Consider mounting Chainlit as a FastAPI sub application (https://docs.chainlit.io/integrations/fastapi) 
# Consider using OpenAI's vector store (https://platform.openai.com/storage/vector_stores/vs_684a8cd5cdcc8191b2e0df58b23949be) (https://www.likeminds.community/blog/openai-assistants-vector-stores-and-file-storage-tool#:~:text=One%20of%20the%20critical%20components,vector%20representations%2C%20of%20text%20data.)
# Consider transforming the data and history of the chatbot coming from users and using it to fine-tune the OpenAI model (and use that model for chatbot specifically) (https://platform.openai.com/finetune)
# Learn more about FastAPI (https://www.youtube.com/watch?v=J_5pABCGVfc)
# Learn more about LangChain, and what to do with it. 
# Learn more about Chainlit, and what to do with it. 
# Different users can upload different files multiple times. Ensure that each user deals with the right file at the right time. Authentication might be one solution (https://www.youtube.com/watch?v=I11jbMOCY0c)
# Review the status codes and update (if necessary)
# List the user names, etc. in a database. 
# For file uploads specifically, you'll want to implement a file storage system that associates uploaded files with user identities. 
    # When User A uploads files, you store them with a reference to their user ID. When they return later, your application can query for all files associated with their user ID.
    # uploaded_files table:
    # - file_id (primary key)
    # - user_id (foreign key to users table)
    # - original_filename
    # - stored_filename (the actual file path on your server)
    # - upload_timestamp
    # - file_size
    # - file_type
    # - is_active (boolean to handle soft deletes)    
# Consider scalability and robustness of the pipeline as the data grows.
# Check mavlink document, find a better way to read and process the data. 
# Consider using threads, garbage collection, and other relevant methods during data reading and processing.
# Consider caching some information ? 
# For now, I am (fozyurt) the only user. If multiple users upload different files in different time frames, the current system won't handle that well. Fix that issue. 
# Check the library of "concurrent" in Python (e.g., from concurrent.futures import ThreadPoolExecutor) and see if it can be utilized. 
# Compare different vector databases (e.g., Chroma, Pinecone, FAISS) (https://benchmark.vectorview.ai/vectordbs.html) and choose the most suitable one.
# Consider preparing a dashboard that is automatically updated based on the stored data in the relational database in AWS/Azure and maybe vector store.
# Add try and except blocks to necessary places to avoid unexpected errors.
# Set the size of document chunks appropriately (process_url in process.py).
# Learn the impact of parameters of LLM (temperature, etc.) and Optimize them accordingly. 
# Prepare one LLM for text generation and another separate LLM for text retrieval because they have different roles and likely different parameters.
# Decide if I should use AsyncOpenAI or langchain's ChatOpenAI. 
# Consider batch processing.
# Consider replacing flight_data_store with a better option. Store the data in a database instead of in-memory.
# Try to understand every single line before submitting and make sure to know what they do.
# Use a stable, anonymized ID for each user (not an email or username) — for example, a UUID stored in your database.

from uuid import uuid4
from pathlib import Path
from models import *
from process import *
from typing import Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import Header

app = FastAPI(title = "Drone Log API", description = "API for processing drone flight logs", version = "1.0.0")

upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)
flight_data_store: Dict[str, Dict[str, dict]] = {}

app.add_middleware(CORSMiddleware,
                   allow_origins = ["http://localhost:3000", "http://localhost:8080", "*"], 
                   allow_credentials = True,
                   allow_methods = ["GET", "POST", "DELETE"],
                   allow_headers = ["*"])

@app.post("/api/files/{file_id}", response_model = FileReceiveResponse, status_code = 201, description = "Upload and process a drone flight log file")
async def receive_file(file_id: str, file: UploadFile = File(...), user_id: str = Header(...)):
    if not file.filename.endswith(('.bin', '.log')):
        raise HTTPException(status_code=400, detail="Only .bin and .log files are supported")
    
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 100MB")
    
    file_path = upload_dir / f"{file_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)    
            
        gps_summary = generate_gps_summary(str(file_path))
        
        file_data = {"file_id": file_id,
                     "filename": file.filename,
                     "file_path": str(file_path),
                     "summary": gps_summary,
                     "message": f"File {file.filename} uploaded and processed successfully"}
        
        if user_id not in flight_data_store:
            flight_data_store[user_id] = {}
        
        flight_data_store[user_id][file_id] = file_data
        return FileReceiveResponse(**file_data)
        
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code = 500, detail = f"Failed to process file: {str(e)}")

@app.get("/api/files/{file_id}/status", description = "Get the status and summary of an uploaded file")
async def get_file_status(file_id: str, user_id: str = Header(...)):
    if user_id not in flight_data_store or file_id not in flight_data_store[user_id]:
        raise HTTPException(status_code = 404, detail="File not found")
    file_data = flight_data_store[user_id][file_id]
    status_summary = {"user_id": user_id,
                      "has_file": True,
                      "file_id": file_id,
                      "filename": file_data['filename'],
                      "summary": file_data['summary']}    
    return status_summary

@app.get("/api/files/", description = "Get a list of all uploaded files")
async def list_files():
    files = []
    for user_id in flight_data_store.keys(): 
        user_data = flight_data_store[user_id]
        for file_id in user_data.keys():
            data = user_data[file_id]
            file_summary = {"file_id": file_id, "filename": data["filename"]}
            files.append(file_summary)
    return files

@app.post("/api/files/{file_id}/chat", description = "Ask questions about a specific uploaded file")
async def chat_with_file_data(request: ChatMessage, file_id: str, user_id: str = Header(...)):
    if user_id not in flight_data_store or file_id not in flight_data_store[user_id]:
        raise HTTPException(status_code = 404, detail="File not found")
    
    file_data = flight_data_store[user_id][file_id]
    
    prompt = f"""
    The user uploaded drone flight data from file: {file_data['filename']}
    
    Here's the flight data summary:
    {file_data['summary']}

    User question: {request.message}

    Answer the question based on the flight data above.
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