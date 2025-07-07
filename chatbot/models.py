from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class FileReceiveResponse(BaseModel):
    file_id: str
    file_path: str
    filename: str
    
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's question about the flight data")

class ChatResponse(BaseModel):
    response: str
    file_id: str
    filename: str

class FileStatus(BaseModel):
    has_file: bool
    file_id: Optional[str] = None
    filename: Optional[str] = None
    summary: Optional[str] = None

class FileInfo(BaseModel):
    file_id: str
    filename: str
    uploaded_at: Optional[datetime] = None

class FileListResponse(BaseModel):
    files: List[FileInfo]
    count: int

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime

class DeleteFileResponse(BaseModel):
    message: str
    file_id: str
    
class VectorstoreUpdateRequest(BaseModel):
    content: str
    index_path: str
    
class VectorstoreQueryRequest(BaseModel):
    content: str
    index_path: str
