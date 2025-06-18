from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class FileReceiveResponse(BaseModel):
    file_id: str
    filename: str
    file_path: str
    summary: str
    message: str
    
class ChatMessage(BaseModel):
    """Simple chat message - file_id comes from URL path"""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question about the flight data")

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str
    file_id: str
    filename: str

class FileStatus(BaseModel):
    """Status of a specific file"""
    has_file: bool
    file_id: Optional[str] = None
    filename: Optional[str] = None
    summary: Optional[str] = None

class FileInfo(BaseModel):
    """Basic file information for listing"""
    file_id: str
    filename: str
    uploaded_at: Optional[datetime] = None

class FileListResponse(BaseModel):
    """Response for listing all files"""
    files: List[FileInfo]
    count: int

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    error_code: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime

class DeleteFileResponse(BaseModel):
    """Response after deleting a file"""
    message: str
    file_id: str