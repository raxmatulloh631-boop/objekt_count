from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DetectedObjectOut(BaseModel):
    id: str
    label: str
    confidence: Optional[float] = None
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    is_manual: bool = False
    is_excluded: bool = False

    class Config:
        from_attributes = True


class CountResult(BaseModel):
    session_id: str
    target_object: str
    detected_count: int
    original_ai_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    model_name: Optional[str] = None
    average_confidence: Optional[float] = None
    status: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    original_image_url: str
    processed_image_url: Optional[str] = None
    objects: List[DetectedObjectOut] = []
    warning: Optional[str] = None


class HistoryItem(BaseModel):
    session_id: str
    target_object: str
    detected_count: int
    created_at: datetime
    thumbnail_url: Optional[str] = None
    permanent: bool = False

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]


class MessageResponse(BaseModel):
    message: str
    success: bool = True
