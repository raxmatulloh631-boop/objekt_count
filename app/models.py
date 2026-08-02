from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid
from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class CountSession(Base):
    __tablename__ = "count_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    target_object = Column(String(120), nullable=False, index=True)
    detected_count = Column(Integer, nullable=False, default=0)
    original_ai_count = Column(Integer)
    original_image_path = Column(Text, nullable=False)
    processed_image_path = Column(Text)
    model_name = Column(String(80))
    processing_time_ms = Column(Integer)
    average_confidence = Column(Float)
    status = Column(String(20), default="completed")
    permanent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime)
    note = Column(Text)

    objects = relationship("DetectedObject", back_populates="session", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(days=30)


class DetectedObject(Base):
    __tablename__ = "detected_objects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    count_session_id = Column(String(36), ForeignKey("count_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    label = Column(String(120), nullable=False)
    confidence = Column(Float)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    center_x = Column(Float)
    center_y = Column(Float)
    is_manual = Column(Boolean, default=False)
    is_excluded = Column(Boolean, default=False)

    session = relationship("CountSession", back_populates="objects")
