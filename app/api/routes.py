from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil
from datetime import datetime

from app.database import get_db
from app.models import CountSession, DetectedObject
from app.schemas import CountResult, HistoryResponse, HistoryItem, MessageResponse, DetectedObjectOut
from app.services.ai_service import count_objects
from app.core.config import settings

router = APIRouter()


@router.post("/count", response_model=CountResult)
async def create_count(
    image: UploadFile = File(...),
    target_object: str = Form(...),
    db: Session = Depends(get_db),
):
    # Validate
    target_object = target_object.strip()
    if not target_object or len(target_object) < 2:
        raise HTTPException(status_code=400, detail="Obyekt nomi juda qisqa")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Faqat rasm fayllari qabul qilinadi")

    # Save original
    ext = Path(image.filename).suffix.lower() or ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"
    original_path = Path(settings.UPLOAD_DIR) / filename

    with open(original_path, "wb") as f:
        content = await image.read()
        if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Rasm hajmi {settings.MAX_IMAGE_SIZE_MB}MB dan oshmasligi kerak")
        f.write(content)

    try:
        # AI inference
        ai_result = count_objects(str(original_path), target_object)

        # Save to DB
        session = CountSession(
            target_object=target_object,
            detected_count=ai_result["count"],
            original_ai_count=ai_result["count"],
            original_image_path=str(original_path),
            processed_image_path=ai_result.get("processed_image_path"),
            model_name=ai_result.get("model_name"),
            processing_time_ms=ai_result.get("processing_time_ms"),
            average_confidence=ai_result.get("average_confidence"),
            status="completed",
        )
        db.add(session)
        db.flush()

        for det in ai_result["detections"]:
            obj = DetectedObject(
                count_session_id=session.id,
                label=det["label"],
                confidence=det["confidence"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                center_x=det["center_x"],
                center_y=det["center_y"],
            )
            db.add(obj)

        db.commit()
        db.refresh(session)

        # Build response
        objects_out = [
            DetectedObjectOut(
                id=obj.id,
                label=obj.label,
                confidence=obj.confidence,
                x1=obj.x1,
                y1=obj.y1,
                x2=obj.x2,
                y2=obj.y2,
                center_x=obj.center_x,
                center_y=obj.center_y,
                is_manual=obj.is_manual,
                is_excluded=obj.is_excluded,
            )
            for obj in session.objects
        ]

        return CountResult(
            session_id=session.id,
            target_object=session.target_object,
            detected_count=session.detected_count,
            original_ai_count=session.original_ai_count,
            processing_time_ms=session.processing_time_ms,
            model_name=session.model_name,
            average_confidence=session.average_confidence,
            status=session.status,
            created_at=session.created_at,
            expires_at=session.expires_at,
            original_image_url=f"/api/images/original/{session.id}",
            processed_image_url=f"/api/images/processed/{session.id}" if session.processed_image_path else None,
            objects=objects_out,
            warning=ai_result.get("warning"),
        )

    except Exception as e:
        # Cleanup on error
        if original_path.exists():
            original_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"AI tahlil xatosi: {str(e)}")


@router.get("/history", response_model=HistoryResponse)
def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(CountSession).order_by(CountSession.created_at.desc())

    if search:
        query = query.filter(CountSession.target_object.ilike(f"%{search}%"))

    total = query.count()
    items = query.offset((page - 1) * limit).limit(limit).all()

    return HistoryResponse(
        total=total,
        items=[
            HistoryItem(
                session_id=s.id,
                target_object=s.target_object,
                detected_count=s.detected_count,
                created_at=s.created_at,
                thumbnail_url=f"/api/images/processed/{s.id}" if s.processed_image_path else f"/api/images/original/{s.id}",
                permanent=s.permanent,
            )
            for s in items
        ],
    )


@router.get("/count/{session_id}", response_model=CountResult)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(CountSession).filter(CountSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session topilmadi")

    objects_out = [
        DetectedObjectOut(
            id=obj.id,
            label=obj.label,
            confidence=obj.confidence,
            x1=obj.x1,
            y1=obj.y1,
            x2=obj.x2,
            y2=obj.y2,
            center_x=obj.center_x,
            center_y=obj.center_y,
            is_manual=obj.is_manual,
            is_excluded=obj.is_excluded,
        )
        for obj in session.objects if not obj.is_excluded
    ]

    return CountResult(
        session_id=session.id,
        target_object=session.target_object,
        detected_count=session.detected_count,
        original_ai_count=session.original_ai_count,
        processing_time_ms=session.processing_time_ms,
        model_name=session.model_name,
        average_confidence=session.average_confidence,
        status=session.status,
        created_at=session.created_at,
        expires_at=session.expires_at,
        original_image_url=f"/api/images/original/{session.id}",
        processed_image_url=f"/api/images/processed/{session.id}" if session.processed_image_path else None,
        objects=objects_out,
    )


@router.delete("/count/{session_id}", response_model=MessageResponse)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(CountSession).filter(CountSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session topilmadi")

    # Delete files
    for path in [session.original_image_path, session.processed_image_path]:
        if path and Path(path).exists():
            Path(path).unlink(missing_ok=True)

    db.delete(session)
    db.commit()
    return MessageResponse(message="O'chirildi", success=True)


@router.get("/images/original/{session_id}")
def get_original_image(session_id: str, db: Session = Depends(get_db)):
    session = db.query(CountSession).filter(CountSession.id == session_id).first()
    if not session or not Path(session.original_image_path).exists():
        raise HTTPException(status_code=404, detail="Rasm topilmadi")
    return FileResponse(session.original_image_path)


@router.get("/images/processed/{session_id}")
def get_processed_image(session_id: str, db: Session = Depends(get_db)):
    session = db.query(CountSession).filter(CountSession.id == session_id).first()
    if not session or not session.processed_image_path or not Path(session.processed_image_path).exists():
        raise HTTPException(status_code=404, detail="Rasm topilmadi")
    return FileResponse(session.processed_image_path)


@router.get("/health")
def health():
    return {"status": "ok", "service": "AI Object Counter"}
