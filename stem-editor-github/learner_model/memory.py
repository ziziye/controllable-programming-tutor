from sqlalchemy.orm import Session
from database import models


def update_misconception_memory(
    db: Session,
    student_id: int,
    misconception: str | None
):
    if not misconception:
        return None

    record = db.query(models.MisconceptionRecord).filter(
        models.MisconceptionRecord.student_id == student_id,
        models.MisconceptionRecord.misconception == misconception
    ).first()

    if record:
        record.occurrence_count += 1
    else:
        record = models.MisconceptionRecord(
            student_id=student_id,
            misconception=misconception,
            occurrence_count=1,
            resolved=0
        )
        db.add(record)

    db.commit()
    db.refresh(record)

    return record