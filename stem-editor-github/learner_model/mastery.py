from sqlalchemy.orm import Session
from database import models


def update_mastery(
    db: Session,
    student_id: int,
    concept: str,
    success: bool
):
    record = db.query(models.StudentMastery).filter(
        models.StudentMastery.student_id == student_id,
        models.StudentMastery.concept == concept
    ).first()

    if not record:
        record = models.StudentMastery(
            student_id=student_id,
            concept=concept,
            mastery_score=0.5
        )
        db.add(record)

    # 简单渐进更新
    if success:
        record.mastery_score += 0.1
    else:
        record.mastery_score -= 0.08

    # 限制在 0 ~ 1
    record.mastery_score = max(
        0.0,
        min(1.0, record.mastery_score)
    )

    db.commit()
    db.refresh(record)

    return record