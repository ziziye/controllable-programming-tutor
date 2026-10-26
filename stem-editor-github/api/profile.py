from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.db import get_db
from database import models, schemas


router = APIRouter(
    prefix="/profile",
    tags=["Learner Profile"]
)


@router.post(
    "",
    response_model=schemas.StudentProfileResponse
)
def create_profile(
    profile: schemas.StudentProfileCreate,
    db: Session = Depends(get_db)
):
    student = db.query(
        models.Student
    ).filter(
        models.Student.id == profile.student_id
    ).first()

    if not student:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    existing = db.query(
        models.StudentProfile
    ).filter(
        models.StudentProfile.student_id == profile.student_id
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Profile already exists"
        )

    new_profile = models.StudentProfile(
        student_id=profile.student_id,
        personality=profile.personality,
        learning_preference=profile.learning_preference,
        challenge_tolerance=profile.challenge_tolerance
    )

    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)

    return new_profile


@router.get(
    "/{student_id}",
    response_model=schemas.StudentProfileResponse
)
def get_profile(
    student_id: int,
    db: Session = Depends(get_db)
):
    profile = db.query(
        models.StudentProfile
    ).filter(
        models.StudentProfile.student_id == student_id
    ).first()

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Profile not found"
        )

    return profile