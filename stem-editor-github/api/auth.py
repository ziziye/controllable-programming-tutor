import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import models
from database import auth_models
from database.db import get_db


router = APIRouter(prefix="/auth", tags=["Authentication"])

PASSWORD_ITERATIONS = 240_000
SESSION_HOURS = 24


class StudentRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=6, max_length=128)


class StudentLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=6, max_length=128)


class TeacherRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=128)
    invite_code: str = Field(min_length=1, max_length=200)


class TeacherLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=128)


# Password / token helpers

def normalize_email(
    value: str
) -> str:
    return str(value).strip().lower()


def hash_password(
    password: str
) -> str:
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS
    )

    return (
        f"pbkdf2_sha256$"
        f"{PASSWORD_ITERATIONS}$"
        f"{salt.hex()}$"
        f"{digest.hex()}"
    )


def verify_password(
    password: str,
    stored_hash: str
) -> bool:
    try:
        algorithm, iteration_text, salt_hex, digest_hex = (
            stored_hash.split("$", 3)
        )

        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(
            iteration_text
        )

        salt = bytes.fromhex(
            salt_hex
        )

        expected = bytes.fromhex(
            digest_hex
        )

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations
        )

        return hmac.compare_digest(
            actual,
            expected
        )

    except (
        ValueError,
        TypeError
    ):
        return False


def hash_token(
    token: str
) -> str:
    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )


def create_session(
    db: Session,
    role: str,
    student_id: int | None = None,
    teacher_id: int | None = None
) -> tuple[str, auth_models.AuthSession]:
    raw_token = secrets.token_urlsafe(
        32
    )

    session = auth_models.AuthSession(
        token_hash=hash_token(
            raw_token
        ),
        role=role,
        student_id=student_id,
        teacher_id=teacher_id,
        expires_at=(
            utc_now()
            +
            timedelta(
                hours=SESSION_HOURS
            )
        )
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return (
        raw_token,
        session
    )


def extract_bearer_token(
    authorization: str | None
) -> str | None:
    if not authorization:
        return None

    prefix = "Bearer "

    if not authorization.startswith(
        prefix
    ):
        return None

    token = authorization[
        len(prefix):
    ].strip()

    return token or None


def resolve_session(
    db: Session,
    raw_token: str | None
):
    if not raw_token:
        return None

    session = db.query(
        auth_models.AuthSession
    ).filter(
        auth_models.AuthSession.token_hash
        ==
        hash_token(
            raw_token
        )
    ).first()

    if not session:
        return None

    if session.expires_at < utc_now():
        db.delete(session)
        db.commit()
        return None

    return session


def session_user_payload(
    db: Session,
    session
) -> dict:
    if session.role == "student":
        student = db.query(models.Student).filter(
            models.Student.id == session.student_id
        ).first()
        if not student:
            raise HTTPException(status_code=401, detail="Student account is no longer available.")
        return {
            "role": "student",
            "student_id": student.id,
            "display_name": student.name,
            "email": student.email
        }

    if session.role == "teacher":
        teacher = db.query(auth_models.TeacherAccount).filter(
            auth_models.TeacherAccount.id == session.teacher_id
        ).first()
        if not teacher:
            raise HTTPException(status_code=401, detail="Teacher account is no longer available.")
        return {
            "role": "teacher",
            "teacher_id": teacher.id,
            "display_name": teacher.name,
            "email": teacher.email
        }

    raise HTTPException(status_code=401, detail="Unknown account role.")


def require_session(
    authorization: str | None = Header(
        default=None
    ),
    db: Session = Depends(
        get_db
    )
):
    raw_token = extract_bearer_token(
        authorization
    )

    session = resolve_session(
        db,
        raw_token
    )

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required."
        )

    return session


# Student authentication

@router.post("/student/register")
def register_student(payload: StudentRegisterRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Please enter a valid email address.")

    existing = db.query(models.Student).filter(models.Student.email == email).first()
    if existing:
        existing_credential = db.query(auth_models.StudentLoginCredential).filter(
            auth_models.StudentLoginCredential.student_id == existing.id
        ).first()
        if existing_credential:
            raise HTTPException(status_code=409, detail="A student account with this email already exists.")
        student = existing
        student.name = payload.name.strip()
    else:
        student = models.Student(
            name=payload.name.strip(),
            email=email,
            learning_preference="guided"
        )
        db.add(student)
        db.flush()

    profile = db.query(models.StudentProfile).filter(
        models.StudentProfile.student_id == student.id
    ).first()
    if not profile:
        db.add(models.StudentProfile(
            student_id=student.id,
            personality="balanced",
            learning_preference="guided",
            challenge_tolerance=0.5,
            current_confusion=0.0,
            current_engagement=0.5
        ))

    db.add(auth_models.StudentLoginCredential(
        student_id=student.id,
        password_hash=hash_password(payload.password)
    ))
    db.commit()
    db.refresh(student)

    token, session = create_session(db=db, role="student", student_id=student.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": SESSION_HOURS,
        "user": session_user_payload(db, session)
    }


@router.post("/student/login")
def login_student(payload: StudentLoginRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    student = db.query(models.Student).filter(models.Student.email == email).first()
    if not student:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    credential = db.query(auth_models.StudentLoginCredential).filter(
        auth_models.StudentLoginCredential.student_id == student.id
    ).first()
    if not credential or not verify_password(payload.password, credential.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    profile = db.query(models.StudentProfile).filter(
        models.StudentProfile.student_id == student.id
    ).first()
    if not profile:
        db.add(models.StudentProfile(
            student_id=student.id,
            personality="balanced",
            learning_preference=student.learning_preference or "guided",
            challenge_tolerance=0.5,
            current_confusion=0.0,
            current_engagement=0.5
        ))
        db.commit()

    token, session = create_session(db=db, role="student", student_id=student.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": SESSION_HOURS,
        "user": session_user_payload(db, session)
    }


# Teacher authentication

@router.post("/teacher/register")
def register_teacher(payload: TeacherRegisterRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    if "@" not in email:
        raise HTTPException(status_code=422, detail="Please enter a valid email address.")

    configured_invite = (os.getenv("TEACHER_INVITE_CODE") or "").strip()
    if not configured_invite:
        raise HTTPException(
            status_code=503,
            detail="Teacher registration is not configured. Set TEACHER_INVITE_CODE in backend/.env."
        )
    if not hmac.compare_digest(payload.invite_code, configured_invite):
        raise HTTPException(status_code=403, detail="Invalid teacher invite code.")

    existing = db.query(auth_models.TeacherAccount).filter(
        auth_models.TeacherAccount.email == email
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="A teacher account with this email already exists.")

    teacher = auth_models.TeacherAccount(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password)
    )
    db.add(teacher)
    db.commit()
    db.refresh(teacher)

    token, session = create_session(db=db, role="teacher", teacher_id=teacher.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": SESSION_HOURS,
        "user": session_user_payload(db, session)
    }


@router.post("/teacher/login")
def login_teacher(payload: TeacherLoginRequest, db: Session = Depends(get_db)):
    email = normalize_email(payload.email)
    teacher = db.query(auth_models.TeacherAccount).filter(
        auth_models.TeacherAccount.email == email
    ).first()
    if not teacher or not verify_password(payload.password, teacher.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token, session = create_session(db=db, role="teacher", teacher_id=teacher.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_hours": SESSION_HOURS,
        "user": session_user_payload(db, session)
    }


@router.get("/me")
def get_me(session=Depends(require_session), db: Session = Depends(get_db)):
    return session_user_payload(db, session)


@router.post("/logout")
def logout(session=Depends(require_session), db: Session = Depends(get_db)):
    db.delete(session)
    db.commit()
    return {"ok": True}
