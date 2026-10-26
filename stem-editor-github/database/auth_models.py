from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from database.db import Base


class StudentLoginCredential(Base):
    __tablename__ = "student_login_credentials"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        unique=True,
        nullable=False,
        index=True
    )
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TeacherAccount(Base):
    __tablename__ = "teacher_accounts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    role = Column(String, nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teacher_accounts.id"), nullable=True, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
