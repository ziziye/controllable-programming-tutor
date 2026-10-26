from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from database.db import get_db
from database import models

from agents.teacher_agent import TeacherAgent


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


teacher_agent = TeacherAgent()


# Helpers

def _number(value, default=0.0):
    """
    Convert database numeric values safely.

    Old rows can contain NULL even when a model currently has a
    default value, so analytics should never assume a value exists.
    """
    if value is None:
        return float(default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value, default=0):
    if value is None:
        return int(default)

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _profile_payload(profile):
    if not profile:
        return {
            "personality": None,
            "learning_preference": None,
            "challenge_tolerance": 0.5,
            "confusion_level": 0.0,
            "engagement": 0.0
        }

    return {
        "personality":
            getattr(
                profile,
                "personality",
                None
            ),

        "learning_preference":
            getattr(
                profile,
                "learning_preference",
                None
            ),

        "challenge_tolerance":
            _number(
                getattr(
                    profile,
                    "challenge_tolerance",
                    0.5
                ),
                0.5
            ),

        "confusion_level":
            _number(
                getattr(
                    profile,
                    "current_confusion",
                    0.0
                ),
                0.0
            ),

        "engagement":
            _number(
                getattr(
                    profile,
                    "current_engagement",
                    0.0
                ),
                0.0
            )
    }


def _fallback_teacher_recommendation(
    misconception_stats,
    total_students
):
    """
    Class analytics should still render even if TeacherAgent fails.
    """

    if total_students <= 0:
        return {
            "priority": "low",
            "summary":
                "No learner data are currently available.",
            "recommendation":
                "Collect learner interactions before generating a class-level intervention."
        }

    if not misconception_stats:
        return {
            "priority": "low",
            "summary":
                "No recurring misconceptions are currently recorded.",
            "recommendation":
                "Continue monitoring class mastery, confusion, and engagement."
        }

    top = misconception_stats[0]

    affected_students = _integer(
        top.get(
            "affected_students"
        ),
        0
    )

    ratio = (
        affected_students
        /
        total_students
        if total_students
        else 0
    )

    if ratio >= 0.5:
        priority = "high"
    elif ratio >= 0.25:
        priority = "medium"
    else:
        priority = "low"

    misconception = (
        top.get(
            "misconception"
        )
        or
        "the most frequent misconception"
    )

    return {
        "priority":
            priority,

        "summary":
            (
                f"{affected_students} of {total_students} tracked learners "
                f"show evidence related to '{misconception}'."
            ),

        "recommendation":
            (
                "Consider a targeted review activity and then reassess "
                "mastery and confusion for the affected concept."
            )
    }


# All Students

@router.get("/students")
def get_all_students(
    db: Session = Depends(get_db)
):
    students = (
        db.query(
            models.Student
        )
        .order_by(
            models.Student.id.asc()
        )
        .all()
    )

    result = []

    for student in students:

        profile = (
            db.query(
                models.StudentProfile
            )
            .filter(
                models.StudentProfile.student_id
                ==
                student.id
            )
            .first()
        )

        result.append({
            "id":
                student.id,

            "name":
                student.name,

            "email":
                student.email,

            "profile":
                _profile_payload(
                    profile
                )
        })

    return result


# Individual Student Analytics

@router.get("/students/{student_id}")
def get_student_analytics(
    student_id: int,
    db: Session = Depends(get_db)
):

    student = (
        db.query(
            models.Student
        )
        .filter(
            models.Student.id
            ==
            student_id
        )
        .first()
    )

    if not student:
        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    profile = (
        db.query(
            models.StudentProfile
        )
        .filter(
            models.StudentProfile.student_id
            ==
            student_id
        )
        .first()
    )

    mastery_records = (
        db.query(
            models.StudentMastery
        )
        .filter(
            models.StudentMastery.student_id
            ==
            student_id
        )
        .all()
    )

    misconception_records = (
        db.query(
            models.MisconceptionRecord
        )
        .filter(
            models.MisconceptionRecord.student_id
            ==
            student_id
        )
        .all()
    )

    # Use a left join so a historical/orphaned submission cannot
    # crash the analytics page simply because its assignment was
    # removed or migrated.
    submission_rows = (
        db.query(
            models.Submission,
            models.Assignment.title
        )
        .outerjoin(
            models.Assignment,
            models.Assignment.id
            ==
            models.Submission.assignment_id
        )
        .filter(
            models.Submission.student_id
            ==
            student_id
        )
        .order_by(
            models.Submission.created_at.desc()
        )
        .all()
    )

    submission_results = []

    for submission, assignment_title in submission_rows:

        submission_results.append({
            "id":
                submission.id,

            "student_id":
                submission.student_id,

            "assignment_id":
                submission.assignment_id,

            "assignment_title":
                assignment_title
                or
                "Unknown Assignment",

            "code":
                submission.code,

            "status":
                getattr(
                    submission,
                    "status",
                    "submitted"
                )
                or
                "submitted",

            "score":
                getattr(
                    submission,
                    "score",
                    None
                ),

            "teacher_feedback":
                getattr(
                    submission,
                    "teacher_feedback",
                    None
                ),

            "graded_at":
                getattr(
                    submission,
                    "graded_at",
                    None
                ),

            "created_at":
                getattr(
                    submission,
                    "created_at",
                    None
                )
        })

    mastery = [
        {
            "concept":
                record.concept,

            "score":
                _number(
                    record.mastery_score,
                    0.0
                )
        }
        for record
        in mastery_records
    ]

    misconceptions = [
        {
            "misconception":
                record.misconception,

            "occurrence_count":
                _integer(
                    record.occurrence_count,
                    0
                ),

            "resolved":
                bool(
                    record.resolved
                )
        }
        for record
        in misconception_records
    ]

    return {
        "student": {
            "id":
                student.id,

            "name":
                student.name,

            "email":
                student.email
        },

        "profile":
            _profile_payload(
                profile
            ),

        "mastery":
            mastery,

        "misconceptions":
            misconceptions,

        "submissions":
            submission_results
    }


# Class-level Analytics

@router.get("/class")
def get_class_analytics(
    db: Session = Depends(get_db)
):

    students = (
        db.query(
            models.Student
        )
        .all()
    )

    total_students = len(
        students
    )

    profiles = (
        db.query(
            models.StudentProfile
        )
        .all()
    )

    # Historical rows can contain NULL.  Convert every value before
    # calculating an average so one incomplete profile cannot break
    # the whole class dashboard.
    confusion_values = [
        _number(
            getattr(
                profile,
                "current_confusion",
                0.0
            ),
            0.0
        )
        for profile
        in profiles
    ]

    engagement_values = [
        _number(
            getattr(
                profile,
                "current_engagement",
                0.0
            ),
            0.0
        )
        for profile
        in profiles
    ]

    average_confusion = (
        sum(
            confusion_values
        )
        /
        len(
            confusion_values
        )
        if confusion_values
        else 0.0
    )

    average_engagement = (
        sum(
            engagement_values
        )
        /
        len(
            engagement_values
        )
        if engagement_values
        else 0.0
    )

    # Misconceptions

    misconception_records = (
        db.query(
            models.MisconceptionRecord
        )
        .all()
    )

    misconception_map = {}

    for record in misconception_records:

        name = (
            record.misconception
            or
            "unspecified"
        )

        if name not in misconception_map:
            misconception_map[name] = {
                "students":
                    set(),

                "total_occurrences":
                    0
            }

        misconception_map[
            name
        ]["students"].add(
            record.student_id
        )

        misconception_map[
            name
        ]["total_occurrences"] += (
            _integer(
                record.occurrence_count,
                0
            )
        )

    misconception_stats = []

    for name, data in misconception_map.items():

        affected_students = len(
            data["students"]
        )

        affected_ratio = (
            affected_students
            /
            total_students
            if total_students > 0
            else 0.0
        )

        misconception_stats.append({
            "misconception":
                name,

            "affected_students":
                affected_students,

            "total_occurrences":
                data[
                    "total_occurrences"
                ],

            "affected_ratio":
                round(
                    affected_ratio,
                    3
                )
        })

    misconception_stats.sort(
        key=lambda item:
            item[
                "affected_students"
            ],
        reverse=True
    )

    # Mastery

    mastery_records = (
        db.query(
            models.StudentMastery
        )
        .all()
    )

    mastery_map = {}

    for record in mastery_records:

        concept = (
            record.concept
            or
            "unspecified"
        )

        mastery_map.setdefault(
            concept,
            []
        ).append(
            _number(
                record.mastery_score,
                0.0
            )
        )

    mastery_stats = []

    for concept, scores in mastery_map.items():

        average_score = (
            sum(scores)
            /
            len(scores)
            if scores
            else 0.0
        )

        mastery_stats.append({
            "concept":
                concept,

            "average_mastery":
                round(
                    average_score,
                    3
                ),

            "student_count":
                len(scores)
        })

    mastery_stats.sort(
        key=lambda item:
            item[
                "average_mastery"
            ]
    )

    # Submissions

    total_submissions = (
        db.query(
            models.Submission
        )
        .count()
    )

    # Teacher Agent

    try:

        teacher_recommendation = (
            teacher_agent
            .generate_recommendation(
                misconception_stats=
                    misconception_stats,

                total_students=
                    total_students
            )
        )

        # Defend the UI against a malformed/empty agent response.
        if not isinstance(
            teacher_recommendation,
            dict
        ):
            raise ValueError(
                "TeacherAgent returned a non-dict recommendation."
            )

        teacher_recommendation = {
            "priority":
                teacher_recommendation.get(
                    "priority",
                    "low"
                ),

            "summary":
                teacher_recommendation.get(
                    "summary",
                    "Class analytics are available."
                ),

            "recommendation":
                teacher_recommendation.get(
                    "recommendation",
                    "Continue monitoring learner progress."
                )
        }

    except Exception as error:

        print(
            "[analytics/class] "
            f"TeacherAgent fallback: {error}"
        )

        teacher_recommendation = (
            _fallback_teacher_recommendation(
                misconception_stats=
                    misconception_stats,

                total_students=
                    total_students
            )
        )

    return {
        "class_summary": {
            "total_students":
                total_students,

            "average_confusion":
                round(
                    average_confusion,
                    3
                ),

            "average_engagement":
                round(
                    average_engagement,
                    3
                ),

            "total_submissions":
                total_submissions
        },

        "misconceptions":
            misconception_stats,

        "mastery":
            mastery_stats,

        "teacher_recommendation":
            teacher_recommendation
    }
