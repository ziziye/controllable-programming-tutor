from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# Student

class StudentCreate(BaseModel):
    name: str
    email: str
    learning_preference: Optional[str] = None


class StudentResponse(StudentCreate):
    id: int

    model_config = ConfigDict(
        from_attributes=True
    )


# Course

class CourseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    programming_language: str = "Python"


class CourseResponse(CourseCreate):
    id: int

    model_config = ConfigDict(
        from_attributes=True
    )


# Assignment

class AssignmentBase(BaseModel):
    title: str
    description: str

    starter_code: Optional[str] = None

    # Backward-compatible field used by the current tutor pipeline.
    learning_objective: Optional[str] = None

    # Normalized assignment representation.
    function_name: Optional[str] = None
    language: str = "Python"
    difficulty: Optional[str] = None

    question_type: str = "coding_task"

    learning_objectives: list[str] = Field(
        default_factory=list
    )

    concepts: list[str] = Field(
        default_factory=list
    )

    constraints: list[str] = Field(
        default_factory=list
    )

    visible_examples: list[dict[str, Any]] = Field(
        default_factory=list
    )

    max_hint_level: int = Field(
        default=2,
        ge=0
    )

    course_id: int


class AssignmentCreate(AssignmentBase):
    # Internal teacher/evaluation fields.
    hidden_tests: list[dict[str, Any]] = Field(
        default_factory=list
    )

    reference_solution: Optional[str] = None

    # Provenance / quality-control fields.
    source: str = "teacher"
    source_url: Optional[str] = None

    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    is_validated: bool = False


class AssignmentUpdate(BaseModel):
    """
    Partial update schema.

    Every field is optional so PATCH can update only the fields
    that were explicitly supplied by the teacher.
    """

    title: Optional[str] = None
    description: Optional[str] = None
    starter_code: Optional[str] = None
    learning_objective: Optional[str] = None

    function_name: Optional[str] = None
    language: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: Optional[str] = None

    learning_objectives: Optional[list[str]] = None
    concepts: Optional[list[str]] = None
    constraints: Optional[list[str]] = None

    visible_examples: Optional[list[dict[str, Any]]] = None
    hidden_tests: Optional[list[dict[str, Any]]] = None

    reference_solution: Optional[str] = None

    source: Optional[str] = None
    source_url: Optional[str] = None

    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    is_validated: Optional[bool] = None

    max_hint_level: Optional[int] = Field(
        default=None,
        ge=0
    )

    course_id: Optional[int] = None


class AssignmentTeacherResponse(AssignmentCreate):
    id: int

    model_config = ConfigDict(
        from_attributes=True
    )


class AssignmentStudentResponse(AssignmentBase):
    """
    Student-safe assignment representation.

    Deliberately excludes:
    - hidden_tests
    - reference_solution
    - quality-control/internal provenance fields
    """

    id: int

    model_config = ConfigDict(
        from_attributes=True
    )


# Backward-compatibility alias.
AssignmentResponse = AssignmentTeacherResponse


# Question Bank

class QuestionBankItemBase(BaseModel):
    title: str
    description: str

    starter_code: Optional[str] = None
    learning_objective: Optional[str] = None

    function_name: Optional[str] = None
    language: str = "Python"
    difficulty: Optional[str] = None
    question_type: str = "coding_task"

    learning_objectives: list[str] = Field(
        default_factory=list
    )

    concepts: list[str] = Field(
        default_factory=list
    )

    constraints: list[str] = Field(
        default_factory=list
    )

    visible_examples: list[dict[str, Any]] = Field(
        default_factory=list
    )

    hidden_tests: list[dict[str, Any]] = Field(
        default_factory=list
    )

    reference_solution: Optional[str] = None

    source: str = "teacher"
    source_url: Optional[str] = None
    source_external_id: Optional[str] = None

    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    is_validated: bool = False

    max_hint_level: int = Field(
        default=2,
        ge=0
    )


class QuestionBankItemCreate(
    QuestionBankItemBase
):
    pass


class QuestionBankItemUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

    starter_code: Optional[str] = None
    learning_objective: Optional[str] = None

    function_name: Optional[str] = None
    language: Optional[str] = None
    difficulty: Optional[str] = None
    question_type: Optional[str] = None

    learning_objectives: Optional[list[str]] = None
    concepts: Optional[list[str]] = None
    constraints: Optional[list[str]] = None

    visible_examples: Optional[
        list[dict[str, Any]]
    ] = None

    hidden_tests: Optional[
        list[dict[str, Any]]
    ] = None

    reference_solution: Optional[str] = None

    source: Optional[str] = None
    source_url: Optional[str] = None
    source_external_id: Optional[str] = None

    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    is_validated: Optional[bool] = None

    max_hint_level: Optional[int] = Field(
        default=None,
        ge=0
    )


class QuestionBankItemResponse(
    QuestionBankItemBase
):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class QuestionBankPublishRequest(BaseModel):
    """
    Publish/copy a question-bank item into a course Assignment.

    title can be overridden during publication if the teacher wants
    to customize the course-facing version.
    """

    course_id: int
    title: Optional[str] = None
    max_hint_level: Optional[int] = Field(
        default=None,
        ge=0
    )


# Submission

class SubmissionCreate(BaseModel):
    student_id: int
    assignment_id: int
    code: str


class SubmissionGradeUpdate(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    teacher_feedback: Optional[str] = None


class SubmissionResponse(SubmissionCreate):
    id: int
    status: str
    score: Optional[float] = None
    teacher_feedback: Optional[str] = None
    graded_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


# Student Profile

class StudentProfileCreate(BaseModel):
    student_id: int
    personality: str = "balanced"
    learning_preference: str = "guided"
    challenge_tolerance: float = 0.5


class StudentProfileResponse(StudentProfileCreate):
    id: int
    current_confusion: float
    current_engagement: float

    model_config = ConfigDict(
        from_attributes=True
    )


# Mastery / Misconception

class MasteryResponse(BaseModel):
    concept: str
    mastery_score: float

    model_config = ConfigDict(
        from_attributes=True
    )


class MisconceptionResponse(BaseModel):
    misconception: str
    occurrence_count: int
    resolved: int

    model_config = ConfigDict(
        from_attributes=True
    )
