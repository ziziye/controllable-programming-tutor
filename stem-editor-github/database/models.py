from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    DateTime,
    Boolean,
    JSON
)

from sqlalchemy.orm import relationship

from database.db import Base


# Student

class Student(Base):

    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String,
        nullable=False
    )

    email = Column(
        String,
        unique=True,
        nullable=False,
        index=True
    )

    learning_preference = Column(
        String,
        nullable=True
    )

    personality_profile = Column(
        String,
        nullable=True
    )


# Course

class Course(Base):

    __tablename__ = "courses"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    title = Column(
        String,
        nullable=False
    )

    description = Column(
        Text,
        nullable=True
    )

    programming_language = Column(
        String,
        default="Python"
    )

    assignments = relationship(
        "Assignment",
        back_populates="course",
        cascade="all, delete-orphan"
    )


# Assignment
#
# A single normalized representation is used for both
# teacher-authored tasks and imported/question-bank tasks.
# Existing fields are retained for backward compatibility.

class Assignment(Base):

    __tablename__ = "assignments"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    title = Column(
        String,
        nullable=False
    )

    description = Column(
        Text,
        nullable=True
    )

    starter_code = Column(
        Text,
        nullable=True
    )

    # Keep the original single-objective field because the
    # current tutor pipeline already reads assignment.learning_objective.
    learning_objective = Column(
        Text,
        nullable=True
    )

    # New normalized assignment metadata.
    function_name = Column(
        String,
        nullable=True
    )

    language = Column(
        String,
        default="Python",
        nullable=False
    )

    difficulty = Column(
        String,
        nullable=True
    )

    question_type = Column(
        String,
        default="coding_task",
        nullable=False,
        index=True
    )

    learning_objectives = Column(
        JSON,
        default=list,
        nullable=False
    )

    concepts = Column(
        JSON,
        default=list,
        nullable=False
    )

    constraints = Column(
        JSON,
        default=list,
        nullable=False
    )

    visible_examples = Column(
        JSON,
        default=list,
        nullable=False
    )

    # Internal-only evaluation data. These fields must not be
    # returned by student-facing endpoints.
    hidden_tests = Column(
        JSON,
        default=list,
        nullable=False
    )

    reference_solution = Column(
        Text,
        nullable=True
    )

    # Question-bank provenance / quality-control metadata.
    source = Column(
        String,
        default="teacher",
        nullable=False
    )

    source_url = Column(
        Text,
        nullable=True
    )

    quality_score = Column(
        Float,
        nullable=True
    )

    is_validated = Column(
        Boolean,
        default=False,
        nullable=False
    )

    max_hint_level = Column(
        Integer,
        default=2
    )

    course_id = Column(
        Integer,
        ForeignKey("courses.id"),
        nullable=False
    )

    course = relationship(
        "Course",
        back_populates="assignments"
    )


# Question Bank Item
#
# A reusable programming problem stored independently from any course.
# A teacher can later publish/copy one of these items into Assignment.

class QuestionBankItem(Base):

    __tablename__ = "question_bank_items"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    title = Column(
        String,
        nullable=False,
        index=True
    )

    description = Column(
        Text,
        nullable=False
    )

    starter_code = Column(
        Text,
        nullable=True
    )

    learning_objective = Column(
        Text,
        nullable=True
    )

    function_name = Column(
        String,
        nullable=True,
        index=True
    )

    language = Column(
        String,
        default="Python",
        nullable=False,
        index=True
    )

    difficulty = Column(
        String,
        nullable=True,
        index=True
    )

    question_type = Column(
        String,
        default="coding_task",
        nullable=False,
        index=True
    )

    learning_objectives = Column(
        JSON,
        default=list,
        nullable=False
    )

    concepts = Column(
        JSON,
        default=list,
        nullable=False
    )

    constraints = Column(
        JSON,
        default=list,
        nullable=False
    )

    visible_examples = Column(
        JSON,
        default=list,
        nullable=False
    )

    hidden_tests = Column(
        JSON,
        default=list,
        nullable=False
    )

    reference_solution = Column(
        Text,
        nullable=True
    )

    source = Column(
        String,
        default="teacher",
        nullable=False,
        index=True
    )

    source_url = Column(
        Text,
        nullable=True
    )

    # Optional ID supplied by an external dataset/source.
    source_external_id = Column(
        String,
        nullable=True,
        index=True
    )

    quality_score = Column(
        Float,
        nullable=True
    )

    is_validated = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True
    )

    max_hint_level = Column(
        Integer,
        default=2,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


# Submission

class Submission(Base):

    __tablename__ = "submissions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False
    )

    assignment_id = Column(
        Integer,
        ForeignKey("assignments.id"),
        nullable=False
    )

    code = Column(
        Text,
        nullable=False
    )

    status = Column(
        String,
        default="submitted"
    )

    score = Column(
        Float,
        nullable=True
    )

    teacher_feedback = Column(
        Text,
        nullable=True
    )

    graded_at = Column(
        DateTime,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# Stable Student Profile

class StudentProfile(Base):

    __tablename__ = "student_profiles"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        unique=True,
        nullable=False
    )

    personality = Column(
        String,
        default="balanced"
    )

    learning_preference = Column(
        String,
        default="guided"
    )

    challenge_tolerance = Column(
        Float,
        default=0.5
    )

    current_confusion = Column(
        Float,
        default=0.0
    )

    current_engagement = Column(
        Float,
        default=0.5
    )


# Student Mastery

class StudentMastery(Base):

    __tablename__ = "student_mastery"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False
    )

    concept = Column(
        String,
        nullable=False
    )

    mastery_score = Column(
        Float,
        default=0.0
    )


# Misconception Memory

class MisconceptionRecord(Base):

    __tablename__ = "misconception_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False
    )

    misconception = Column(
        String,
        nullable=False
    )

    occurrence_count = Column(
        Integer,
        default=1
    )

    resolved = Column(
        Integer,
        default=0
    )


# Dynamic Learner Profile

class StudentDynamicProfile(Base):

    __tablename__ = "student_dynamic_profiles"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        unique=True,
        nullable=False,
        index=True
    )

    persistence = Column(
        Float,
        default=0.5
    )

    help_seeking_tendency = Column(
        Float,
        default=0.5
    )

    directness_preference = Column(
        Float,
        default=0.5
    )

    challenge_tolerance = Column(
        Float,
        default=0.5
    )

    reflective_tendency = Column(
        Float,
        default=0.5
    )

    learning_preference = Column(
        String,
        default="guided"
    )

    # Existing database field name retained for compatibility.
    # We now interpret this as a non-clinical learning-style summary.
    personality_tendency = Column(
        String,
        default="balanced"
    )

    profile_confidence = Column(
        Float,
        default=0.0
    )

    evidence_count = Column(
        Integer,
        default=0
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


# Learning Episode
#
# Operational tutoring memory.

class LearningEpisode(Base):

    __tablename__ = "learning_episodes"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False,
        index=True
    )

    assignment_id = Column(
        Integer,
        ForeignKey("assignments.id"),
        nullable=False,
        index=True
    )

    code = Column(
        Text,
        nullable=True
    )

    pause_time = Column(
        Float,
        default=0.0
    )

    error_count = Column(
        Integer,
        default=0
    )

    confusion_level = Column(
        Float,
        default=0.0
    )

    engagement_level = Column(
        Float,
        default=0.5
    )

    misconception = Column(
        String,
        nullable=True
    )

    diagnosis_severity = Column(
        String,
        nullable=True
    )

    mastery_concept = Column(
        String,
        nullable=True
    )

    mastery_score = Column(
        Float,
        nullable=True
    )

    pedagogy_strategy = Column(
        String,
        nullable=True
    )

    hint_level = Column(
        Integer,
        nullable=True
    )

    tutor_response = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )


# Experiment Log
#
# Research / evaluation data.
#
# IMPORTANT:
# LearningEpisode = learner memory used by the tutor.
# ExperimentLog   = research measurements used for analysis.
#
# Keeping them separate prevents experimental instrumentation
# from becoming part of the learner model itself.

class ExperimentLog(Base):

    __tablename__ = "experiment_logs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # Identity / condition

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False,
        index=True
    )

    assignment_id = Column(
        Integer,
        ForeignKey("assignments.id"),
        nullable=False,
        index=True
    )

    episode_id = Column(
        Integer,
        ForeignKey("learning_episodes.id"),
        nullable=True
    )

    # Later:
    # rule_based
    # single_llm
    # personalized_pomdp
    model_variant = Column(
        String,
        default="personalized_pomdp",
        nullable=False,
        index=True
    )

    # Current learner state

    pause_time = Column(
        Float,
        nullable=True
    )

    error_count = Column(
        Integer,
        nullable=True
    )

    confusion_level = Column(
        Float,
        nullable=True
    )

    engagement_level = Column(
        Float,
        nullable=True
    )

    mastery_score = Column(
        Float,
        nullable=True
    )

    misconception = Column(
        String,
        nullable=True
    )

    # Personalization

    profile_confidence = Column(
        Float,
        nullable=True
    )

    learning_preference = Column(
        String,
        nullable=True
    )

    challenge_tolerance = Column(
        Float,
        nullable=True
    )

    # POMDP state / uncertainty

    most_likely_state = Column(
        String,
        nullable=True
    )

    most_likely_state_probability = Column(
        Float,
        nullable=True
    )

    belief_entropy = Column(
        Float,
        nullable=True
    )

    # Store complete belief/action values as JSON text.
    belief_state_json = Column(
        Text,
        nullable=True
    )

    action_scores_json = Column(
        Text,
        nullable=True
    )

    # Teaching decision

    pedagogy_strategy = Column(
        String,
        nullable=True
    )

    hint_level = Column(
        Integer,
        nullable=True
    )

    answer_policy = Column(
        String,
        nullable=True
    )

    teacher_max_hint_level = Column(
        Integer,
        nullable=True
    )

    teacher_policy_limited = Column(
        Boolean,
        default=False
    )

    # LLM measurement

    llm_used = Column(
        Boolean,
        default=False
    )

    llm_provider = Column(
        String,
        nullable=True
    )

    llm_model = Column(
        String,
        nullable=True
    )

    prompt_tokens = Column(
        Integer,
        nullable=True
    )

    completion_tokens = Column(
        Integer,
        nullable=True
    )

    total_tokens = Column(
        Integer,
        nullable=True
    )

    llm_latency_ms = Column(
        Float,
        nullable=True
    )

    # Whole-system performance

    total_latency_ms = Column(
        Float,
        nullable=True
    )

    tutor_response = Column(
        Text,
        nullable=True
    )

    response_length_chars = Column(
        Integer,
        nullable=True
    )

    # Timestamp

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )