import json
import time
from services.evaluation_service import (
    evaluation_service
)
from pedagogy.baselines import (
    RuleBasedBaseline
)

from services.llm_service import (
    llm_service
)

from services.code_execution_service import (
    code_execution_service
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from pydantic import BaseModel

from sqlalchemy.orm import Session


from agents.affective_agent import (
    AffectiveAgent
)

from agents.diagnostic_agent import (
    DiagnosticAgent
)

from agents.learner_analyzer import (
    LearnerAnalyzer
)

from agents.pedagogy_agent import (
    PedagogyAgent
)

from agents.tutor_executor import (
    TutorExecutor
)


from database import models

from database.db import (
    get_db
)


from fuzzy.confusion_system import (
    calculate_confusion
)


from learner_model.affect import (
    update_affective_state
)

from learner_model.mastery import (
    update_mastery
)

from learner_model.memory import (
    update_misconception_memory
)

from learner_model.profile import (
    dynamic_profile_to_dict,
    get_or_create_dynamic_profile,
    synchronize_stable_profile,
    update_dynamic_profile
)


# Router

router = APIRouter(
    prefix="/tutor",
    tags=["Tutor"]
)


# Agents

diagnostic_agent = (
    DiagnosticAgent()
)

learner_analyzer = (
    LearnerAnalyzer()
)

pedagogy_agent = (
    PedagogyAgent()
)

affective_agent = (
    AffectiveAgent()
)

tutor_executor = (
    TutorExecutor()
)

rule_based_baseline = (
    RuleBasedBaseline()
)
# Request Schemas

class CodeAnalysisRequest(
    BaseModel
):

    code: str


class TutorPlanRequest(
    BaseModel
):

    student_id: int

    assignment_id: int

    # Conceptual/API questions may not have learner code.
    code: str = ""

    pause_time: float

    error_count: int

    # UI language preference for generated tutoring messages.
    preferred_language: str = "en"


class BaselineComparisonRequest(
    BaseModel
):

    student_id: int

    assignment_id: int

    # Conceptual/API questions may not have learner code.
    code: str = ""

    pause_time: float

    error_count: int

    # UI language preference for baseline response generation.
    preferred_language: str = "en"

# Assignment / Question-Type Routing Helpers

SUPPORTED_QUESTION_TYPES = {
    "coding_task",
    "debugging",
    "conceptual",
    "api_usage",
    "code_review"
}


def get_assignment_question_type(
    assignment
) -> str:
    """
    Return a stable question type.

    Old assignments created before question_type existed are treated as
    coding_task so the previous execution/diagnosis behavior is preserved.
    """

    raw_type = getattr(
        assignment,
        "question_type",
        None
    )

    if not raw_type:
        return "coding_task"

    question_type = str(
        raw_type
    ).strip().lower()

    if (
        question_type
        not in SUPPORTED_QUESTION_TYPES
    ):
        return "coding_task"

    return question_type


def get_primary_concept(
    assignment
) -> str:
    """
    Pick a useful mastery concept from the assignment.

    Generic labels such as 'python' are skipped when a more specific
    concept exists.
    """

    concepts = (
        getattr(
            assignment,
            "concepts",
            None
        )
        or
        []
    )

    normalized = [
        str(concept).strip()
        for concept in concepts
        if str(concept).strip()
    ]

    generic = {
        "python",
        "programming",
        "general_python"
    }

    for concept in normalized:
        if concept.lower() not in generic:
            return concept

    if normalized:
        return normalized[0]

    return "general_python"


def make_non_code_diagnosis(
    assignment,
    question_type: str
) -> dict:
    """
    Create a neutral diagnosis for knowledge-oriented questions.

    This avoids pretending that a conceptual/API question is a failed
    executable programming submission.
    """

    concept = get_primary_concept(
        assignment
    )

    if question_type == "conceptual":
        message = (
            "Knowledge-focused conceptual question. "
            "Code execution is not required for this interaction."
        )
        error_type = "conceptual_query"

    elif question_type == "api_usage":
        message = (
            "API-usage question. The tutor should focus on explaining "
            "the relevant API or usage pattern rather than judging code "
            "with hidden tests."
        )
        error_type = "api_usage_query"

    elif question_type == "code_review":
        message = (
            "Code-review interaction. The tutor should analyze quality, "
            "clarity, and design rather than force a binary correctness "
            "judgment."
        )
        error_type = "code_review"

    else:
        message = (
            "Non-executable tutoring interaction."
        )
        error_type = "non_executable_query"

    return {
        "syntax_valid": None,
        "is_correct": None,
        "error_type": error_type,
        "message": message,
        "concept": concept,
        "misconception": None,
        "severity": "low",
        "evidence": {
            "question_type":
                question_type,

            "assignment_title":
                getattr(
                    assignment,
                    "title",
                    None
                ),

            "concepts":
                (
                    getattr(
                        assignment,
                        "concepts",
                        None
                    )
                    or
                    []
                )
        },
        "confidence": 1.0,
        "diagnosis_source":
            "question_type_route",
        "question_type":
            question_type,
        "requires_code_execution":
            False
    }


def diagnose_submission(
    code: str,
    assignment
):
    """
    Route a submission according to Assignment.question_type.

    coding_task
        Hidden tests when available, otherwise structural/semantic diagnosis.

    debugging
        Hidden tests when available. Without tests, use DiagnosticAgent's
        structural + semantic fallback on the learner code.

    conceptual / api_usage
        No code execution. Produce a neutral knowledge-oriented diagnosis.

    code_review
        If learner code exists, analyze it without requiring hidden tests.
        If there is no learner code, use a neutral review diagnosis.
    """

    question_type = (
        get_assignment_question_type(
            assignment
        )
    )

    code = code or ""

    function_name = getattr(
        assignment,
        "function_name",
        None
    )

    hidden_tests = (
        getattr(
            assignment,
            "hidden_tests",
            None
        )
        or
        []
    )

    # Conceptual / API usage: never force executable tests.

    if question_type in {
        "conceptual",
        "api_usage"
    }:
        return (
            make_non_code_diagnosis(
                assignment,
                question_type
            ),
            None
        )

    # Code review: inspect supplied code, but do not require
    # hidden tests or turn the task into pass/fail grading.

    if question_type == "code_review":

        if code.strip():
            diagnosis = (
                diagnostic_agent
                .analyze_code(
                    code=code,
                    assignment=assignment
                )
            )

            diagnosis[
                "question_type"
            ] = question_type

            diagnosis[
                "requires_code_execution"
            ] = False

            # Code review is not automatically a mastery success/failure.
            diagnosis[
                "is_correct"
            ] = None

            diagnosis[
                "diagnosis_source"
            ] = (
                diagnosis.get(
                    "diagnosis_source"
                )
                or
                "code_review_analysis"
            )

            return diagnosis, None

        return (
            make_non_code_diagnosis(
                assignment,
                question_type
            ),
            None
        )

    # Coding task / debugging with executable tests.

    if (
        function_name
        and
        hidden_tests
        and
        code.strip()
    ):

        execution_result = (
            code_execution_service
            .evaluate(
                code=code,
                function_name=function_name,
                hidden_tests=hidden_tests
            )
        )

        diagnosis = (
            diagnostic_agent
            .analyze_code(
                code=code,
                assignment=assignment,
                execution_result=execution_result
            )
        )

        diagnosis[
            "question_type"
        ] = question_type

        diagnosis[
            "requires_code_execution"
        ] = True

        return (
            diagnosis,
            execution_result
        )

    # Debugging/coding task without executable tests:
    # use the existing deterministic + semantic diagnosis.

    if code.strip():

        diagnosis = (
            diagnostic_agent
            .analyze_code(
                code=code,
                assignment=assignment
            )
        )

        diagnosis[
            "question_type"
        ] = question_type

        diagnosis[
            "requires_code_execution"
        ] = False

        return diagnosis, None

    # No learner code was supplied for a code-oriented task.
    diagnosis = {
        "syntax_valid": None,
        "is_correct": None,
        "error_type":
            "missing_student_code",
        "message":
            (
                "This question type normally requires learner code, "
                "but no code was supplied."
            ),
        "concept":
            get_primary_concept(
                assignment
            ),
        "misconception":
            None,
        "severity":
            "low",
        "evidence": {
            "question_type":
                question_type
        },
        "confidence":
            1.0,
        "diagnosis_source":
            "question_type_route",
        "question_type":
            question_type,
        "requires_code_execution":
            False
    }

    return diagnosis, None


def get_or_create_neutral_mastery(
    db: Session,
    student_id: int,
    concept: str
):
    """
    Return mastery without changing the score.

    Used when an interaction does not provide objective correctness evidence,
    e.g. conceptual/API/code-review questions.
    """

    record = (
        db.query(
            models.StudentMastery
        )
        .filter(
            models.StudentMastery.student_id
            ==
            student_id,

            models.StudentMastery.concept
            ==
            concept
        )
        .first()
    )

    if record:
        return record

    record = models.StudentMastery(
        student_id=
            student_id,

        concept=
            concept,

        mastery_score=
            0.5
    )

    db.add(
        record
    )

    db.commit()

    db.refresh(
        record
    )

    return record


# Diagnostic Endpoint

@router.post(
    "/diagnose"
)
def diagnose_code(
    request: CodeAnalysisRequest
):

    diagnosis = (
        diagnostic_agent
        .analyze_code(
            request.code
        )
    )

    return {

        "agent":
            "Diagnostic Agent",

        "diagnosis":
            diagnosis
    }


# Recent Episodic Context

def get_recent_episode_context(
    db,
    student_id: int,
    limit: int = 5
):

    episodes = (

        db.query(
            models.LearningEpisode
        )

        .filter(
            models.LearningEpisode.student_id
            ==
            student_id
        )

        .order_by(
            models.LearningEpisode.created_at.desc()
        )

        .limit(
            limit
        )

        .all()
    )

    episodes = list(
        reversed(
            episodes
        )
    )

    return [

        {
            "assignment_id":
                episode.assignment_id,

            "pause_time":
                episode.pause_time,

            "error_count":
                episode.error_count,

            "confusion_level":
                episode.confusion_level,

            "engagement_level":
                episode.engagement_level,

            "misconception":
                episode.misconception,

            "mastery_concept":
                episode.mastery_concept,

            "mastery_score":
                episode.mastery_score,

            "pedagogy_strategy":
                episode.pedagogy_strategy,

            "hint_level":
                episode.hint_level
        }

        for episode
        in episodes
    ]


# Main Tutor Plan

@router.post(
    "/plan"
)
def create_tutoring_plan(

    request: TutorPlanRequest,

    db: Session = Depends(
        get_db
    )
):

    # Start Whole-Pipeline Timer

    pipeline_start = (
        time.perf_counter()
    )

    # 1. Student

    student = (

        db.query(
            models.Student
        )

        .filter(
            models.Student.id
            ==
            request.student_id
        )

        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    # 2. Stable Profile

    stable_profile = (

        db.query(
            models.StudentProfile
        )

        .filter(
            models.StudentProfile.student_id
            ==
            request.student_id
        )

        .first()
    )

    if not stable_profile:

        raise HTTPException(
            status_code=404,
            detail="Student profile not found"
        )

    # 3. Dynamic Profile

    dynamic_profile = (
        get_or_create_dynamic_profile(

            db=db,

            student_id=
                request.student_id
        )
    )

    # 4. Assignment

    assignment = (

        db.query(
            models.Assignment
        )

        .filter(
            models.Assignment.id
            ==
            request.assignment_id
        )

        .first()
    )

    if not assignment:

        raise HTTPException(
            status_code=404,
            detail="Assignment not found"
        )

    # 5. Fuzzy Confusion

    confusion_level = (
        calculate_confusion(

            pause_time=
                request.pause_time,

            error_count=
                request.error_count
        )
    )

    # 6. Affective State

    affective_state = (
        update_affective_state(

            db=db,

            profile=
                stable_profile,

            confusion_level=
                confusion_level,

            pause_time=
                request.pause_time,

            error_count=
                request.error_count
        )
    )

    # 7. Diagnostic Agent

    diagnosis, execution_result = (
        diagnose_submission(
            code=request.code,
            assignment=assignment
        )
    )

    misconception = (
        diagnosis.get(
            "misconception"
        )
    )

    # 8. Misconception Memory

    memory_record = None

    if misconception:

        memory_record = (
            update_misconception_memory(

                db=db,

                student_id=
                    request.student_id,

                misconception=
                    misconception
            )
        )

    # 9. Mastery

    concept = (
        diagnosis.get(
            "concept"
        )
        or
        "general_python"
    )

    diagnosis_correctness = (
        diagnosis.get(
            "is_correct"
        )
    )

    question_type = (
        get_assignment_question_type(
            assignment
        )
    )

    # Mastery evidence gate
    #
    # Mastery changes only when diagnosis provides an explicit
    # objective correctness signal (True/False).
    #
    # This applies to ALL question types. In particular,
    # debugging/coding tasks without executable evidence must
    # not gain mastery merely because no structural error was
    # detected.

    objective_mastery_evidence = (
        diagnosis_correctness
        is not None
    )

    if not objective_mastery_evidence:

        mastery_record = (
            get_or_create_neutral_mastery(

                db=db,

                student_id=
                    request.student_id,

                concept=
                    concept
            )
        )

    else:

        success = bool(
            diagnosis_correctness
        )

        mastery_record = (
            update_mastery(

                db=db,

                student_id=
                    request.student_id,

                concept=
                    concept,

                success=
                    success
            )
        )

    # 10. Episodic Memory

    recent_episodes = (
        get_recent_episode_context(

            db=db,

            student_id=
                request.student_id,

            limit=
                5
        )
    )

    # 11. Learner Evidence

    current_question_type = (
        get_assignment_question_type(
            assignment
        )
    )

    objective_correctness_available = (
        diagnosis.get(
            "is_correct"
        )
        is not None
    )

    if objective_correctness_available:

        performance_status = (
            "success"
            if bool(
                diagnosis.get(
                    "is_correct"
                )
            )
            else
            "failure"
        )

    else:

        performance_status = (
            "not_assessed"
        )

    learner_evidence = {

        "assignment": {

            "title":
                assignment.title,

            "learning_objective":
                assignment.learning_objective,

            "question_type":
                current_question_type
        },

        "behavior": {

            "pause_time":
                request.pause_time,

            "error_count":
                request.error_count,

            "confusion_level":
                affective_state[
                    "confusion_level"
                ],

            "engagement_level":
                affective_state[
                    "engagement_level"
                ]
        },

        "diagnosis": {

            "misconception":
                misconception,

            "severity":
                diagnosis.get(
                    "severity"
                ),

            "is_correct":
                diagnosis.get(
                    "is_correct"
                )
        },

        "performance_evidence": {

            "objective_correctness_available":
                objective_correctness_available,

            "status":
                performance_status,

            "instruction":
                (
                    "Treat this interaction as objective performance "
                    "evidence only when status is success or failure. "
                    "When status is not_assessed, do not describe the "
                    "interaction as a successful or failed event."
                )
        },

        "mastery": {

            "concept":
                mastery_record.concept,

            "score":
                mastery_record.mastery_score
        },

        "misconception_memory": {

            "misconception":
                (
                    memory_record.misconception
                    if memory_record
                    else None
                ),

            "occurrence_count":
                (
                    memory_record.occurrence_count
                    if memory_record
                    else 0
                )
        }
    }

    # 12. Selective Learner Analysis
    #
    # Expensive LLM analysis is NOT executed every turn.

    current_dynamic_profile = (
        dynamic_profile_to_dict(
            dynamic_profile
        )
    )

    # Count existing interactions.
    #
    # Current interaction has not yet been stored as a
    # LearningEpisode, so +1 represents this interaction.
    previous_interaction_count = (

        db.query(
            models.LearningEpisode
        )

        .filter(
            models.LearningEpisode.student_id
            ==
            request.student_id
        )

        .count()
    )

    interaction_number = (
        previous_interaction_count
        +
        1
    )

    should_analyze, analysis_trigger = (
        learner_analyzer
        .should_analyze(

            interaction_number=
                interaction_number,

            current_evidence=
                learner_evidence,

            recent_episodes=
                recent_episodes
        )
    )

    # Neutral interaction guard
    #
    # A conceptual/API/code-review interaction may have
    # misconception=None simply because no objective code
    # diagnosis is appropriate. That must not be interpreted
    # as "the previous misconception disappeared".

    neutral_nonperformance_interaction = (
        current_question_type
        in {
            "conceptual",
            "api_usage",
            "code_review"
        }
        and
        not objective_correctness_available
    )

    if (
        neutral_nonperformance_interaction
        and
        analysis_trigger
        ==
        "misconception_changed"
    ):

        should_analyze = False

        analysis_trigger = (
            "non_performance_interaction"
        )

    # 12A. LLM Learner Analysis

    if should_analyze:

        learner_analysis = (
            learner_analyzer
            .analyze(

                current_evidence=
                    learner_evidence,

                recent_episodes=
                    recent_episodes,

                current_profile=
                    current_dynamic_profile
            )
        )

        learner_analysis[
            "analysis_trigger"
        ] = analysis_trigger

        # Semantic guard for non-performance interactions.
        #
        # Even when analysis is legitimately triggered by an
        # initial/periodic rule, conceptual/API/code-review
        # questions do not provide objective success/failure
        # evidence unless a correctness signal exists.

        if neutral_nonperformance_interaction:

            learner_analysis[
                "performance_status"
            ] = "not_assessed"

            learner_analysis[
                "performance_interpretation"
            ] = (
                "No objective correctness evidence is available "
                "for this interaction. Behavioral signals may "
                "inform short-term learner state, but this event "
                "must not be interpreted as a successful or failed "
                "performance attempt."
            )

            existing_summary = str(
                learner_analysis.get(
                    "evidence_summary"
                )
                or
                ""
            )

            if existing_summary:

                learner_analysis[
                    "evidence_summary"
                ] = (
                    "This interaction contains no objective "
                    "correctness evidence. "
                    + existing_summary
                )

            else:

                learner_analysis[
                    "evidence_summary"
                ] = (
                    "This interaction contains no objective "
                    "correctness evidence; it should not be "
                    "described as a successful or failed event."
                )

        # Profile-update gate
        #
        # Running the Learner Analyzer does NOT automatically
        # mean that the longitudinal profile should be changed.
        #
        # Low-confidence analysis or analysis that explicitly
        # reports insufficient evidence is kept for inspection
        # but does not increment profile evidence_count or
        # modify the stored dynamic profile.

        try:

            analysis_confidence = float(
                learner_analysis.get(
                    "confidence",
                    0.0
                )
                or
                0.0
            )

        except (
            TypeError,
            ValueError
        ):

            analysis_confidence = 0.0

        evidence_summary_lower = str(
            learner_analysis.get(
                "evidence_summary"
            )
            or
            ""
        ).lower()

        insufficient_evidence_markers = (
            "insufficient data",
            "insufficient evidence",
            "not enough evidence",
            "does not provide enough evidence",
            "do not provide enough evidence"
        )

        explicitly_insufficient = any(
            marker in evidence_summary_lower
            for marker
            in insufficient_evidence_markers
        )

        profile_update_allowed = (
            analysis_confidence >= 0.25
            and
            not explicitly_insufficient
        )

        if profile_update_allowed:

            dynamic_profile = (
                update_dynamic_profile(

                    db=db,

                    dynamic_profile=
                        dynamic_profile,

                    analysis=
                        learner_analysis
                )
            )

            profile_updated = True

            profile_update_reason = (
                "sufficient_new_evidence"
            )

        else:

            profile_updated = False

            if explicitly_insufficient:

                profile_update_reason = (
                    "insufficient_evidence"
                )

            else:

                profile_update_reason = (
                    "low_analysis_confidence"
                )

            learner_analysis[
                "profile_update_skipped"
            ] = True

            learner_analysis[
                "profile_update_reason"
            ] = profile_update_reason

    # 12B. Reuse Existing Profile

    else:

        learner_analysis = (
            learner_analyzer
            .build_cached_analysis(

                current_profile=
                    current_dynamic_profile,

                reason=
                    analysis_trigger
            )
        )

        # IMPORTANT:
        # Do NOT call update_dynamic_profile().
        #
        # Cached data is not new evidence.
        profile_updated = False

        profile_update_reason = (
            "analysis_not_triggered"
        )

    # 13. Stable Profile Synchronization

    if profile_updated:

        stable_profile = (
            synchronize_stable_profile(

                db=db,

                stable_profile=
                    stable_profile,

                dynamic_profile=
                    dynamic_profile
            )
        )

    dynamic_profile_context = (
        dynamic_profile_to_dict(
            dynamic_profile
        )
    )

    # 14. Personalized POMDP

    pedagogy = (
        pedagogy_agent
        .choose_strategy(

            diagnosis=
                diagnosis,

            confusion_level=
                confusion_level,

            learning_preference=
                stable_profile.learning_preference,

            teacher_max_hint_level=
                assignment.max_hint_level,

            dynamic_profile=
                dynamic_profile_context,

            mastery_score=
                mastery_record.mastery_score,

            recent_episodes=
                recent_episodes
        )
    )

    # 15. Persona

    persona = (
        affective_agent
        .choose_persona(

            confusion_level=
                confusion_level,

            personality=
                stable_profile.personality,

            learning_preference=
                stable_profile.learning_preference,

            dynamic_profile=
                dynamic_profile_context
        )
    )

    # 16. Executor Context

    student_profile_context = {

        "stable_profile": {

            "personality":
                stable_profile.personality,

            "learning_preference":
                stable_profile.learning_preference,

            "challenge_tolerance":
                stable_profile.challenge_tolerance
        },

        "dynamic_profile":
            dynamic_profile_context,

        "current_state": {

            "confusion_level":
                affective_state[
                    "confusion_level"
                ],

            "engagement_level":
                affective_state[
                    "engagement_level"
                ]
        }
    }

    mastery_context = {

        "concept":
            mastery_record.concept,

        "score":
            mastery_record.mastery_score
    }

    memory_context = {

        "current_misconception":
            (
                memory_record.misconception
                if memory_record
                else None
            ),

        "occurrence_count":
            (
                memory_record.occurrence_count
                if memory_record
                else 0
            ),

        "recent_episodes":
            recent_episodes
    }

    assignment_context = {

        "id":
            assignment.id,

        "title":
            assignment.title,

        "description":
            assignment.description,

        "learning_objective":
            assignment.learning_objective,

        "function_name":
            getattr(
                assignment,
                "function_name",
                None
            ),

        "language":
            getattr(
                assignment,
                "language",
                "Python"
            ),

        "difficulty":
            getattr(
                assignment,
                "difficulty",
                None
            ),

        "question_type":
            get_assignment_question_type(
                assignment
            ),

        "learning_objectives":
            (
                getattr(
                    assignment,
                    "learning_objectives",
                    None
                )
                or
                []
            ),

        "concepts":
            (
                getattr(
                    assignment,
                    "concepts",
                    None
                )
                or
                []
            ),

        "constraints":
            (
                getattr(
                    assignment,
                    "constraints",
                    None
                )
                or
                []
            ),

        "visible_examples":
            (
                getattr(
                    assignment,
                    "visible_examples",
                    None
                )
                or
                []
            ),

        "max_hint_level":
            assignment.max_hint_level
    }

    # 17. Qwen Tutor Executor + Usage Metadata

    executor_result = (
        tutor_executor
        .generate_response_with_metadata(

            diagnosis=
                diagnosis,

            pedagogy=
                pedagogy,

            persona=
                persona,

            student_profile=
                student_profile_context,

            mastery=
                mastery_context,

            memory=
                memory_context,

            assignment=
                assignment_context,

            code=
                request.code,

            preferred_language=
                request.preferred_language
        )
    )

    tutor_response = (
        executor_result[
            "response"
        ]
    )

    llm_used = (
        executor_result.get(
            "llm_used",
            False
        )
    )

    llm_metadata = (
        executor_result.get(
            "llm_metadata",
            {}
        )
    )

    # 18. Save Learning Episode

    episode = models.LearningEpisode(

        student_id=
            request.student_id,

        assignment_id=
            request.assignment_id,

        code=
            request.code,

        pause_time=
            request.pause_time,

        error_count=
            request.error_count,

        confusion_level=
            affective_state[
                "confusion_level"
            ],

        engagement_level=
            affective_state[
                "engagement_level"
            ],

        misconception=
            misconception,

        diagnosis_severity=
            diagnosis.get(
                "severity"
            ),

        mastery_concept=
            mastery_record.concept,

        mastery_score=
            mastery_record.mastery_score,

        pedagogy_strategy=
            pedagogy.get(
                "strategy"
            ),

        hint_level=
            pedagogy.get(
                "hint_level"
            ),

        tutor_response=
            tutor_response
    )

    db.add(
        episode
    )

    db.commit()

    db.refresh(
        episode
    )

    # 19. Calculate Whole Pipeline Latency

    total_latency_ms = (
        (
            time.perf_counter()
            -
            pipeline_start
        )
        *
        1000
    )

    # 20. Extract POMDP Data

    pomdp_data = (
        pedagogy.get(
            "pomdp",
            {}
        )
        or
        {}
    )

    belief_state = (
        pomdp_data.get(
            "belief_state",
            {}
        )
        or
        {}
    )

    most_likely_state = (
        pomdp_data.get(
            "most_likely_state"
        )
    )

    most_likely_probability = None

    if most_likely_state:

        most_likely_probability = (
            belief_state.get(
                most_likely_state
            )
        )

    action_scores = (
        pomdp_data.get(
            "action_scores",
            {}
        )
        or
        {}
    )

    # 21. Save Experiment Log

    experiment_log = (
        models.ExperimentLog(

            # Condition

            student_id=
                request.student_id,

            assignment_id=
                request.assignment_id,

            episode_id=
                episode.id,

            model_variant=
                "personalized_pomdp",

            # Learner state

            pause_time=
                request.pause_time,

            error_count=
                request.error_count,

            confusion_level=
                affective_state[
                    "confusion_level"
                ],

            engagement_level=
                affective_state[
                    "engagement_level"
                ],

            mastery_score=
                mastery_record.mastery_score,

            misconception=
                misconception,

            # Personalization

            profile_confidence=
                dynamic_profile_context.get(
                    "profile_confidence"
                ),

            learning_preference=
                dynamic_profile_context.get(
                    "learning_preference"
                ),

            challenge_tolerance=
                dynamic_profile_context.get(
                    "challenge_tolerance"
                ),

            # POMDP

            most_likely_state=
                most_likely_state,

            most_likely_state_probability=
                most_likely_probability,

            belief_entropy=
                pomdp_data.get(
                    "belief_entropy"
                ),

            belief_state_json=
                json.dumps(
                    belief_state,
                    ensure_ascii=False
                ),

            action_scores_json=
                json.dumps(
                    action_scores,
                    ensure_ascii=False
                ),

            # Teaching decision

            pedagogy_strategy=
                pedagogy.get(
                    "strategy"
                ),

            hint_level=
                pedagogy.get(
                    "hint_level"
                ),

            answer_policy=
                pedagogy.get(
                    "answer_policy"
                ),

            teacher_max_hint_level=
                assignment.max_hint_level,

            teacher_policy_limited=
                bool(
                    pedagogy.get(
                        "teacher_policy_limited",
                        False
                    )
                ),

            # LLM

            llm_used=
                llm_used,

            llm_provider=
                llm_metadata.get(
                    "provider"
                ),

            llm_model=
                llm_metadata.get(
                    "model"
                ),

            prompt_tokens=
                llm_metadata.get(
                    "prompt_tokens"
                ),

            completion_tokens=
                llm_metadata.get(
                    "completion_tokens"
                ),

            total_tokens=
                llm_metadata.get(
                    "total_tokens"
                ),

            llm_latency_ms=
                llm_metadata.get(
                    "latency_ms"
                ),

            # Whole system

            total_latency_ms=
                round(
                    total_latency_ms,
                    2
                ),

            tutor_response=
                tutor_response,

            response_length_chars=
                len(
                    tutor_response
                    or
                    ""
                )
        )
    )

    db.add(
        experiment_log
    )

    db.commit()

    db.refresh(
        experiment_log
    )

    # 22. Response

    return {

        "student": {

            "id":
                student.id,

            "name":
                student.name
        },

        "assignment":
            assignment_context,

        "learner_analysis":
            learner_analysis,
        "learner_analysis_status": {

            "interaction_number":
                interaction_number,

            "llm_analysis_performed":
                should_analyze,

            "profile_updated":
                profile_updated,

            "profile_update_reason":
                profile_update_reason,

            "trigger":
                analysis_trigger
        },
        "student_profile":
            student_profile_context,

        "diagnostic_agent":
            diagnosis,

        "routing": {

            "question_type":
                get_assignment_question_type(
                    assignment
                ),

            "code_execution_required":
                bool(
                    diagnosis.get(
                        "requires_code_execution",
                        False
                    )
                ),

            "route":
                (
                    "execution_and_diagnosis"
                    if execution_result
                    is not None
                    else
                    (
                        "knowledge_tutoring"
                        if get_assignment_question_type(
                            assignment
                        )
                        in {
                            "conceptual",
                            "api_usage"
                        }
                        else
                        "semantic_or_structural_analysis"
                    )
                )
        },

        "code_execution": {

            "used":
                execution_result
                is not None,

            "all_tests_passed":
                (
                    execution_result.get(
                        "all_tests_passed"
                    )
                    if execution_result
                    else None
                ),

            "tests_run":
                (
                    execution_result.get(
                        "tests_run"
                    )
                    if execution_result
                    else 0
                ),

            "tests_passed":
                (
                    execution_result.get(
                        "tests_passed"
                    )
                    if execution_result
                    else 0
                ),

            "tests_failed":
                (
                    execution_result.get(
                        "tests_failed"
                    )
                    if execution_result
                    else 0
                ),

            "error_type":
                (
                    execution_result.get(
                        "error_type"
                    )
                    if execution_result
                    else None
                )
        },

        "pedagogy_agent":
            pedagogy,

        "affective_agent":
            persona,

        "learner_affect":
            affective_state,

        "memory":
            memory_context,

        "mastery":
            mastery_context,

        "teaching_decision": {

            "strategy":
                pedagogy.get(
                    "strategy"
                ),

            "hint_level":
                pedagogy.get(
                    "hint_level"
                ),

            "answer_policy":
                pedagogy.get(
                    "answer_policy"
                ),

            "decision_source":
                pedagogy.get(
                    "decision_source"
                )
        },

        "episode": {

            "id":
                episode.id,

            "saved":
                True
        },

        # Experiment information

        "experiment": {

            "log_id":
                experiment_log.id,

            "model_variant":
                experiment_log.model_variant,

            "total_latency_ms":
                experiment_log.total_latency_ms,

            "llm": {

                "used":
                    experiment_log.llm_used,

                "provider":
                    experiment_log.llm_provider,

                "model":
                    experiment_log.llm_model,

                "prompt_tokens":
                    experiment_log.prompt_tokens,

                "completion_tokens":
                    experiment_log.completion_tokens,

                "total_tokens":
                    experiment_log.total_tokens,

                "latency_ms":
                    experiment_log.llm_latency_ms
            }
        },

        "tutor_response":
            tutor_response
    }


# Experiment Logs
#
# Simple research-inspection endpoint.

@router.get(
    "/experiment-logs"
)
def get_experiment_logs(

    limit: int = 100,

    db: Session = Depends(
        get_db
    )
):

    safe_limit = max(
        1,
        min(
            limit,
            1000
        )
    )

    logs = (

        db.query(
            models.ExperimentLog
        )

        .order_by(
            models.ExperimentLog.created_at.desc()
        )

        .limit(
            safe_limit
        )

        .all()
    )

    return [

        {
            "id":
                log.id,

            "student_id":
                log.student_id,

            "assignment_id":
                log.assignment_id,

            "episode_id":
                log.episode_id,

            "model_variant":
                log.model_variant,

            "confusion_level":
                log.confusion_level,

            "engagement_level":
                log.engagement_level,

            "mastery_score":
                log.mastery_score,

            "misconception":
                log.misconception,

            "profile_confidence":
                log.profile_confidence,

            "most_likely_state":
                log.most_likely_state,

            "most_likely_state_probability":
                log.most_likely_state_probability,

            "belief_entropy":
                log.belief_entropy,

            "pedagogy_strategy":
                log.pedagogy_strategy,

            "hint_level":
                log.hint_level,

            "answer_policy":
                log.answer_policy,

            "teacher_max_hint_level":
                log.teacher_max_hint_level,

            "teacher_policy_limited":
                log.teacher_policy_limited,

            "llm_used":
                log.llm_used,

            "llm_provider":
                log.llm_provider,

            "llm_model":
                log.llm_model,

            "prompt_tokens":
                log.prompt_tokens,

            "completion_tokens":
                log.completion_tokens,

            "total_tokens":
                log.total_tokens,

            "llm_latency_ms":
                log.llm_latency_ms,

            "total_latency_ms":
                log.total_latency_ms,

            "response_length_chars":
                log.response_length_chars,

            "created_at":
                log.created_at
        }

        for log in logs
    ]

# Baseline Comparison Helpers

def build_single_llm_prompt(
    assignment,
    code: str,
    pause_time: float,
    error_count: int,
    confusion_level: float
) -> str:

    return f"""
You are an AI programming tutor.

Help the student with the current programming problem.

You must independently decide:
- what the student is struggling with
- how much help to provide
- whether to ask a question
- whether to explain directly
- whether to show code

There is no external pedagogical planner.


==================================================
ASSIGNMENT
==================================================

Title:
{assignment.title}

Description:
{assignment.description}

Learning objective:
{assignment.learning_objective}


==================================================
CURRENT STUDENT CODE
==================================================

{code}


==================================================
OBSERVED BEHAVIOR
==================================================

Typing pause time:
{pause_time}

Error count:
{error_count}

Estimated confusion:
{round(confusion_level, 3)}


==================================================
INSTRUCTION
==================================================

Give the student the tutoring response you believe is most
appropriate.

Be concise, educational, and supportive.

Generate only the tutoring message.
""".strip()


# Save Experiment Result

def save_comparison_log(
    db,
    request,
    assignment,
    variant: str,
    diagnosis: dict,
    confusion_level: float,
    engagement_level: float | None,
    mastery_score: float | None,
    dynamic_profile_context: dict,
    pedagogy: dict | None,
    tutor_response: str,
    llm_used: bool,
    llm_metadata: dict,
    total_latency_ms: float
):

    pedagogy = (
        pedagogy
        or {}
    )

    pomdp_data = (
        pedagogy.get(
            "pomdp",
            {}
        )
        or {}
    )

    belief_state = (
        pomdp_data.get(
            "belief_state",
            {}
        )
        or {}
    )

    most_likely_state = (
        pomdp_data.get(
            "most_likely_state"
        )
    )

    most_likely_probability = None

    if (
        most_likely_state
        and
        belief_state
    ):

        most_likely_probability = (
            belief_state.get(
                most_likely_state
            )
        )

    experiment_log = (
        models.ExperimentLog(

            student_id=
                request.student_id,

            assignment_id=
                request.assignment_id,

            # Comparison runs do not create learner-memory
            # episodes.
            episode_id=
                None,

            model_variant=
                variant,

            pause_time=
                request.pause_time,

            error_count=
                request.error_count,

            confusion_level=
                confusion_level,

            engagement_level=
                engagement_level,

            mastery_score=
                mastery_score,

            misconception=
                diagnosis.get(
                    "misconception"
                ),

            profile_confidence=
                dynamic_profile_context.get(
                    "profile_confidence"
                ),

            learning_preference=
                dynamic_profile_context.get(
                    "learning_preference"
                ),

            challenge_tolerance=
                dynamic_profile_context.get(
                    "challenge_tolerance"
                ),

            most_likely_state=
                most_likely_state,

            most_likely_state_probability=
                most_likely_probability,

            belief_entropy=
                pomdp_data.get(
                    "belief_entropy"
                ),

            belief_state_json=
                (
                    json.dumps(
                        belief_state,
                        ensure_ascii=False
                    )
                    if belief_state
                    else None
                ),

            action_scores_json=
                (
                    json.dumps(
                        pomdp_data.get(
                            "action_scores",
                            {}
                        ),
                        ensure_ascii=False
                    )
                    if pomdp_data
                    else None
                ),

            pedagogy_strategy=
                pedagogy.get(
                    "strategy"
                ),

            hint_level=
                pedagogy.get(
                    "hint_level"
                ),

            answer_policy=
                pedagogy.get(
                    "answer_policy"
                ),

            teacher_max_hint_level=
                assignment.max_hint_level,

            teacher_policy_limited=
                bool(
                    pedagogy.get(
                        "teacher_policy_limited",
                        False
                    )
                ),

            llm_used=
                llm_used,

            llm_provider=
                llm_metadata.get(
                    "provider"
                ),

            llm_model=
                llm_metadata.get(
                    "model"
                ),

            prompt_tokens=
                llm_metadata.get(
                    "prompt_tokens"
                ),

            completion_tokens=
                llm_metadata.get(
                    "completion_tokens"
                ),

            total_tokens=
                llm_metadata.get(
                    "total_tokens"
                ),

            llm_latency_ms=
                llm_metadata.get(
                    "latency_ms"
                ),

            total_latency_ms=
                round(
                    total_latency_ms,
                    2
                ),

            tutor_response=
                tutor_response,

            response_length_chars=
                len(
                    tutor_response
                    or
                    ""
                )
        )
    )

    db.add(
        experiment_log
    )

    db.commit()

    db.refresh(
        experiment_log
    )

    return experiment_log


# Baseline Comparison Endpoint

@router.post(
    "/compare"
)
def compare_tutoring_methods(

    request: BaselineComparisonRequest,

    db: Session = Depends(
        get_db
    )
):

    # 1. Load Student

    student = (

        db.query(
            models.Student
        )

        .filter(
            models.Student.id
            ==
            request.student_id
        )

        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found"
        )

    # 2. Assignment

    assignment = (

        db.query(
            models.Assignment
        )

        .filter(
            models.Assignment.id
            ==
            request.assignment_id
        )

        .first()
    )

    if not assignment:

        raise HTTPException(
            status_code=404,
            detail="Assignment not found"
        )

    # 3. Stable Profile

    stable_profile = (

        db.query(
            models.StudentProfile
        )

        .filter(
            models.StudentProfile.student_id
            ==
            request.student_id
        )

        .first()
    )

    if not stable_profile:

        raise HTTPException(
            status_code=404,
            detail="Student profile not found"
        )

    # 4. Dynamic Profile
    #
    # READ ONLY.
    #
    # Comparison experiment must not modify the learner model.

    dynamic_profile = (
        get_or_create_dynamic_profile(

            db=db,

            student_id=
                request.student_id
        )
    )

    dynamic_profile_context = (
        dynamic_profile_to_dict(
            dynamic_profile
        )
    )

    # 5. Same Observation For Every Method

    confusion_level = (
        calculate_confusion(

            pause_time=
                request.pause_time,

            error_count=
                request.error_count
        )
    )

    # Avoid update_affective_state() here because comparison
    # should not modify the persistent student profile.

    engagement_level = max(
        0.0,
        min(
            1.0,
            (
                1.0
                -
                0.65
                *
                confusion_level
                -
                0.04
                *
                min(
                    request.error_count,
                    5
                )
            )
        )
    )

    # 6. Same Diagnosis

    diagnosis, execution_result = (
        diagnose_submission(
            code=request.code,
            assignment=assignment
        )
    )

    # 7. Existing Mastery — READ ONLY

    concept = (
        diagnosis.get(
            "concept"
        )
        or
        "general_python"
    )

    mastery_record = (

        db.query(
            models.StudentMastery
        )

        .filter(
            models.StudentMastery.student_id
            ==
            request.student_id,

            models.StudentMastery.concept
            ==
            concept
        )

        .first()
    )

    mastery_score = (
        mastery_record.mastery_score
        if mastery_record
        else 0.5
    )

    # 8. Existing Recent Episodes — READ ONLY

    recent_episodes = (
        get_recent_episode_context(

            db=db,

            student_id=
                request.student_id,

            limit=
                5
        )
    )

    # Shared Assignment Context

    assignment_context = {

        "id":
            assignment.id,

        "title":
            assignment.title,

        "description":
            assignment.description,

        "learning_objective":
            assignment.learning_objective,

        "function_name":
            getattr(
                assignment,
                "function_name",
                None
            ),

        "language":
            getattr(
                assignment,
                "language",
                "Python"
            ),

        "difficulty":
            getattr(
                assignment,
                "difficulty",
                None
            ),

        "question_type":
            get_assignment_question_type(
                assignment
            ),

        "learning_objectives":
            (
                getattr(
                    assignment,
                    "learning_objectives",
                    None
                )
                or
                []
            ),

        "concepts":
            (
                getattr(
                    assignment,
                    "concepts",
                    None
                )
                or
                []
            ),

        "constraints":
            (
                getattr(
                    assignment,
                    "constraints",
                    None
                )
                or
                []
            ),

        "visible_examples":
            (
                getattr(
                    assignment,
                    "visible_examples",
                    None
                )
                or
                []
            ),

        "max_hint_level":
            assignment.max_hint_level
    }

    # Shared Student Context

    student_profile_context = {

        "stable_profile": {

            "personality":
                stable_profile.personality,

            "learning_preference":
                stable_profile.learning_preference,

            "challenge_tolerance":
                stable_profile.challenge_tolerance
        },

        "dynamic_profile":
            dynamic_profile_context,

        "current_state": {

            "confusion_level":
                confusion_level,

            "engagement_level":
                engagement_level
        }
    }

    mastery_context = {

        "concept":
            concept,

        "score":
            mastery_score
    }

    memory_context = {

        "recent_episodes":
            recent_episodes
    }

    # Results

    results = {}

    # BASELINE 1
    #
    # RULE-BASED TUTOR

    rule_start = (
        time.perf_counter()
    )

    rule_plan = (
        rule_based_baseline
        .create_plan(

            diagnosis=
                diagnosis,

            confusion_level=
                confusion_level,

            teacher_max_hint_level=
                assignment.max_hint_level
        )
    )

    # Rule baseline deliberately uses deterministic local
    # response generation rather than Qwen.

    rule_response = (
        tutor_executor
        ._fallback_response(

            diagnosis=
                diagnosis,

            pedagogy=
                rule_plan
        )
    )

    rule_latency_ms = (
        (
            time.perf_counter()
            -
            rule_start
        )
        *
        1000
    )

    rule_log = (
        save_comparison_log(

            db=db,

            request=request,

            assignment=assignment,

            variant=
                "rule_based",

            diagnosis=
                diagnosis,

            confusion_level=
                confusion_level,

            engagement_level=
                engagement_level,

            mastery_score=
                mastery_score,

            dynamic_profile_context=
                {},

            pedagogy=
                rule_plan,

            tutor_response=
                rule_response,

            llm_used=
                False,

            # Rule-Based Baseline Does Not Use an LLM
            #
            # Store explicit zeros instead of NULL so aggregate
            # statistics correctly report zero token usage.

            llm_metadata=
                {

                    "provider":
                        None,

                    "model":
                        None,

                    "prompt_tokens":
                        0,

                    "completion_tokens":
                        0,

                    "total_tokens":
                        0,

                    "latency_ms":
                        0.0
                },

            total_latency_ms=
                rule_latency_ms
        )
    )

    results[
        "rule_based"
    ] = {

        "experiment_log_id":
            rule_log.id,

        "teaching_decision":
            rule_plan,

        "tutor_response":
            rule_response,

        "performance": {

            "llm_used":
                False,

            "total_tokens":
                0,

            "total_latency_ms":
                round(
                    rule_latency_ms,
                    2
                )
        }
    }

    # BASELINE 2
    #
    # SINGLE LLM
    #
    # No:
    #   POMDP
    #   learner profile
    #   learner memory
    #   external pedagogy planner

    single_start = (
        time.perf_counter()
    )

    single_prompt = (
        build_single_llm_prompt(

            assignment=
                assignment,

            code=
                request.code,

            pause_time=
                request.pause_time,

            error_count=
                request.error_count,

            confusion_level=
                confusion_level
        )
    )

    # DEBUG: Single-LLM Prompt Size

    print(
        "[DEBUG] Single LLM prompt chars:",
        len(
            single_prompt
        )
    )

    print(
        "[DEBUG] Single LLM prompt preview:"
    )

    print(
        single_prompt[:500]
    )

    print(
        "[DEBUG] End Single LLM prompt preview"
    )

    single_result = (
        llm_service
        .generate_with_metadata(

            prompt=
                single_prompt,

            temperature=
                0.3,

            max_tokens=
                350
        )
    )

    single_latency_ms = (
        (
            time.perf_counter()
            -
            single_start
        )
        *
        1000
    )

    if single_result:

        single_response = (
            single_result.get(
                "text",
                ""
            )
        )

        single_metadata = {

            "provider":
                single_result.get(
                    "provider"
                ),

            "model":
                single_result.get(
                    "model"
                ),

            "prompt_tokens":
                single_result.get(
                    "prompt_tokens"
                ),

            "completion_tokens":
                single_result.get(
                    "completion_tokens"
                ),

            "total_tokens":
                single_result.get(
                    "total_tokens"
                ),

            "latency_ms":
                single_result.get(
                    "latency_ms"
                )
        }

        single_llm_used = True

    else:

        single_response = (
            "The language model was unavailable."
        )

        single_metadata = {}

        single_llm_used = False

    single_log = (
        save_comparison_log(

            db=db,

            request=request,

            assignment=assignment,

            variant="single_llm",

            diagnosis=diagnosis,

            confusion_level=
                confusion_level,

            engagement_level=
                engagement_level,

            mastery_score=
                mastery_score,

            dynamic_profile_context=
                {},

            pedagogy=
                {},

            tutor_response=
                single_response,

            llm_used=
                single_llm_used,

            llm_metadata=
                single_metadata,

            total_latency_ms=
                single_latency_ms
        )
    )

    results[
        "single_llm"
    ] = {

        "experiment_log_id":
            single_log.id,

        "teaching_decision":
            None,

        "tutor_response":
            single_response,

        "performance": {

            "llm_used":
                single_llm_used,

            "prompt_tokens":
                single_metadata.get(
                    "prompt_tokens"
                ),

            "completion_tokens":
                single_metadata.get(
                    "completion_tokens"
                ),

            "total_tokens":
                single_metadata.get(
                    "total_tokens"
                ),

            "llm_latency_ms":
                single_metadata.get(
                    "latency_ms"
                ),

            "total_latency_ms":
                round(
                    single_latency_ms,
                    2
                )
        }
    }

    # METHOD 3
    #
    # PERSONALIZED POMDP
    #
    # This is the proposed method.
    #
    # IMPORTANT:
    # Profile is READ ONLY during comparison.

    pomdp_start = (
        time.perf_counter()
    )

    pomdp_plan = (
        pedagogy_agent
        .choose_strategy(

            diagnosis=
                diagnosis,

            confusion_level=
                confusion_level,

            learning_preference=
                stable_profile.learning_preference,

            teacher_max_hint_level=
                assignment.max_hint_level,

            dynamic_profile=
                dynamic_profile_context,

            mastery_score=
                mastery_score,

            recent_episodes=
                recent_episodes
        )
    )

    persona = (
        affective_agent
        .choose_persona(

            confusion_level=
                confusion_level,

            personality=
                stable_profile.personality,

            learning_preference=
                stable_profile.learning_preference,

            dynamic_profile=
                dynamic_profile_context
        )
    )

    pomdp_executor_result = (
        tutor_executor
        .generate_response_with_metadata(

            diagnosis=
                diagnosis,

            pedagogy=
                pomdp_plan,

            persona=
                persona,

            student_profile=
                student_profile_context,

            mastery=
                mastery_context,

            memory=
                memory_context,

            assignment=
                assignment_context,

            code=
                request.code,

            preferred_language=
                request.preferred_language
        )
    )

    pomdp_response = (
        pomdp_executor_result.get(
            "response",
            ""
        )
    )

    pomdp_metadata = (
        pomdp_executor_result.get(
            "llm_metadata",
            {}
        )
        or
        {}
    )

    pomdp_llm_used = (
        pomdp_executor_result.get(
            "llm_used",
            False
        )
    )

    pomdp_latency_ms = (
        (
            time.perf_counter()
            -
            pomdp_start
        )
        *
        1000
    )

    pomdp_log = (
        save_comparison_log(

            db=db,

            request=request,

            assignment=assignment,

            variant=
                "personalized_pomdp",

            diagnosis=
                diagnosis,

            confusion_level=
                confusion_level,

            engagement_level=
                engagement_level,

            mastery_score=
                mastery_score,

            dynamic_profile_context=
                dynamic_profile_context,

            pedagogy=
                pomdp_plan,

            tutor_response=
                pomdp_response,

            llm_used=
                pomdp_llm_used,

            llm_metadata=
                pomdp_metadata,

            total_latency_ms=
                pomdp_latency_ms
        )
    )

    results[
        "personalized_pomdp"
    ] = {

        "experiment_log_id":
            pomdp_log.id,

        "teaching_decision": {

            "strategy":
                pomdp_plan.get(
                    "strategy"
                ),

            "hint_level":
                pomdp_plan.get(
                    "hint_level"
                ),

            "answer_policy":
                pomdp_plan.get(
                    "answer_policy"
                ),

            "decision_source":
                pomdp_plan.get(
                    "decision_source"
                ),

            "most_likely_state":
                (
                    pomdp_plan
                    .get(
                        "pomdp",
                        {}
                    )
                    .get(
                        "most_likely_state"
                    )
                ),

            "belief_entropy":
                (
                    pomdp_plan
                    .get(
                        "pomdp",
                        {}
                    )
                    .get(
                        "belief_entropy"
                    )
                )
        },

        "tutor_response":
            pomdp_response,

        "performance": {

            "llm_used":
                pomdp_llm_used,

            "prompt_tokens":
                pomdp_metadata.get(
                    "prompt_tokens"
                ),

            "completion_tokens":
                pomdp_metadata.get(
                    "completion_tokens"
                ),

            "total_tokens":
                pomdp_metadata.get(
                    "total_tokens"
                ),

            "llm_latency_ms":
                pomdp_metadata.get(
                    "latency_ms"
                ),

            "total_latency_ms":
                round(
                    pomdp_latency_ms,
                    2
                )
        }
    }
    # Automatic Evaluation

    evaluation = (
        evaluation_service
        .evaluate_comparison(

            results=
                results,

            teacher_max_hint_level=
                assignment.max_hint_level,

            student_code=
                request.code,

            misconception=
                diagnosis.get(
                    "misconception"
                )
        )
    )

    # Final Comparison Response

    return {

        "comparison": {

            "student": {

                "id":
                    student.id,

                "name":
                    student.name
            },

            "assignment": {

                "id":
                    assignment.id,

                "title":
                    assignment.title
            },

            "shared_input": {

                "pause_time":
                    request.pause_time,

                "error_count":
                    request.error_count,

                "confusion_level":
                    round(
                        confusion_level,
                        3
                    ),

                "misconception":
                    diagnosis.get(
                        "misconception"
                    ),

                "mastery_score":
                    mastery_score
            },

            "important_note":
                (
                    "All methods received the same interaction "
                    "snapshot. Comparison mode did not update "
                    "learner mastery, memory, or dynamic profile."
                )
        },

        "results":
            results,

        "evaluation":
            evaluation
    }
# Evaluation Summary
#
# Research summary for controlled baseline comparisons only.
#
# Important:
#
# Normal /tutor/plan:
#     episode_id != None
#
# Baseline /tutor/compare:
#     episode_id == None
#
# Therefore we can cleanly exclude ordinary tutoring sessions
# from controlled comparison statistics.

@router.get(
    "/evaluation-summary"
)
def get_evaluation_summary(

    limit: int = 1000,

    db: Session = Depends(
        get_db
    )
):

    # Safe Limit

    safe_limit = max(
        1,
        min(
            limit,
            5000
        )
    )

    # Controlled Comparison Logs Only

    logs = (

        db.query(
            models.ExperimentLog
        )

        # Comparison experiments use episode_id = NULL.
        #
        # This prevents normal /tutor/plan personalized
        # tutoring runs from contaminating baseline results.

        .filter(
            models.ExperimentLog.episode_id
            .is_(None)
        )

        # Only recognized comparison variants.

        .filter(
            models.ExperimentLog.model_variant
            .in_(
                [
                    "rule_based",
                    "single_llm",
                    "personalized_pomdp"
                ]
            )
        )

        .order_by(
            models.ExperimentLog.created_at.desc()
        )

        .limit(
            safe_limit
        )

        .all()
    )

    # Aggregate Evaluation

    summary = (
        evaluation_service
        .summarize_logs(
            logs
        )
    )

    # Experiment Metadata

    summary[
        "experiment_scope"
    ] = (
        "baseline_comparison_only"
    )

    summary[
        "included_logs"
    ] = (
        len(
            logs
        )
    )

    # Check Whether Comparison Is Balanced

    variant_counts = {

        "rule_based":
            0,

        "single_llm":
            0,

        "personalized_pomdp":
            0
    }

    for log in logs:

        if (
            log.model_variant
            in
            variant_counts
        ):

            variant_counts[
                log.model_variant
            ] += 1

    summary[
        "variant_counts"
    ] = (
        variant_counts
    )

    counts = list(
        variant_counts.values()
    )

    balanced = (
        len(
            set(
                counts
            )
        )
        ==
        1
    )

    summary[
        "balanced_comparison"
    ] = (
        balanced
    )

    if balanced:

        summary[
            "balance_note"
        ] = (
            "All baseline methods currently have the same "
            "number of controlled comparison samples."
        )

    else:

        summary[
            "balance_note"
        ] = (
            "Baseline sample counts are currently unequal. "
            "Use matched /tutor/compare runs before drawing "
            "research conclusions."
        )

    return summary