#!/usr/bin/env python3
"""
Local controlled-experiment runner.

This version DOES NOT call protected FastAPI endpoints.
It imports the existing backend tutoring code and database directly, so
authentication/session cookies do not interfere with research experiments.

Run from backend/:

    python scripts/run_controlled_experiments.py --repeats 1 --base-only

Then:

    python scripts/run_controlled_experiments.py --repeats 3

Important:
- System-level controlled evaluation only.
- No real-student learning-outcome claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# Make backend imports work even when launched from scripts/

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


from database.db import SessionLocal
from database import models

import api.tutor as tutor_api

from api.tutor import (
    BaselineComparisonRequest,
    compare_tutoring_methods
)


EXPERIMENT_STUDENT_EMAIL = (
    "controlled-eval@local.test"
)

EXPERIMENT_STUDENT_NAME = (
    "Controlled Evaluation Student"
)

EXPERIMENT_COURSE_TITLE = (
    "Controlled Evaluation Benchmark"
)

METHODS = [
    "rule_based",
    "single_llm",
    "personalized_pomdp"
]


# Helpers

def model_has(
    model_cls,
    field: str
) -> bool:
    return hasattr(
        model_cls,
        field
    )


def filtered_kwargs(
    model_cls,
    values: dict[str, Any]
) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if model_has(
            model_cls,
            key
        )
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]]
) -> None:
    # Defensive: recreate the result directory if it was removed or
    # became unavailable while a long experiment was running.
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8"
        )
        return

    fields = []

    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(
    values: list[Any]
) -> float | None:
    cleaned = []

    for value in values:
        if value is None:
            continue

        try:
            cleaned.append(
                float(value)
            )
        except (
            TypeError,
            ValueError
        ):
            pass

    if not cleaned:
        return None

    return (
        sum(cleaned)
        /
        len(cleaned)
    )


def pct_or_none(
    values: list[Any]
) -> float | None:
    cleaned = [
        bool(value)
        for value in values
        if value is not None
    ]

    if not cleaned:
        return None

    return (
        100.0
        *
        sum(
            1
            for value in cleaned
            if value
        )
        /
        len(cleaned)
    )


# Dedicated experiment objects

def ensure_experiment_student(
    db
):
    student = (
        db.query(
            models.Student
        )
        .filter(
            models.Student.email
            ==
            EXPERIMENT_STUDENT_EMAIL
        )
        .first()
    )

    if not student:
        student = models.Student(
            **filtered_kwargs(
                models.Student,
                {
                    "name":
                        EXPERIMENT_STUDENT_NAME,

                    "email":
                        EXPERIMENT_STUDENT_EMAIL,

                    "learning_preference":
                        "guided",

                    "personality_profile":
                        "balanced"
                }
            )
        )

        db.add(student)
        db.commit()
        db.refresh(student)

        print(
            "[OK] Created dedicated "
            f"experiment student id={student.id}"
        )
    else:
        print(
            "[OK] Reusing dedicated "
            f"experiment student id={student.id}"
        )

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

    if not profile:
        profile = models.StudentProfile(
            **filtered_kwargs(
                models.StudentProfile,
                {
                    "student_id":
                        student.id,

                    "personality":
                        "balanced",

                    "learning_preference":
                        "guided",

                    "challenge_tolerance":
                        0.5,

                    "current_confusion":
                        0.0,

                    "current_engagement":
                        0.5
                }
            )
        )

        db.add(profile)
        db.commit()
        db.refresh(profile)

        print(
            "[OK] Created experiment "
            "StudentProfile"
        )

    return student


def ensure_course(
    db
):
    course = (
        db.query(
            models.Course
        )
        .filter(
            models.Course.title
            ==
            EXPERIMENT_COURSE_TITLE
        )
        .first()
    )

    if not course:
        course = models.Course(
            **filtered_kwargs(
                models.Course,
                {
                    "title":
                        EXPERIMENT_COURSE_TITLE,

                    "description":
                        (
                            "Synthetic controlled "
                            "system-level evaluation."
                        ),

                    "programming_language":
                        "Python"
                }
            )
        )

        db.add(course)
        db.commit()
        db.refresh(course)

        print(
            "[OK] Created experiment "
            f"course id={course.id}"
        )
    else:
        print(
            "[OK] Reusing experiment "
            f"course id={course.id}"
        )

    return course


def task_concepts(
    task_key: str
) -> list[str]:
    return {
        "sum_even": [
            "loops",
            "conditionals",
            "modulo",
            "accumulation"
        ],

        "count_vowels": [
            "strings",
            "loops",
            "membership",
            "case_normalization"
        ],

        "find_max": [
            "lists",
            "iteration",
            "comparison",
            "state_tracking"
        ],

        "palindrome": [
            "strings",
            "normalization",
            "reversal",
            "boolean_logic"
        ],

        "safe_dict_get": [
            "dictionaries",
            "safe_access",
            "exceptions"
        ]
    }.get(
        task_key,
        ["general_python"]
    )


def ensure_assignments(
    db,
    course,
    tasks
):
    output = {}

    for task in tasks:
        title = (
            "[EXP] "
            +
            task["title"]
        )

        assignment = (
            db.query(
                models.Assignment
            )
            .filter(
                models.Assignment.title
                ==
                title
            )
            .first()
        )

        values = {
            "title":
                title,

            "description":
                task[
                    "learning_objective"
                ],

            "starter_code":
                task[
                    "starter_code"
                ],

            "learning_objective":
                task[
                    "learning_objective"
                ],

            "function_name":
                task[
                    "function_name"
                ],

            "language":
                "Python",

            "difficulty":
                "introductory",

            "question_type":
                "coding_task",

            "learning_objectives":
                [
                    task[
                        "learning_objective"
                    ]
                ],

            "concepts":
                task_concepts(
                    task[
                        "task_key"
                    ]
                ),

            "constraints":
                [],

            "visible_examples":
                [],

            "hidden_tests":
                task[
                    "hidden_tests"
                ],

            "reference_solution":
                task[
                    "reference_solution"
                ],

            "source":
                "controlled_experiment",

            "source_url":
                None,

            "quality_score":
                1.0,

            "is_validated":
                True,

            "max_hint_level":
                5,

            "course_id":
                course.id
        }

        if not assignment:
            assignment = (
                models.Assignment(
                    **filtered_kwargs(
                        models.Assignment,
                        values
                    )
                )
            )

            db.add(
                assignment
            )

        else:
            for key, value in (
                filtered_kwargs(
                    models.Assignment,
                    values
                )
                .items()
            ):
                setattr(
                    assignment,
                    key,
                    value
                )

        db.commit()
        db.refresh(
            assignment
        )

        output[
            task[
                "task_key"
            ]
        ] = assignment

        print(
            "[OK] Assignment",
            task[
                "task_key"
            ],
            "id=",
            assignment.id
        )

    return output


# Experiment state reset

def reset_experiment_student_state(
    db,
    student_id: int
) -> None:
    """
    Keep the dedicated experiment account but remove longitudinal
    state that could leak from a previous experiment run.

    Experiment logs are NOT deleted because they are research records.
    Learning episodes/mastery/misconceptions/dynamic profile for this
    dedicated synthetic account are reset before a new benchmark.
    """

    for model_name in [
        "LearningEpisode",
        "StudentMastery",
        "MisconceptionRecord",
        "StudentDynamicProfile"
    ]:
        model_cls = getattr(
            models,
            model_name,
            None
        )

        if (
            model_cls is not None
            and
            hasattr(
                model_cls,
                "student_id"
            )
        ):
            (
                db.query(
                    model_cls
                )
                .filter(
                    model_cls.student_id
                    ==
                    student_id
                )
                .delete(
                    synchronize_session=False
                )
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

    if profile:
        for key, value in {
            "personality":
                "balanced",
            "learning_preference":
                "guided",
            "challenge_tolerance":
                0.5,
            "current_confusion":
                0.0,
            "current_engagement":
                0.5
        }.items():
            if hasattr(
                profile,
                key
            ):
                setattr(
                    profile,
                    key,
                    value
                )

    db.commit()

    print(
        "[OK] Reset dedicated "
        "experiment learner state"
    )


# Direct tutoring call

def call_compare(
    db,
    student_id: int,
    assignment_id: int,
    code: str,
    pause_time: float,
    error_count: int
) -> dict[str, Any]:
    request = (
        BaselineComparisonRequest(
            student_id=
                student_id,

            assignment_id=
                assignment_id,

            code=
                code,

            pause_time=
                pause_time,

            error_count=
                error_count
        )
    )

    # Direct function invocation bypasses FastAPI auth,
    # while executing the SAME tutoring comparison logic.
    return (
        compare_tutoring_methods(
            request=request,
            db=db
        )
    )


# Parsing

def orm_assignment_dict(
    assignment
) -> dict[str, Any]:
    return {
        "id":
            assignment.id,

        "max_hint_level":
            getattr(
                assignment,
                "max_hint_level",
                5
            )
    }


def parse_method_row(
    response,
    scenario,
    repeat_index,
    method,
    assignment,
    phase="base"
):
    comparison = (
        response.get(
            "comparison",
            {}
        )
        or
        {}
    )

    shared = (
        comparison.get(
            "shared_input",
            {}
        )
        or
        {}
    )

    method_result = (
        response.get(
            "results",
            {}
        )
        .get(
            method,
            {}
        )
        or
        {}
    )

    decision = (
        method_result.get(
            "teaching_decision"
        )
        or
        {}
    )

    performance = (
        method_result.get(
            "performance",
            {}
        )
        or
        {}
    )

    response_text = (
        method_result.get(
            "tutor_response"
        )
        or
        ""
    )

    hint_level = (
        decision.get(
            "hint_level"
        )
    )

    max_hint = getattr(
        assignment,
        "max_hint_level",
        5
    )

    if method == "single_llm":
        structural_compliance = None
        traceable = False

    else:
        traceable = bool(
            method_result.get(
                "teaching_decision"
            )
        )

        structural_compliance = (
            None
            if hint_level is None
            else
            float(hint_level)
            <=
            float(max_hint)
        )

    return {
        "experiment_phase":
            phase,

        "scenario_id":
            scenario.get(
                "scenario_id"
            ),

        "task_key":
            scenario.get(
                "task_key"
            ),

        "scenario_name":
            scenario.get(
                "scenario_name"
            ),

        "repeat":
            repeat_index,

        "method":
            method,

        "pause_time":
            shared.get(
                "pause_time"
            ),

        "error_count":
            shared.get(
                "error_count"
            ),

        "confusion_level":
            shared.get(
                "confusion_level"
            ),

        "mastery_score":
            shared.get(
                "mastery_score"
            ),

        "actual_misconception":
            shared.get(
                "misconception"
            ),

        "expected_misconception":
            scenario.get(
                "expected_misconception"
            ),

        "expected_pedagogy":
            scenario.get(
                "expected_pedagogy"
            ),

        "strategy":
            decision.get(
                "strategy"
            ),

        "hint_level":
            hint_level,

        "answer_policy":
            decision.get(
                "answer_policy"
            ),

        "most_likely_state":
            decision.get(
                "most_likely_state"
            ),

        "belief_entropy":
            decision.get(
                "belief_entropy"
            ),

        "decision_traceable":
            traceable,

        "structural_policy_compliant":
            structural_compliance,

        "llm_used":
            performance.get(
                "llm_used"
            ),

        "prompt_tokens":
            performance.get(
                "prompt_tokens"
            ),

        "completion_tokens":
            performance.get(
                "completion_tokens"
            ),

        "total_tokens":
            performance.get(
                "total_tokens"
            ),

        "llm_latency_ms":
            performance.get(
                "llm_latency_ms"
            ),

        "total_latency_ms":
            performance.get(
                "total_latency_ms"
            ),

        "response_length_chars":
            len(
                response_text
            ),

        "tutor_response":
            response_text,

        "experiment_log_id":
            method_result.get(
                "experiment_log_id"
            ),

        "raw_evaluation":
            json.dumps(
                response.get(
                    "evaluation",
                    {}
                ),
                ensure_ascii=False,
                sort_keys=True
            )
    }


# Base benchmark

def run_base(
    db,
    data,
    student,
    assignments,
    repeats
):
    method_rows = []
    call_rows = []

    scenarios = (
        data[
            "base_scenarios"
        ]
    )

    total = (
        len(scenarios)
        *
        repeats
    )

    counter = 0

    for scenario in scenarios:
        assignment = (
            assignments[
                scenario[
                    "task_key"
                ]
            ]
        )

        for repeat_index in range(
            1,
            repeats + 1
        ):
            counter += 1

            print(
                f"[BASE {counter}/{total}] "
                f"{scenario['scenario_id']} "
                f"{scenario['task_key']} "
                f"run={repeat_index}"
            )

            response = call_compare(
                db=db,
                student_id=
                    student.id,
                assignment_id=
                    assignment.id,
                code=
                    scenario[
                        "student_code"
                    ],
                pause_time=
                    scenario[
                        "pause_time"
                    ],
                error_count=
                    scenario[
                        "error_count"
                    ]
            )

            shared = (
                response.get(
                    "comparison",
                    {}
                )
                .get(
                    "shared_input",
                    {}
                )
                or
                {}
            )

            pomdp = (
                response.get(
                    "results",
                    {}
                )
                .get(
                    "personalized_pomdp",
                    {}
                )
                or
                {}
            )

            decision = (
                pomdp.get(
                    "teaching_decision"
                )
                or
                {}
            )

            call_rows.append({
                "scenario_id":
                    scenario[
                        "scenario_id"
                    ],

                "task_key":
                    scenario[
                        "task_key"
                    ],

                "scenario_name":
                    scenario[
                        "scenario_name"
                    ],

                "repeat":
                    repeat_index,

                "pause_time":
                    shared.get(
                        "pause_time"
                    ),

                "error_count":
                    shared.get(
                        "error_count"
                    ),

                "confusion_level":
                    shared.get(
                        "confusion_level"
                    ),

                "mastery_score":
                    shared.get(
                        "mastery_score"
                    ),

                "actual_misconception":
                    shared.get(
                        "misconception"
                    ),

                "pomdp_strategy":
                    decision.get(
                        "strategy"
                    ),

                "pomdp_hint_level":
                    decision.get(
                        "hint_level"
                    ),

                "pomdp_answer_policy":
                    decision.get(
                        "answer_policy"
                    ),

                "pomdp_state":
                    decision.get(
                        "most_likely_state"
                    ),

                "pomdp_belief_entropy":
                    decision.get(
                        "belief_entropy"
                    )
            })

            for method in METHODS:
                method_rows.append(
                    parse_method_row(
                        response=
                            response,
                        scenario=
                            scenario,
                        repeat_index=
                            repeat_index,
                        method=
                            method,
                        assignment=
                            assignment
                    )
                )

            # Checkpoint after every completed compare call so a later
            # filesystem/API interruption does not discard completed work.
            checkpoint_dir = (
                getattr(
                    run_base,
                    "_checkpoint_dir",
                    None
                )
            )

            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                write_csv(
                    checkpoint_dir
                    /
                    "call_level_results.partial.csv",
                    call_rows
                )

                write_csv(
                    checkpoint_dir
                    /
                    "method_level_results.partial.csv",
                    method_rows
                )

    return (
        call_rows,
        method_rows
    )


# Sensitivity

def resolve_compare_mastery_concept(
    assignment,
    code: str
) -> tuple[str, dict]:
    """
    Resolve the exact concept used by the current /tutor/compare logic.

    Newer tutor.py versions use diagnosis['concept'].
    Older versions used diagnosis['misconception'].
    This helper mirrors whichever diagnostic path exists in the project.
    """

    diagnosis = {}

    # Newer question-type / execution-aware tutor pipeline.
    diagnose_submission = getattr(
        tutor_api,
        "diagnose_submission",
        None
    )

    if callable(
        diagnose_submission
    ):
        result = diagnose_submission(
            code=code,
            assignment=assignment
        )

        if (
            isinstance(
                result,
                tuple
            )
            and
            len(result) >= 1
        ):
            diagnosis = (
                result[0]
                or
                {}
            )

        elif isinstance(
            result,
            dict
        ):
            diagnosis = result

    # Backward-compatible tutor pipeline.
    if not diagnosis:
        diagnostic_agent = getattr(
            tutor_api,
            "diagnostic_agent",
            None
        )

        if diagnostic_agent is not None:
            try:
                diagnosis = (
                    diagnostic_agent
                    .analyze_code(
                        code
                    )
                    or
                    {}
                )
            except Exception:
                diagnosis = {}

    concept = (
        diagnosis.get(
            "concept"
        )
        or
        diagnosis.get(
            "misconception"
        )
        or
        "general_python"
    )

    return (
        str(concept),
        diagnosis
    )


def set_sensitivity_mastery(
    db,
    student_id: int,
    assignment,
    code: str,
    score: float
) -> str:
    """
    Set mastery for the EXACT concept that compare_tutoring_methods()
    will read.

    We also synchronize general_python and assignment concepts as a
    defensive fallback. This only affects the dedicated synthetic
    experiment student.
    """

    primary_concept, diagnosis = (
        resolve_compare_mastery_concept(
            assignment=
                assignment,
            code=
                code
        )
    )

    concepts = {
        "general_python",
        primary_concept
    }

    assignment_concepts = (
        getattr(
            assignment,
            "concepts",
            None
        )
        or
        []
    )

    for concept in assignment_concepts:
        concept = str(
            concept
        ).strip()

        if concept:
            concepts.add(
                concept
            )

    # Update existing records first.
    existing = (
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

    existing_by_concept = {
        str(
            record.concept
        ):
            record
        for record in existing
    }

    for concept in concepts:
        record = (
            existing_by_concept
            .get(
                concept
            )
        )

        if record is None:
            record = (
                models.StudentMastery(
                    student_id=
                        student_id,
                    concept=
                        concept,
                    mastery_score=
                        score
                )
            )

            db.add(
                record
            )

        else:
            record.mastery_score = (
                score
            )

    db.commit()

    print(
        "[STATE] mastery target",
        score,
        "compare concept=",
        primary_concept,
        "diagnosis concept=",
        diagnosis.get(
            "concept"
        ),
        "misconception=",
        diagnosis.get(
            "misconception"
        )
    )

    return primary_concept

def run_sensitivity(
    db,
    data,
    student,
    assignments
):
    assignment = (
        assignments[
            "sum_even"
        ]
    )

    task = next(
        task
        for task in data[
            "tasks"
        ]
        if task[
            "task_key"
        ]
        ==
        "sum_even"
    )

    code = task[
        "reference_solution"
    ]

    conditions = [
        (
            "low",
            1.0,
            0
        ),
        (
            "medium",
            4.5,
            2
        ),
        (
            "high",
            9.0,
            4
        )
    ]

    rows = []
    index = 0

    for mastery in [
        0.2,
        0.5,
        0.8
    ]:
        resolved_concept = (
            set_sensitivity_mastery(
                db=db,
                student_id=
                    student.id,
                assignment=
                    assignment,
                code=
                    code,
                score=
                    mastery
            )
        )

        for (
            label,
            pause,
            errors
        ) in conditions:
            index += 1

            print(
                f"[STATE {index}/9] "
                f"mastery={mastery} "
                f"condition={label}"
            )

            response = (
                call_compare(
                    db=db,
                    student_id=
                        student.id,
                    assignment_id=
                        assignment.id,
                    code=
                        code,
                    pause_time=
                        pause,
                    error_count=
                        errors
                )
            )

            shared = (
                response[
                    "comparison"
                ][
                    "shared_input"
                ]
            )

            decision = (
                response[
                    "results"
                ][
                    "personalized_pomdp"
                ]
                .get(
                    "teaching_decision"
                )
                or
                {}
            )

            actual_mastery = (
                shared.get(
                    "mastery_score"
                )
            )

            rows.append({
                "mastery_concept_used":
                    resolved_concept,

                "target_mastery":
                    mastery,

                "actual_mastery":
                    actual_mastery,

                "mastery_override_applied":
                    (
                        actual_mastery
                        is not None
                        and
                        abs(
                            float(
                                actual_mastery
                            )
                            -
                            mastery
                        )
                        <
                        1e-6
                    ),

                "behavior_condition":
                    label,

                "pause_time":
                    pause,

                "error_count":
                    errors,

                "actual_confusion":
                    shared.get(
                        "confusion_level"
                    ),

                "strategy":
                    decision.get(
                        "strategy"
                    ),

                "hint_level":
                    decision.get(
                        "hint_level"
                    ),

                "answer_policy":
                    decision.get(
                        "answer_policy"
                    ),

                "most_likely_state":
                    decision.get(
                        "most_likely_state"
                    ),

                "belief_entropy":
                    decision.get(
                        "belief_entropy"
                    )
            })

    return rows


# Hint stress

def run_hint_stress(
    db,
    data,
    student,
    assignments
):
    assignment = (
        assignments[
            "sum_even"
        ]
    )

    scenario = next(
        item
        for item in data[
            "base_scenarios"
        ]
        if (
            item[
                "task_key"
            ]
            ==
            "sum_even"
            and
            item[
                "scenario_name"
            ]
            ==
            "incomplete"
        )
    )

    original_max = getattr(
        assignment,
        "max_hint_level",
        5
    )

    rows = []

    try:
        for level in range(6):
            print(
                f"[HINT {level}/5] "
                f"teacher max={level}"
            )

            assignment.max_hint_level = (
                level
            )

            db.commit()
            db.refresh(
                assignment
            )

            response = (
                call_compare(
                    db=db,
                    student_id=
                        student.id,
                    assignment_id=
                        assignment.id,
                    code=
                        scenario[
                            "student_code"
                        ],
                    pause_time=
                        scenario[
                            "pause_time"
                        ],
                    error_count=
                        scenario[
                            "error_count"
                        ]
                )
            )

            decision = (
                response[
                    "results"
                ][
                    "personalized_pomdp"
                ]
                .get(
                    "teaching_decision"
                )
                or
                {}
            )

            hint = (
                decision.get(
                    "hint_level"
                )
            )

            rows.append({
                "teacher_max_hint_level":
                    level,

                "selected_strategy":
                    decision.get(
                        "strategy"
                    ),

                "effective_hint_level":
                    hint,

                "answer_policy":
                    decision.get(
                        "answer_policy"
                    ),

                "teacher_constraint_satisfied":
                    (
                        None
                        if hint is None
                        else
                        float(hint)
                        <=
                        float(level)
                    ),

                "most_likely_state":
                    decision.get(
                        "most_likely_state"
                    ),

                "belief_entropy":
                    decision.get(
                        "belief_entropy"
                    )
            })

    finally:
        assignment.max_hint_level = (
            original_max
        )

        db.commit()
        db.refresh(
            assignment
        )

    return rows


# Summaries

def summarize_methods(
    rows
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            row["method"]
        ].append(row)

    output = []

    for method in METHODS:
        group = grouped[
            method
        ]

        output.append({
            "method":
                method,

            "n":
                len(group),

            "decision_traceability_rate_pct":
                pct_or_none([
                    row[
                        "decision_traceable"
                    ]
                    for row in group
                ]),

            "structural_policy_compliance_pct":
                pct_or_none([
                    row[
                        "structural_policy_compliant"
                    ]
                    for row in group
                ]),

            "mean_total_latency_ms":
                mean_or_none([
                    row[
                        "total_latency_ms"
                    ]
                    for row in group
                ]),

            "mean_llm_latency_ms":
                mean_or_none([
                    row[
                        "llm_latency_ms"
                    ]
                    for row in group
                ]),

            "mean_prompt_tokens":
                mean_or_none([
                    row[
                        "prompt_tokens"
                    ]
                    for row in group
                ]),

            "mean_completion_tokens":
                mean_or_none([
                    row[
                        "completion_tokens"
                    ]
                    for row in group
                ]),

            "mean_total_tokens":
                mean_or_none([
                    row[
                        "total_tokens"
                    ]
                    for row in group
                ]),

            "mean_response_length_chars":
                mean_or_none([
                    row[
                        "response_length_chars"
                    ]
                    for row in group
                ])
        })

    return output


def strategy_distribution(
    rows
):
    pomdp_rows = [
        row
        for row in rows
        if row[
            "method"
        ]
        ==
        "personalized_pomdp"
    ]

    counts = Counter(
        row.get(
            "strategy"
        )
        or
        "None"
        for row in pomdp_rows
    )

    total = (
        len(
            pomdp_rows
        )
        or
        1
    )

    return [
        {
            "strategy":
                strategy,

            "count":
                count,

            "percentage":
                (
                    count
                    /
                    total
                    *
                    100.0
                )
        }
        for (
            strategy,
            count
        )
        in sorted(
            counts.items()
        )
    ]


# Main

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--base-only",
        action="store_true"
    )

    parser.add_argument(
        "--skip-sensitivity",
        action="store_true"
    )

    parser.add_argument(
        "--skip-hint-stress",
        action="store_true"
    )

    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help=(
            "Skip the 25-case base benchmark and run only "
            "the 9 learner-state sensitivity cases plus "
            "the 6 teacher max-hint stress cases."
        )
    )

    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help=(
            "Run only the 9 learner-state sensitivity cases. "
            "Use this to re-run the corrected mastery manipulation "
            "without repeating the base or hint-stress experiments."
        )
    )

    parser.add_argument(
        "--data",
        default=None
    )

    parser.add_argument(
        "--output-dir",
        default=None
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit(
            "--repeats must be >= 1"
        )

    data_path = (
        Path(
            args.data
        ).expanduser().resolve()
        if args.data
        else
        (
            BACKEND_DIR
            /
            "experiment_data"
            /
            "controlled_experiment_cases.json"
        )
    )

    if not data_path.exists():
        raise SystemExit(
            f"Experiment JSON not found: "
            f"{data_path}"
        )

    data = json.loads(
        data_path.read_text(
            encoding="utf-8"
        )
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else
        (
            BACKEND_DIR
            /
            "experiment_results"
            /
            timestamp
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n=== LOCAL Controlled Experiment Runner ==="
    )

    print(
        "No FastAPI authentication is used."
    )

    print(
        "Output:",
        output_dir
    )

    print(
        "Base scenarios:",
        len(
            data[
                "base_scenarios"
            ]
        )
    )

    print(
        "Repeats:",
        args.repeats
    )

    if (
        args.supplement_only
        or
        args.sensitivity_only
    ):
        print(
            "Mode: supplement-only "
            "(9 sensitivity + 6 hint-stress cases)"
        )
    else:
        print(
            "Expected base method-level rows:",
            (
                len(
                    data[
                        "base_scenarios"
                    ]
                )
                *
                args.repeats
                *
                3
            )
        )

    db = SessionLocal()

    started = time.time()

    try:
        student = (
            ensure_experiment_student(
                db
            )
        )

        reset_experiment_student_state(
            db,
            student.id
        )

        # get_or_create_dynamic_profile inside tutor comparison
        # will recreate a neutral dynamic profile as needed.

        course = ensure_course(
            db
        )

        assignments = (
            ensure_assignments(
                db,
                course,
                data[
                    "tasks"
                ]
            )
        )

        # Allow run_base() to checkpoint progress into this run folder.
        run_base._checkpoint_dir = (
            output_dir
        )

        call_rows = []
        method_rows = []
        method_summary = []
        strategies = []

        if (
            not args.supplement_only
            and
            not args.sensitivity_only
        ):
            (
                call_rows,
                method_rows
            ) = run_base(
                db=db,
                data=data,
                student=student,
                assignments=
                    assignments,
                repeats=
                    args.repeats
            )

            write_csv(
                output_dir
                /
                "call_level_results.csv",
                call_rows
            )

            write_csv(
                output_dir
                /
                "method_level_results.csv",
                method_rows
            )

            method_summary = (
                summarize_methods(
                    method_rows
                )
            )

            write_csv(
                output_dir
                /
                "method_summary.csv",
                method_summary
            )

            strategies = (
                strategy_distribution(
                    method_rows
                )
            )

            write_csv(
                output_dir
                /
                "strategy_distribution.csv",
                strategies
            )

        sensitivity = []
        hint_stress = []

        if (
            not args.base_only
            or args.supplement_only
            or args.sensitivity_only
        ):
            if (
                not
                args.skip_sensitivity
            ):
                sensitivity = (
                    run_sensitivity(
                        db=db,
                        data=data,
                        student=
                            student,
                        assignments=
                            assignments
                    )
                )

                write_csv(
                    output_dir
                    /
                    "learner_state_sensitivity.csv",
                    sensitivity
                )

            if (
                not args.skip_hint_stress
                and
                not args.sensitivity_only
            ):
                hint_stress = (
                    run_hint_stress(
                        db=db,
                        data=data,
                        student=
                            student,
                        assignments=
                            assignments
                    )
                )

                write_csv(
                    output_dir
                    /
                    "teacher_hint_stress.csv",
                    hint_stress
                )

        elapsed = (
            time.time()
            -
            started
        )

        summary = {
            "generated_at":
                datetime.now()
                .isoformat(),

            "runner_mode":
                "direct_backend_function_call",

            "authentication_bypassed":
                True,

            "important_note":
                (
                    "FastAPI authentication is bypassed "
                    "only because this is a local research "
                    "runner importing the same backend "
                    "comparison function directly."
                ),

            "base_scenarios":
                len(
                    data[
                        "base_scenarios"
                    ]
                ),

            "repeats":
                args.repeats,

            "base_compare_calls":
                len(
                    call_rows
                ),

            "base_method_level_rows":
                len(
                    method_rows
                ),

            "sensitivity_rows":
                len(
                    sensitivity
                ),

            "hint_stress_rows":
                len(
                    hint_stress
                ),

            "elapsed_seconds":
                round(
                    elapsed,
                    2
                ),

            "method_summary":
                method_summary,

            "strategy_distribution":
                strategies
        }

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        (
            output_dir
            /
            "run_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            "\n=== Finished ==="
        )

        print(
            "Base method-level rows:",
            len(
                method_rows
            )
        )

        print(
            "Results:",
            output_dir
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
