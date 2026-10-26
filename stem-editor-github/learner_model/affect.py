"""
Learner affective-state model.

This module stores and updates observable learner state.

Current MVP states:
- confusion
- engagement

Confusion is currently obtained from the fuzzy system.

Engagement is estimated heuristically from:
- confusion
- pause duration
- error count

This heuristic can later be replaced by a learned model.
"""


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0
) -> float:
    """
    Clamp a numeric value into a range.
    """

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def estimate_engagement(
    confusion_level: float,
    pause_time: float,
    error_count: int
) -> float:
    """
    Estimate engagement using simple behavioral signals.

    This is an MVP heuristic, not a validated
    psychological measurement model.
    """

    confusion_level = clamp(
        float(
            confusion_level
        )
    )

    pause_time = max(
        0.0,
        min(
            10.0,
            float(
                pause_time
            )
        )
    )

    error_count = max(
        0,
        min(
            5,
            int(
                error_count
            )
        )
    )


    # Start from a moderately engaged state.
    engagement = 0.75


    # Long pauses may indicate reduced engagement
    # or difficulty.
    pause_penalty = (
        pause_time / 10.0
    ) * 0.20


    # Repeated errors can reduce engagement.
    error_penalty = (
        error_count / 5.0
    ) * 0.15


    # High confusion also reduces estimated engagement.
    confusion_penalty = (
        confusion_level
        *
        0.20
    )


    engagement = (
        engagement
        -
        pause_penalty
        -
        error_penalty
        -
        confusion_penalty
    )


    return round(
        clamp(
            engagement
        ),
        3
    )


def update_affective_state(
    db,
    profile,
    confusion_level: float,
    pause_time: float,
    error_count: int
):
    """
    Update StudentProfile affective state
    and persist it to the database.
    """

    confusion_level = clamp(
        float(
            confusion_level
        )
    )


    engagement_level = (
        estimate_engagement(
            confusion_level=
                confusion_level,

            pause_time=
                pause_time,

            error_count=
                error_count
        )
    )


    profile.current_confusion = (
        confusion_level
    )

    profile.current_engagement = (
        engagement_level
    )


    db.commit()

    db.refresh(
        profile
    )


    return {
        "confusion_level":
            round(
                confusion_level,
                3
            ),

        "engagement_level":
            engagement_level
    }