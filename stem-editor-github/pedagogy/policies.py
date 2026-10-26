"""
Pedagogical policy rules.

Policies constrain the decision made by the Pedagogy Planner.

For example:
- a teacher may limit the maximum hint level;
- hint levels must stay inside the valid range;
- instructional autonomy should be respected.
"""


MIN_HINT_LEVEL = 0
MAX_HINT_LEVEL = 5


def clamp_hint_level(
    hint_level: int
) -> int:
    """
    Keep hint level inside the valid system range.
    """

    return max(
        MIN_HINT_LEVEL,
        min(
            MAX_HINT_LEVEL,
            int(hint_level)
        )
    )


def apply_teacher_hint_policy(
    pedagogy_plan: dict,
    teacher_max_hint_level: int
) -> dict:
    """
    Apply the teacher's maximum allowed hint level.

    The AI tutor is not allowed to exceed this limit.
    """

    plan = pedagogy_plan.copy()

    requested_level = clamp_hint_level(
        plan.get(
            "hint_level",
            0
        )
    )

    teacher_limit = clamp_hint_level(
        teacher_max_hint_level
    )


    plan["requested_hint_level"] = (
        requested_level
    )

    plan["teacher_max_hint_level"] = (
        teacher_limit
    )


    # AI originally wanted more help than
    # the teacher allows.
    if requested_level > teacher_limit:

        plan["hint_level"] = (
            teacher_limit
        )

        plan[
            "teacher_policy_limited"
        ] = True

        original_reason = (
            plan.get(
                "reason",
                ""
            )
        )

        plan["reason"] = (
            f"{original_reason} "
            f"The requested hint level "
            f"({requested_level}) was limited "
            f"to {teacher_limit} by instructor policy."
        )

    else:

        plan["hint_level"] = (
            requested_level
        )

        plan[
            "teacher_policy_limited"
        ] = False


    return plan


def apply_learning_preference_policy(
    pedagogy_plan: dict,
    learning_preference: str
) -> dict:
    """
    Slightly adapt how the pedagogical strategy
    should be delivered.

    This policy does NOT override teacher restrictions.
    """

    plan = pedagogy_plan.copy()


    if learning_preference == "direct":

        plan["delivery_style"] = (
            "concise_and_explicit"
        )


    elif learning_preference == "guided":

        plan["delivery_style"] = (
            "guided_step_by_step"
        )


    elif learning_preference == "challenge_seeking":

        plan["delivery_style"] = (
            "challenge_oriented"
        )


    else:

        plan["delivery_style"] = (
            "balanced"
        )


    return plan