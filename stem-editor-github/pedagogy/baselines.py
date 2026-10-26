class RuleBasedBaseline:
    """
    Simple deterministic tutoring baseline.

    No:
        - learner profile reasoning
        - belief state
        - POMDP
        - LLM strategy selection

    It uses only:
        confusion
        diagnosis
        teacher hint constraint
    """

    def create_plan(
        self,
        diagnosis: dict,
        confusion_level: float,
        teacher_max_hint_level: int = 5
    ) -> dict:

        confusion = self._clamp(
            confusion_level
        )

        misconception = (
            diagnosis.get(
                "misconception"
            )
        )

        max_hint = max(
            0,
            min(
                5,
                int(
                    teacher_max_hint_level
                    or 0
                )
            )
        )

        # Rule-Based Strategy

        if max_hint <= 0:

            strategy = "challenge"
            requested_hint = 0
            answer_policy = "NO_ANSWER"

        elif confusion >= 0.75:

            strategy = "scaffolding"
            requested_hint = 2
            answer_policy = "PARTIAL_SOLUTION"

        elif confusion >= 0.45:

            strategy = "socratic"
            requested_hint = 1
            answer_policy = "HINT_ONLY"

        else:

            strategy = "challenge"
            requested_hint = 1
            answer_policy = "HINT_ONLY"

        # Simple Misconception Adjustment

        if (
            misconception
            and
            confusion >= 0.85
            and
            max_hint >= 3
        ):

            strategy = "direct_explanation"
            requested_hint = 3
            answer_policy = "DIRECT_EXPLANATION"

        # Teacher Constraint

        final_hint = min(
            requested_hint,
            max_hint
        )

        teacher_limited = (
            final_hint
            <
            requested_hint
        )

        if final_hint <= 0:

            answer_policy = "NO_ANSWER"

        elif final_hint == 1:

            answer_policy = "HINT_ONLY"

        elif final_hint == 2:

            answer_policy = "PARTIAL_SOLUTION"

        elif final_hint >= 3:

            answer_policy = "DIRECT_EXPLANATION"

        return {

            "strategy":
                strategy,

            "hint_level":
                final_hint,

            "requested_hint_level":
                requested_hint,

            "teacher_max_hint_level":
                max_hint,

            "teacher_policy_limited":
                teacher_limited,

            "answer_policy":
                answer_policy,

            "decision_source":
                "rule_based",

            "reason":
                (
                    "Strategy selected using fixed "
                    "confusion-threshold rules."
                )
        }

    def _clamp(
        self,
        value
    ):

        try:
            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):
            value = 0.5

        return max(
            0.0,
            min(
                1.0,
                value
            )
        )