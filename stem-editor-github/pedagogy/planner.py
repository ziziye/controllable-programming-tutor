from pedagogy.strategies import (
    get_strategy
)

from pedagogy.policies import (
    apply_learning_preference_policy,
    apply_teacher_hint_policy
)

from pedagogy.pomdp import (
    POMDPPlanner
)


class PedagogyPlanner:
    """
    Personalized pedagogical planner.

    Research focus:

        learner state
            ↓
        POMDP belief/action
            ↓
        pedagogy strategy
            ↓
        answer policy

    The LLM does NOT freely choose whether to give the answer.
    """

    def __init__(self):

        self.pomdp = POMDPPlanner(
            gamma=0.75
        )

    # Public Planning Method

    def create_plan(
        self,
        diagnosis: dict,
        confusion_level: float = 0.3,
        learning_preference: str = "guided",
        teacher_max_hint_level: int = 5,
        dynamic_profile: dict | None = None,
        mastery_score: float = 0.5,
        recent_episodes: list | None = None
    ) -> dict:

        dynamic_profile = (
            dynamic_profile
            or {}
        )

        recent_episodes = (
            recent_episodes
            or []
        )

        confusion_level = self._clamp(
            confusion_level
        )

        mastery_score = self._clamp(
            mastery_score
        )

        # 1. POMDP Core Decision

        decision = (
            self.pomdp
            .select_action(

                mastery_score=
                    mastery_score,

                confusion_level=
                    confusion_level,

                diagnosis=
                    diagnosis,

                dynamic_profile=
                    dynamic_profile,

                recent_episodes=
                    recent_episodes,

                teacher_max_hint_level=
                    teacher_max_hint_level
            )
        )

        selected_action = (
            decision[
                "action"
            ]
        )

        plan = get_strategy(
            selected_action
        )

        # 2. Determine Answer Policy

        answer_policy = (
            self._choose_answer_policy(

                action=
                    selected_action,

                hint_level=
                    plan.get(
                        "hint_level",
                        1
                    ),

                confusion_level=
                    confusion_level,

                mastery_score=
                    mastery_score,

                diagnosis=
                    diagnosis,

                dynamic_profile=
                    dynamic_profile,

                recent_episodes=
                    recent_episodes,

                teacher_max_hint_level=
                    teacher_max_hint_level
            )
        )

        # 3. Learning Preference

        effective_preference = (

            dynamic_profile.get(
                "learning_preference"
            )

            or

            learning_preference

            or

            "guided"
        )

        plan = (
            apply_learning_preference_policy(

                pedagogy_plan=
                    plan,

                learning_preference=
                    effective_preference
            )
        )

        # 4. Final Instructor Constraint

        plan = (
            apply_teacher_hint_policy(

                pedagogy_plan=
                    plan,

                teacher_max_hint_level=
                    teacher_max_hint_level
            )
        )

        # 5. Metadata

        plan["answer_policy"] = (
            answer_policy
        )

        plan["confusion_level"] = round(
            confusion_level,
            3
        )

        plan["mastery_score"] = round(
            mastery_score,
            3
        )

        plan["decision_source"] = (
            "personalized_pomdp"
        )

        plan["learner_adaptation"] = {

            "learning_preference":
                effective_preference,

            "challenge_tolerance":
                dynamic_profile.get(
                    "challenge_tolerance",
                    0.5
                ),

            "directness_preference":
                dynamic_profile.get(
                    "directness_preference",
                    0.5
                ),

            "help_seeking_tendency":
                dynamic_profile.get(
                    "help_seeking_tendency",
                    0.5
                ),

            "persistence":
                dynamic_profile.get(
                    "persistence",
                    0.5
                ),

            "reflective_tendency":
                dynamic_profile.get(
                    "reflective_tendency",
                    0.5
                ),

            "profile_confidence":
                dynamic_profile.get(
                    "profile_confidence",
                    0.0
                )
        }

        plan["pomdp"] = {

            "selected_action":
                decision.get(
                    "action"
                ),

            "most_likely_state":
                decision.get(
                    "most_likely_state"
                ),

            "belief_state":
                decision.get(
                    "belief_state",
                    {}
                ),

            "belief_entropy":
                decision.get(
                    "belief_entropy"
                ),

            "action_scores":
                decision.get(
                    "action_scores",
                    {}
                ),

            "allowed_actions":
                decision.get(
                    "allowed_actions",
                    []
                ),

            "gamma":
                decision.get(
                    "gamma"
                )
        }

        plan["reason"] = (
            decision.get(
                "reason",
                "Selected using the personalized POMDP planner."
            )
        )

        return plan

    # Answer Policy

    def _choose_answer_policy(
        self,
        action: str,
        hint_level: int,
        confusion_level: float,
        mastery_score: float,
        diagnosis: dict,
        dynamic_profile: dict,
        recent_episodes: list,
        teacher_max_hint_level: int
    ) -> str:

        # Teacher explicitly permits no help.
        if teacher_max_hint_level <= 0:
            return "NO_ANSWER"

        repeated_error_count = (
            self._count_repeated_misconception(

                misconception=
                    diagnosis.get(
                        "misconception"
                    ),

                recent_episodes=
                    recent_episodes
            )
        )

        help_seeking = self._clamp(
            dynamic_profile.get(
                "help_seeking_tendency",
                0.5
            )
        )

        directness = self._clamp(
            dynamic_profile.get(
                "directness_preference",
                0.5
            )
        )

        # Challenge

        if action == "challenge":
            return "NO_ANSWER"

        # Socratic

        if action == "socratic":
            return "HINT_ONLY"

        # Scaffolding

        if action == "scaffolding":

            if teacher_max_hint_level >= 2:
                return "PARTIAL_SOLUTION"

            return "HINT_ONLY"

        # Direct Explanation

        if action == "direct_explanation":

            # Full solution is deliberately difficult to unlock.
            if (
                teacher_max_hint_level >= 5
                and
                confusion_level >= 0.85
                and
                mastery_score <= 0.30
                and
                repeated_error_count >= 3
                and
                (
                    help_seeking >= 0.60
                    or
                    directness >= 0.65
                )
            ):

                return "FULL_SOLUTION_ALLOWED"

            return "DIRECT_EXPLANATION"

        # Generic fallback

        if hint_level <= 1:
            return "HINT_ONLY"

        if hint_level == 2:
            return "PARTIAL_SOLUTION"

        return "DIRECT_EXPLANATION"

    # Repeated Misconception

    def _count_repeated_misconception(
        self,
        misconception,
        recent_episodes: list
    ) -> int:

        if not misconception:
            return 0

        count = 0

        for episode in recent_episodes:

            if (
                episode.get(
                    "misconception"
                )
                ==
                misconception
            ):
                count += 1

        return count

    # Clamp

    def _clamp(
        self,
        value
    ) -> float:

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