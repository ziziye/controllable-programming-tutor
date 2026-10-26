from pedagogy.belief_state import (
    STATES,
    BeliefStateEstimator
)

from pedagogy.reward import (
    RewardModel,
    ACTION_HINT_LEVELS
)


# Actions

ACTIONS = (
    "challenge",
    "socratic",
    "scaffolding",
    "direct_explanation"
)


# POMDP Planner

class POMDPPlanner:
    """
    Lightweight interpretable POMDP planner.

    Components:

        S:
            hidden student knowledge/confusion state

        A:
            pedagogical intervention

        P:
            heuristic state transition model

        R:
            educational utility

        O:
            observations from mastery, fuzzy confusion,
            diagnostics, and memory

        gamma:
            future-state discount factor

    This is deliberately interpretable for the research
    prototype. Transition/reward parameters can later
    be learned from empirical interaction data.
    """

    def __init__(
        self,
        gamma: float = 0.75
    ):

        self.gamma = (
            gamma
        )

        self.belief_estimator = (
            BeliefStateEstimator()
        )

        self.reward_model = (
            RewardModel()
        )


    # Select Action

    def select_action(
        self,
        mastery_score: float,
        confusion_level: float,
        diagnosis: dict,
        dynamic_profile: dict | None = None,
        recent_episodes: list | None = None,
        teacher_max_hint_level: int = 5
    ) -> dict:

        dynamic_profile = (
            dynamic_profile
            or {}
        )

        recent_episodes = (
            recent_episodes
            or []
        )


        # 1. Observation → Belief State

        belief = (
            self.belief_estimator
            .estimate(

                mastery_score=
                    mastery_score,

                confusion_level=
                    confusion_level,

                diagnosis=
                    diagnosis,

                recent_episodes=
                    recent_episodes
            )
        )


        most_likely_state = (
            self.belief_estimator
            .most_likely_state(
                belief
            )
        )


        uncertainty = (
            self.belief_estimator
            .entropy(
                belief
            )
        )


        # 2. Teacher Policy
        #
        # Invalid actions are removed BEFORE optimization.

        allowed_actions = []


        for action in ACTIONS:

            hint_level = (
                ACTION_HINT_LEVELS[
                    action
                ]
            )


            if hint_level <= teacher_max_hint_level:

                allowed_actions.append(
                    action
                )


        # Safety fallback
        if not allowed_actions:

            allowed_actions = [
                "challenge"
            ]


        # 3. Expected Utility

        action_scores = {}


        for action in allowed_actions:

            score = (
                self._expected_action_value(

                    belief=
                        belief,

                    action=
                        action,

                    dynamic_profile=
                        dynamic_profile
                )
            )


            action_scores[
                action
            ] = round(
                score,
                6
            )


        # 4. Choose Optimal Action

        best_action = max(
            action_scores,
            key=action_scores.get
        )


        return {

            "action":
                best_action,

            "hint_level":
                ACTION_HINT_LEVELS[
                    best_action
                ],

            "most_likely_state":
                most_likely_state,

            "belief_state":
                belief,

            "belief_entropy":
                uncertainty,

            "action_scores":
                action_scores,

            "allowed_actions":
                allowed_actions,

            "gamma":
                self.gamma,

            "reason":
                self._build_reason(

                    action=
                        best_action,

                    state=
                        most_likely_state,

                    belief=
                        belief,

                    uncertainty=
                        uncertainty
                )
        }


    # Expected Action Value

    def _expected_action_value(
        self,
        belief: dict,
        action: str,
        dynamic_profile: dict
    ) -> float:

        expected_value = 0.0


        for (
            state,
            state_probability
        ) in belief.items():

            immediate_reward = (
                self.reward_model.reward(

                    state=
                        state,

                    action=
                        action,

                    dynamic_profile=
                        dynamic_profile
                )
            )


            transitions = (
                self._transition_distribution(

                    state=
                        state,

                    action=
                        action
                )
            )


            future_value = 0.0


            for (
                future_state,
                transition_probability
            ) in transitions.items():

                future_value += (
                    transition_probability
                    *
                    self.reward_model
                    .state_value(
                        future_state
                    )
                )


            state_action_value = (

                immediate_reward

                +

                self.gamma
                *
                future_value
            )


            expected_value += (

                state_probability
                *
                state_action_value
            )


        return expected_value


    # Transition Model P(s' | s, a)

    def _transition_distribution(
        self,
        state: str,
        action: str
    ) -> dict:

        knowledge, confusion = (
            state.split(
                "_",
                1
            )
        )


        knowledge_transition = (
            self._knowledge_transition(
                knowledge,
                action
            )
        )


        confusion_transition = (
            self._confusion_transition(
                confusion,
                action
            )
        )


        result = {}


        for (
            next_knowledge,
            knowledge_probability
        ) in knowledge_transition.items():

            for (
                next_confusion,
                confusion_probability
            ) in confusion_transition.items():

                next_state = (
                    f"{next_knowledge}_"
                    f"{next_confusion}"
                )


                result[
                    next_state
                ] = (

                    knowledge_probability
                    *
                    confusion_probability
                )


        return result


    # Knowledge Transition

    def _knowledge_transition(
        self,
        knowledge: str,
        action: str
    ) -> dict:

        # Low Knowledge

        if knowledge == "low":

            if action == "direct_explanation":

                return {
                    "low": 0.58,
                    "medium": 0.40,
                    "high": 0.02
                }


            if action == "scaffolding":

                return {
                    "low": 0.55,
                    "medium": 0.43,
                    "high": 0.02
                }


            if action == "socratic":

                return {
                    "low": 0.72,
                    "medium": 0.27,
                    "high": 0.01
                }


            return {
                "low": 0.84,
                "medium": 0.15,
                "high": 0.01
            }


        # Medium Knowledge

        if knowledge == "medium":

            if action == "socratic":

                return {
                    "low": 0.04,
                    "medium": 0.62,
                    "high": 0.34
                }


            if action == "scaffolding":

                return {
                    "low": 0.03,
                    "medium": 0.67,
                    "high": 0.30
                }


            if action == "challenge":

                return {
                    "low": 0.05,
                    "medium": 0.65,
                    "high": 0.30
                }


            return {
                "low": 0.03,
                "medium": 0.72,
                "high": 0.25
            }


        # High Knowledge

        if action == "challenge":

            return {
                "low": 0.01,
                "medium": 0.06,
                "high": 0.93
            }


        if action == "socratic":

            return {
                "low": 0.01,
                "medium": 0.08,
                "high": 0.91
            }


        return {
            "low": 0.01,
            "medium": 0.12,
            "high": 0.87
        }


    # Confusion Transition

    def _confusion_transition(
        self,
        confusion: str,
        action: str
    ) -> dict:

        # High Confusion

        if confusion == "high":

            if action == "direct_explanation":

                return {
                    "low": 0.25,
                    "medium": 0.55,
                    "high": 0.20
                }


            if action == "scaffolding":

                return {
                    "low": 0.20,
                    "medium": 0.58,
                    "high": 0.22
                }


            if action == "socratic":

                return {
                    "low": 0.08,
                    "medium": 0.37,
                    "high": 0.55
                }


            return {
                "low": 0.04,
                "medium": 0.21,
                "high": 0.75
            }


        # Medium Confusion

        if confusion == "medium":

            if action == "scaffolding":

                return {
                    "low": 0.45,
                    "medium": 0.47,
                    "high": 0.08
                }


            if action == "socratic":

                return {
                    "low": 0.35,
                    "medium": 0.53,
                    "high": 0.12
                }


            if action == "direct_explanation":

                return {
                    "low": 0.42,
                    "medium": 0.50,
                    "high": 0.08
                }


            return {
                "low": 0.24,
                "medium": 0.56,
                "high": 0.20
            }


        # Low Confusion

        if action == "challenge":

            return {
                "low": 0.75,
                "medium": 0.22,
                "high": 0.03
            }


        if action == "socratic":

            return {
                "low": 0.80,
                "medium": 0.18,
                "high": 0.02
            }


        return {
            "low": 0.83,
            "medium": 0.15,
            "high": 0.02
        }


    # Explain Decision

    def _build_reason(
        self,
        action: str,
        state: str,
        belief: dict,
        uncertainty: float
    ) -> str:

        state_probability = (
            belief.get(
                state,
                0.0
            )
        )


        return (
            f"POMDP selected '{action}' because the "
            f"most likely learner state is '{state}' "
            f"with probability {state_probability:.3f}. "
            f"Belief entropy is {uncertainty:.3f}. "
            f"The selected action maximized expected "
            f"pedagogical utility under the current belief state."
        )