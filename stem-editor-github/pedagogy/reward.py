# Action Metadata

ACTION_HINT_LEVELS = {

    "challenge":
        0,

    "socratic":
        1,

    "scaffolding":
        2,

    "direct_explanation":
        3
}


ACTION_COSTS = {

    "challenge":
        0.02,

    "socratic":
        0.05,

    "scaffolding":
        0.10,

    "direct_explanation":
        0.16
}


# Helpers

def split_state(
    state: str
):

    knowledge, confusion = (
        state.split(
            "_",
            1
        )
    )

    return (
        knowledge,
        confusion
    )


# Reward Model

class RewardModel:
    """
    Educational reward model.

    Reward balances:

        - pedagogical appropriateness
        - learner state
        - learner preferences
        - intervention cost

    Higher reward means the action is more appropriate
    for the estimated learner state.
    """

    # Reward

    def reward(
        self,
        state: str,
        action: str,
        dynamic_profile: dict | None = None
    ) -> float:

        dynamic_profile = (
            dynamic_profile
            or {}
        )


        knowledge, confusion = (
            split_state(
                state
            )
        )


        reward = 0.0


        # Challenge

        if action == "challenge":

            if (
                knowledge == "high"
                and
                confusion == "low"
            ):
                reward += 1.20


            elif (
                knowledge == "medium"
                and
                confusion == "low"
            ):
                reward += 0.70


            elif confusion == "high":

                reward -= 1.00


            elif knowledge == "low":

                reward -= 0.65


        # Socratic

        elif action == "socratic":

            if (
                knowledge == "medium"
                and
                confusion in {
                    "low",
                    "medium"
                }
            ):
                reward += 1.15


            elif (
                knowledge == "high"
                and
                confusion != "high"
            ):
                reward += 0.90


            elif (
                knowledge == "low"
                and
                confusion == "high"
            ):
                reward -= 0.55


            else:

                reward += 0.35


        # Scaffolding

        elif action == "scaffolding":

            if (
                knowledge == "low"
                and
                confusion in {
                    "medium",
                    "high"
                }
            ):
                reward += 1.25


            elif (
                knowledge == "medium"
                and
                confusion == "high"
            ):
                reward += 1.15


            elif (
                knowledge == "medium"
                and
                confusion == "medium"
            ):
                reward += 0.90


            elif (
                knowledge == "high"
                and
                confusion == "low"
            ):
                reward -= 0.40


            else:

                reward += 0.40


        # Direct Explanation

        elif action == "direct_explanation":

            if (
                knowledge == "low"
                and
                confusion == "high"
            ):
                reward += 1.35


            elif (
                knowledge == "low"
                and
                confusion == "medium"
            ):
                reward += 0.90


            elif (
                knowledge == "medium"
                and
                confusion == "high"
            ):
                reward += 0.95


            elif (
                knowledge == "high"
                and
                confusion == "low"
            ):
                reward -= 1.00


            else:

                reward += 0.15


        # Learner Profile Adaptation

        challenge_tolerance = float(
            dynamic_profile.get(
                "challenge_tolerance",
                0.5
            )
            or 0.5
        )


        directness_preference = float(
            dynamic_profile.get(
                "directness_preference",
                0.5
            )
            or 0.5
        )


        help_seeking = float(
            dynamic_profile.get(
                "help_seeking_tendency",
                0.5
            )
            or 0.5
        )


        persistence = float(
            dynamic_profile.get(
                "persistence",
                0.5
            )
            or 0.5
        )


        # Challenge tolerance

        if action == "challenge":

            reward += (
                challenge_tolerance
                -
                0.5
            ) * 0.70


        if action == "socratic":

            reward += (
                persistence
                -
                0.5
            ) * 0.35


        # Explicit guidance preference

        if action in {
            "scaffolding",
            "direct_explanation"
        }:

            reward += (
                directness_preference
                -
                0.5
            ) * 0.45


        if action == "direct_explanation":

            reward += (
                help_seeking
                -
                0.5
            ) * 0.30


        # Computational / Instructional Cost

        reward -= ACTION_COSTS.get(
            action,
            0.10
        )


        return round(
            reward,
            6
        )


    # State Value

    def state_value(
        self,
        state: str
    ) -> float:
        """
        Simple long-term educational utility.

        High mastery is beneficial.
        High confusion carries a penalty.
        """

        knowledge, confusion = (
            split_state(
                state
            )
        )


        knowledge_value = {

            "low":
                0.20,

            "medium":
                0.60,

            "high":
                1.00

        }.get(
            knowledge,
            0.50
        )


        confusion_penalty = {

            "low":
                0.00,

            "medium":
                0.20,

            "high":
                0.50

        }.get(
            confusion,
            0.20
        )


        return (
            knowledge_value
            -
            confusion_penalty
        )