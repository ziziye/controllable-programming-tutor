import math


# Hidden Student States

KNOWLEDGE_LEVELS = (
    "low",
    "medium",
    "high"
)

CONFUSION_LEVELS = (
    "low",
    "medium",
    "high"
)


STATES = [

    f"{knowledge}_{confusion}"

    for knowledge in KNOWLEDGE_LEVELS

    for confusion in CONFUSION_LEVELS
]


# Numerical Centers

KNOWLEDGE_CENTERS = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80
}


CONFUSION_CENTERS = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80
}


# Helpers

def clamp(
    value,
    minimum=0.0,
    maximum=1.0
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
        minimum,
        min(
            maximum,
            value
        )
    )


def gaussian_likelihood(
    observation: float,
    center: float,
    sigma: float = 0.22
) -> float:

    observation = clamp(
        observation
    )


    exponent = -(
        (
            observation
            -
            center
        )
        ** 2
    ) / (
        2
        *
        sigma
        *
        sigma
    )


    return math.exp(
        exponent
    )


def normalize(
    distribution: dict
) -> dict:

    total = sum(
        distribution.values()
    )


    if total <= 0:

        uniform = (
            1.0
            /
            len(STATES)
        )

        return {
            state: uniform
            for state in STATES
        }


    return {

        state:
            value / total

        for (
            state,
            value
        )
        in distribution.items()
    }


# Belief State Estimator

class BeliefStateEstimator:
    """
    Estimate a probability distribution over hidden
    learner states.

    Hidden state dimensions:

        Knowledge:
            low / medium / high

        Confusion:
            low / medium / high

    The estimator combines:

        - semantic mastery memory
        - fuzzy confusion
        - current diagnostic evidence
        - recent episodic history

    This produces the POMDP belief state b(s).
    """

    # Estimate

    def estimate(
        self,
        mastery_score: float,
        confusion_level: float,
        diagnosis: dict | None = None,
        recent_episodes: list | None = None
    ) -> dict:

        diagnosis = (
            diagnosis
            or {}
        )

        recent_episodes = (
            recent_episodes
            or []
        )


        mastery_score = clamp(
            mastery_score
        )

        confusion_level = clamp(
            confusion_level
        )


        # 1. Historical smoothing

        historical_mastery = (
            self._average_recent_value(
                recent_episodes,
                "mastery_score"
            )
        )


        historical_confusion = (
            self._average_recent_value(
                recent_episodes,
                "confusion_level"
            )
        )


        # Current observation receives more weight.
        if historical_mastery is not None:

            mastery_evidence = (
                0.70
                *
                mastery_score
                +
                0.30
                *
                historical_mastery
            )

        else:

            mastery_evidence = (
                mastery_score
            )


        if historical_confusion is not None:

            confusion_evidence = (
                0.75
                *
                confusion_level
                +
                0.25
                *
                historical_confusion
            )

        else:

            confusion_evidence = (
                confusion_level
            )


        # 2. Observation Likelihood O(o | s)

        belief = {}


        misconception = diagnosis.get(
            "misconception"
        )


        severity = diagnosis.get(
            "severity"
        )


        for knowledge in KNOWLEDGE_LEVELS:

            for confusion in CONFUSION_LEVELS:

                state = (
                    f"{knowledge}_{confusion}"
                )


                knowledge_likelihood = (
                    gaussian_likelihood(

                        observation=
                            mastery_evidence,

                        center=
                            KNOWLEDGE_CENTERS[
                                knowledge
                            ]
                    )
                )


                confusion_likelihood = (
                    gaussian_likelihood(

                        observation=
                            confusion_evidence,

                        center=
                            CONFUSION_CENTERS[
                                confusion
                            ]
                    )
                )


                probability = (
                    knowledge_likelihood
                    *
                    confusion_likelihood
                )


                # Diagnostic evidence

                if misconception:

                    if knowledge == "low":

                        probability *= 1.35


                    elif knowledge == "high":

                        probability *= 0.75


                if severity == "high":

                    if knowledge == "low":

                        probability *= 1.25


                    if confusion == "high":

                        probability *= 1.20


                elif severity == "medium":

                    if confusion == "medium":

                        probability *= 1.10


                belief[
                    state
                ] = probability


        belief = normalize(
            belief
        )


        return {
            state:
                round(
                    probability,
                    6
                )

            for (
                state,
                probability
            )
            in belief.items()
        }


    # Most Likely State

    def most_likely_state(
        self,
        belief: dict
    ) -> str:

        if not belief:

            return (
                "medium_medium"
            )


        return max(
            belief,
            key=belief.get
        )


    # Entropy / Uncertainty

    def entropy(
        self,
        belief: dict
    ) -> float:

        entropy_value = 0.0


        for probability in belief.values():

            if probability > 0:

                entropy_value -= (
                    probability
                    *
                    math.log(
                        probability,
                        2
                    )
                )


        return round(
            entropy_value,
            4
        )


    # History Helper

    def _average_recent_value(
        self,
        episodes: list,
        key: str
    ):

        values = []


        for episode in episodes:

            value = episode.get(
                key
            )


            if value is None:
                continue


            try:

                values.append(
                    float(
                        value
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue


        if not values:

            return None


        return (
            sum(values)
            /
            len(values)
        )