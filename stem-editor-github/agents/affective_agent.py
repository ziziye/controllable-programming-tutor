class AffectiveAgent:
    """
    Controls HOW the tutor communicates.

    It must NOT override the pedagogical strategy chosen
    by the pedagogy planner.
    """

    def choose_persona(
        self,
        confusion_level: float,
        personality: str = "balanced",
        learning_preference: str = "guided",
        dynamic_profile: dict | None = None
    ) -> dict:

        dynamic_profile = (
            dynamic_profile
            or {}
        )

        confusion = self._clamp(
            confusion_level
        )

        challenge_tolerance = self._clamp(
            dynamic_profile.get(
                "challenge_tolerance",
                0.5
            )
        )

        directness_preference = self._clamp(
            dynamic_profile.get(
                "directness_preference",
                0.5
            )
        )

        help_seeking = self._clamp(
            dynamic_profile.get(
                "help_seeking_tendency",
                0.5
            )
        )

        reflective = self._clamp(
            dynamic_profile.get(
                "reflective_tendency",
                0.5
            )
        )

        # Base Persona

        persona = {

            "tone":
                "supportive",

            "warmth":
                0.65,

            "directness":
                0.50,

            "encouragement":
                0.60,

            "challenge":
                0.45,

            "pace":
                "balanced",

            "style":
                "adaptive"
        }

        # Current Confusion

        if confusion >= 0.75:

            persona[
                "tone"
            ] = "calm_and_reassuring"

            persona[
                "warmth"
            ] = 0.88

            persona[
                "encouragement"
            ] = 0.82

            persona[
                "challenge"
            ] = 0.20

            persona[
                "pace"
            ] = "slower"

        elif confusion >= 0.45:

            persona[
                "tone"
            ] = "supportive_and_focused"

            persona[
                "warmth"
            ] = 0.72

            persona[
                "encouragement"
            ] = 0.68

            persona[
                "challenge"
            ] = 0.40

        else:

            persona[
                "tone"
            ] = "positive_and_confident"

            persona[
                "warmth"
            ] = 0.60

            persona[
                "challenge"
            ] = 0.58

        # Learner Preference Adaptation

        persona[
            "directness"
        ] = self._clamp(
            0.30
            +
            0.60
            *
            directness_preference
        )

        if (
            challenge_tolerance >= 0.70
            and
            confusion < 0.55
        ):

            persona[
                "challenge"
            ] = max(
                persona[
                    "challenge"
                ],
                0.72
            )

        if (
            help_seeking >= 0.70
            and
            confusion >= 0.50
        ):

            persona[
                "warmth"
            ] = max(
                persona[
                    "warmth"
                ],
                0.80
            )

            persona[
                "encouragement"
            ] = max(
                persona[
                    "encouragement"
                ],
                0.78
            )

        if reflective >= 0.70:

            persona[
                "pace"
            ] = "reflective"

        # Stable Profile Only As Weak Context

        if personality == "reserved":

            persona[
                "tone"
            ] = "calm_and_respectful"

        elif personality == "energetic":

            persona[
                "tone"
            ] = "energetic_and_positive"

        effective_preference = (

            dynamic_profile.get(
                "learning_preference"
            )

            or

            learning_preference
        )

        if effective_preference == "direct":

            persona[
                "directness"
            ] = max(
                persona[
                    "directness"
                ],
                0.75
            )

        elif effective_preference == "guided":

            persona[
                "style"
            ] = "guided_and_patient"

        elif effective_preference == "challenge_seeking":

            persona[
                "challenge"
            ] = max(
                persona[
                    "challenge"
                ],
                0.75
            )

            persona[
                "style"
            ] = "challenge_oriented"

        # Final Clamp

        for key in (
            "warmth",
            "directness",
            "encouragement",
            "challenge"
        ):

            persona[
                key
            ] = round(
                self._clamp(
                    persona[
                        key
                    ]
                ),
                3
            )

        return persona

    # Helper

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