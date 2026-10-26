import json

from services.llm_service import llm_service


class LearnerAnalyzer:
    """
    Learner profiling component.

    Important design principle:

    The learner profile is a longitudinal model.

    Therefore, expensive LLM-based learner analysis should NOT
    be executed on every single interaction.

    The analyzer is triggered periodically or when meaningful
    behavioral changes occur.
    """

    ALLOWED_LEARNING_PREFERENCES = {
        "guided",
        "direct",
        "challenge_seeking",
        "balanced"
    }

    # Should We Run LLM Analysis?

    def should_analyze(
        self,
        interaction_number: int,
        current_evidence: dict,
        recent_episodes: list | None = None
    ) -> tuple[bool, str]:
        """
        Decide whether LLM learner analysis should run.

        Returns:
            (True/False, reason)
        """

        recent_episodes = (
            recent_episodes
            or []
        )

        # Rule 1:
        # First interaction should always be analyzed.

        if interaction_number <= 1:

            return (
                True,
                "initial_learner_analysis"
            )

        # Rule 2:
        # Periodic profile refresh.
        #
        # Analyze every 3 interactions:
        #
        # 1, 4, 7, 10, ...

        if (
            (
                interaction_number
                -
                1
            )
            %
            3
            ==
            0
        ):

            return (
                True,
                "periodic_profile_refresh"
            )

        # Current evidence

        behavior = (
            current_evidence.get(
                "behavior",
                {}
            )
            or
            {}
        )

        diagnosis = (
            current_evidence.get(
                "diagnosis",
                {}
            )
            or
            {}
        )

        current_confusion = self._clamp(
            behavior.get(
                "confusion_level",
                0.5
            )
        )

        current_misconception = (
            diagnosis.get(
                "misconception"
            )
        )

        # No history → keep current profile.

        if not recent_episodes:

            return (
                False,
                "insufficient_change_evidence"
            )

        last_episode = (
            recent_episodes[-1]
        )

        previous_confusion = self._clamp(
            last_episode.get(
                "confusion_level",
                0.5
            )
        )

        previous_misconception = (
            last_episode.get(
                "misconception"
            )
        )

        # Rule 3:
        # Misconception changed.
        #
        # This is meaningful new evidence.

        if (
            current_misconception
            !=
            previous_misconception
        ):

            return (
                True,
                "misconception_changed"
            )

        # Rule 4:
        # Large confusion change.
        #
        # Example:
        # 0.30 → 0.70

        confusion_change = abs(
            current_confusion
            -
            previous_confusion
        )

        if confusion_change >= 0.30:

            return (
                True,
                "large_confusion_change"
            )

        # Otherwise:
        # reuse accumulated learner profile.

        return (
            False,
            "reuse_existing_profile"
        )

    # Public LLM Analysis

    def analyze(
        self,
        current_evidence: dict,
        recent_episodes: list | None = None,
        current_profile: dict | None = None
    ) -> dict:

        recent_episodes = (
            recent_episodes
            or []
        )

        current_profile = (
            current_profile
            or {}
        )

        prompt = (
            self._build_prompt(

                current_evidence=
                    current_evidence,

                recent_episodes=
                    recent_episodes,

                current_profile=
                    current_profile
            )
        )

        result = (
            llm_service
            .generate_json(

                prompt=
                    prompt,

                temperature=
                    0.1,

                max_tokens=
                    500
            )
        )

        if not result:

            fallback = (
                self._fallback_analysis(
                    current_evidence
                )
            )

            fallback[
                "analysis_source"
            ] = "local_fallback"

            return fallback

        sanitized = (
            self._sanitize(
                result
            )
        )

        sanitized[
            "analysis_source"
        ] = "llm"

        return sanitized

    # Cached Profile Representation

    def build_cached_analysis(
        self,
        current_profile: dict,
        reason: str
    ) -> dict:
        """
        Return the existing learner profile without invoking Qwen.

        IMPORTANT:
        This is NOT new evidence and should therefore NOT be fed
        back into update_dynamic_profile().
        """

        return {

            "persistence":
                self._clamp(
                    current_profile.get(
                        "persistence",
                        0.5
                    )
                ),

            "help_seeking_tendency":
                self._clamp(
                    current_profile.get(
                        "help_seeking_tendency",
                        0.5
                    )
                ),

            "directness_preference":
                self._clamp(
                    current_profile.get(
                        "directness_preference",
                        0.5
                    )
                ),

            "challenge_tolerance":
                self._clamp(
                    current_profile.get(
                        "challenge_tolerance",
                        0.5
                    )
                ),

            "reflective_tendency":
                self._clamp(
                    current_profile.get(
                        "reflective_tendency",
                        0.5
                    )
                ),

            "learning_preference":
                current_profile.get(
                    "learning_preference",
                    "guided"
                ),

            "learning_style_summary":
                current_profile.get(
                    "learning_style_summary",
                    "balanced"
                ),

            "confidence":
                self._clamp(
                    current_profile.get(
                        "profile_confidence",
                        0.0
                    ),
                    default=0.0
                ),

            "evidence_summary":
                (
                    "Existing longitudinal learner profile reused; "
                    "no meaningful new evidence required LLM re-analysis."
                ),

            "analysis_source":
                "cached_profile",

            "analysis_trigger":
                reason
        }

    # Prompt

    def _build_prompt(
        self,
        current_evidence: dict,
        recent_episodes: list,
        current_profile: dict
    ) -> str:

        current_json = json.dumps(
            current_evidence,
            ensure_ascii=False,
            indent=2
        )

        # Only send latest 3 episodes.
        # This reduces prompt token usage.
        history_json = json.dumps(
            recent_episodes[-3:],
            ensure_ascii=False,
            indent=2
        )

        profile_json = json.dumps(
            current_profile,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are the learner-analysis component of an adaptive
programming tutoring system.

Estimate only EDUCATIONALLY RELEVANT learner tendencies.

Do not make clinical, demographic, sensitive, or psychological
diagnostic claims.

Temporary state must not automatically become stable trait.


==================================================
CURRENT LONG-TERM PROFILE
==================================================

{profile_json}


==================================================
RECENT LEARNING HISTORY
==================================================

{history_json}


==================================================
CURRENT EVIDENCE
==================================================

{current_json}


==================================================
ESTIMATE
==================================================

All numerical values must be between 0.0 and 1.0.

persistence:
Likelihood that the learner continues attempting difficult work.

help_seeking_tendency:
Current tendency to benefit from instructional support.

directness_preference:
Preference for explicit instruction versus exploratory guidance.

challenge_tolerance:
Current tolerance for instructional challenge.

reflective_tendency:
Tendency toward deliberate reasoning rather than immediate
trial-and-error.


learning_preference must be exactly one of:

guided
direct
challenge_seeking
balanced


learning_style_summary:

Use a short NON-CLINICAL educational description such as:

"guided and reflective"
"independent and challenge-seeking"
"currently support-seeking"
"balanced"


confidence:

Strength of evidence for this analysis.

Use low confidence when evidence is limited.


==================================================
IMPORTANT
==================================================

1. Confusion is temporary.

2. Long pauses may have multiple explanations.

3. Errors do not automatically indicate low ability.

4. Repeated patterns are stronger than a single event.

5. Do not radically change a learner model from one interaction.

6. Distinguish current instructional needs from stable tendencies.

7. Use only supplied evidence.


==================================================
OUTPUT JSON ONLY
==================================================

{{
  "persistence": 0.5,
  "help_seeking_tendency": 0.5,
  "directness_preference": 0.5,
  "challenge_tolerance": 0.5,
  "reflective_tendency": 0.5,
  "learning_preference": "guided",
  "learning_style_summary": "balanced",
  "confidence": 0.2,
  "evidence_summary": "brief evidence-based explanation"
}}
"""

        return prompt.strip()

    # Sanitization

    def _sanitize(
        self,
        result: dict
    ) -> dict:

        preference = (
            result.get(
                "learning_preference",
                "guided"
            )
        )

        if (
            preference
            not in
            self.ALLOWED_LEARNING_PREFERENCES
        ):

            preference = (
                "guided"
            )

        style = (
            result.get(
                "learning_style_summary",
                "balanced"
            )
        )

        if not isinstance(
            style,
            str
        ):

            style = (
                "balanced"
            )

        summary = (
            result.get(
                "evidence_summary",
                ""
            )
        )

        if not isinstance(
            summary,
            str
        ):

            summary = ""

        return {

            "persistence":
                self._clamp(
                    result.get(
                        "persistence"
                    )
                ),

            "help_seeking_tendency":
                self._clamp(
                    result.get(
                        "help_seeking_tendency"
                    )
                ),

            "directness_preference":
                self._clamp(
                    result.get(
                        "directness_preference"
                    )
                ),

            "challenge_tolerance":
                self._clamp(
                    result.get(
                        "challenge_tolerance"
                    )
                ),

            "reflective_tendency":
                self._clamp(
                    result.get(
                        "reflective_tendency"
                    )
                ),

            "learning_preference":
                preference,

            "learning_style_summary":
                style.strip()[:100],

            "confidence":
                self._clamp(
                    result.get(
                        "confidence",
                        0.15
                    ),
                    default=0.15
                ),

            "evidence_summary":
                summary.strip()[:500]
        }

    # Conservative Local Fallback

    def _fallback_analysis(
        self,
        evidence: dict
    ) -> dict:

        behavior = (
            evidence.get(
                "behavior",
                {}
            )
            or
            {}
        )

        confusion = self._clamp(
            behavior.get(
                "confusion_level",
                0.5
            )
        )

        pause_time = self._number(
            behavior.get(
                "pause_time",
                0.0
            ),
            0.0
        )

        error_count = int(
            self._number(
                behavior.get(
                    "error_count",
                    0
                ),
                0
            )
        )

        challenge = 0.5

        help_seeking = 0.5

        directness = 0.5

        if confusion >= 0.75:

            challenge = 0.35

            help_seeking = 0.62

            directness = 0.60

        elif confusion <= 0.25:

            challenge = 0.62

            help_seeking = 0.42

        reflective = (
            0.58
            if pause_time >= 4
            else
            0.50
        )

        if (
            confusion >= 0.70
            and
            error_count >= 3
        ):

            help_seeking = 0.65

        return {

            "persistence":
                0.5,

            "help_seeking_tendency":
                help_seeking,

            "directness_preference":
                directness,

            "challenge_tolerance":
                challenge,

            "reflective_tendency":
                reflective,

            "learning_preference":
                "guided",

            "learning_style_summary":
                "balanced",

            "confidence":
                0.08,

            "evidence_summary":
                (
                    "Low-confidence local fallback because "
                    "LLM learner analysis was unavailable."
                )
        }

    # Helpers

    def _clamp(
        self,
        value,
        default=0.5
    ) -> float:

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            value = default

        return round(
            max(
                0.0,
                min(
                    1.0,
                    value
                )
            ),
            3
        )

    def _number(
        self,
        value,
        default=0.0
    ) -> float:

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default