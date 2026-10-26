from datetime import datetime

from database import models


# Helpers

def clamp(
    value,
    minimum=0.0,
    maximum=1.0,
    default=0.5
):

    try:
        value = float(value)

    except (
        TypeError,
        ValueError
    ):
        value = default

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# Create / Get Dynamic Profile

def get_or_create_dynamic_profile(
    db,
    student_id: int
):

    profile = (
        db.query(
            models.StudentDynamicProfile
        )
        .filter(
            models.StudentDynamicProfile.student_id
            ==
            student_id
        )
        .first()
    )

    if profile:
        return profile

    profile = models.StudentDynamicProfile(

        student_id=student_id,

        persistence=0.5,

        help_seeking_tendency=0.5,

        directness_preference=0.5,

        challenge_tolerance=0.5,

        reflective_tendency=0.5,

        learning_preference="guided",

        personality_tendency="balanced",

        profile_confidence=0.0,

        evidence_count=0
    )

    db.add(
        profile
    )

    db.commit()

    db.refresh(
        profile
    )

    return profile


# Dynamic Update

def update_dynamic_profile(
    db,
    dynamic_profile,
    analysis: dict
):

    if not analysis:
        return dynamic_profile

    evidence_confidence = clamp(
        analysis.get(
            "confidence",
            0.1
        ),
        default=0.1
    )

    evidence_count = int(
        dynamic_profile.evidence_count
        or 0
    )

    # Conservative Update Rate

    stability = max(
        0.35,
        1.0
        /
        (
            1.0
            +
            0.08
            *
            evidence_count
        )
    )

    alpha = (
        0.20
        *
        evidence_confidence
        *
        stability
    )

    alpha = max(
        0.01,
        min(
            0.20,
            alpha
        )
    )

    def smooth(
        old_value,
        new_value
    ):

        old_value = clamp(
            old_value
        )

        new_value = clamp(
            new_value
        )

        updated = (
            old_value
            *
            (
                1.0
                -
                alpha
            )
            +
            new_value
            *
            alpha
        )

        return round(
            clamp(updated),
            3
        )

    # Continuous Traits

    dynamic_profile.persistence = smooth(
        dynamic_profile.persistence,
        analysis.get(
            "persistence",
            dynamic_profile.persistence
        )
    )

    dynamic_profile.help_seeking_tendency = smooth(
        dynamic_profile.help_seeking_tendency,
        analysis.get(
            "help_seeking_tendency",
            dynamic_profile.help_seeking_tendency
        )
    )

    dynamic_profile.directness_preference = smooth(
        dynamic_profile.directness_preference,
        analysis.get(
            "directness_preference",
            dynamic_profile.directness_preference
        )
    )

    dynamic_profile.challenge_tolerance = smooth(
        dynamic_profile.challenge_tolerance,
        analysis.get(
            "challenge_tolerance",
            dynamic_profile.challenge_tolerance
        )
    )

    dynamic_profile.reflective_tendency = smooth(
        dynamic_profile.reflective_tendency,
        analysis.get(
            "reflective_tendency",
            dynamic_profile.reflective_tendency
        )
    )

    # Categorical Learning Preference

    preference = analysis.get(
        "learning_preference"
    )

    allowed = {
        "guided",
        "direct",
        "challenge_seeking",
        "balanced"
    }

    if (
        preference in allowed
        and
        evidence_confidence >= 0.55
        and
        evidence_count >= 2
    ):

        dynamic_profile.learning_preference = (
            preference
        )

    # Educational Style Summary
    #
    # We reuse personality_tendency in the current DB schema,
    # but interpret it as a NON-CLINICAL learning-style label.

    style_summary = analysis.get(
        "learning_style_summary"
    )

    if (
        isinstance(
            style_summary,
            str
        )
        and
        style_summary.strip()
        and
        evidence_confidence >= 0.55
        and
        evidence_count >= 2
    ):

        dynamic_profile.personality_tendency = (
            style_summary.strip()[:100]
        )

    # Accumulate Profile Confidence

    old_confidence = clamp(
        dynamic_profile.profile_confidence,
        default=0.0
    )

    new_confidence = (
        old_confidence
        +
        0.10
        *
        evidence_confidence
        *
        (
            1.0
            -
            old_confidence
        )
    )

    dynamic_profile.profile_confidence = round(
        clamp(
            new_confidence,
            default=0.0
        ),
        3
    )

    dynamic_profile.evidence_count = (
        evidence_count
        +
        1
    )

    dynamic_profile.updated_at = (
        datetime.utcnow()
    )

    db.commit()

    db.refresh(
        dynamic_profile
    )

    return dynamic_profile


# Synchronize Existing Stable Profile

def synchronize_stable_profile(
    db,
    stable_profile,
    dynamic_profile
):

    if not stable_profile:
        return None

    evidence_count = int(
        dynamic_profile.evidence_count
        or 0
    )

    confidence = clamp(
        dynamic_profile.profile_confidence,
        default=0.0
    )

    # Numerical preference can slowly influence stable profile.
    if evidence_count >= 4:

        old_challenge = clamp(
            stable_profile.challenge_tolerance
        )

        observed_challenge = clamp(
            dynamic_profile.challenge_tolerance
        )

        stable_profile.challenge_tolerance = round(
            0.90
            *
            old_challenge
            +
            0.10
            *
            observed_challenge,
            3
        )

    # Learning preference needs stronger evidence.
    if (
        evidence_count >= 5
        and
        confidence >= 0.30
    ):

        preference = (
            dynamic_profile.learning_preference
        )

        if preference in {
            "guided",
            "direct",
            "challenge_seeking",
            "balanced"
        }:

            stable_profile.learning_preference = (
                preference
            )

    db.commit()

    db.refresh(
        stable_profile
    )

    return stable_profile


# Serialize Dynamic Profile

def dynamic_profile_to_dict(
    profile
):

    if not profile:
        return {}

    return {

        "persistence":
            profile.persistence,

        "help_seeking_tendency":
            profile.help_seeking_tendency,

        "directness_preference":
            profile.directness_preference,

        "challenge_tolerance":
            profile.challenge_tolerance,

        "reflective_tendency":
            profile.reflective_tendency,

        "learning_preference":
            profile.learning_preference,

        "learning_style_summary":
            profile.personality_tendency,

        "profile_confidence":
            profile.profile_confidence,

        "evidence_count":
            profile.evidence_count
    }