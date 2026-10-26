"""
Teaching strategy definitions.

This module does not decide which strategy should be used.
It only defines the available pedagogical strategies.

The Pedagogy Planner will select one of these strategies
according to learner state and diagnostic results.
"""


STRATEGIES = {
    "challenge": {
        "strategy": "challenge",
        "hint_level": 0,
        "instruction": (
            "Encourage the learner to continue independently "
            "and test the solution using additional examples."
        )
    },

    "socratic": {
        "strategy": "socratic",
        "hint_level": 1,
        "instruction": (
            "Use guiding questions instead of directly giving "
            "the solution."
        )
    },

    "scaffolding": {
        "strategy": "scaffolding",
        "hint_level": 2,
        "instruction": (
            "Break the problem into smaller conceptual steps "
            "and guide the learner through them."
        )
    },

    "direct_explanation": {
        "strategy": "direct_explanation",
        "hint_level": 3,
        "instruction": (
            "Provide a clearer and more explicit explanation "
            "because the learner appears highly confused."
        )
    }
}


def get_strategy(strategy_name: str) -> dict:
    """
    Return a copy of a teaching strategy.

    A copy is returned so later policy adjustments do not
    modify the global STRATEGIES dictionary.
    """

    strategy = STRATEGIES.get(
        strategy_name,
        STRATEGIES["socratic"]
    )

    return strategy.copy()


def choose_base_strategy(diagnosis: dict) -> dict:
    """
    Select the initial strategy according to the
    Diagnostic Agent result.

    This is the first stage of pedagogical planning.
    Confusion and teacher policies are handled later.
    """

    misconception = diagnosis.get(
        "misconception"
    )

    severity = diagnosis.get(
        "severity"
    )


    # No detected structural misconception

    if not misconception:
        strategy = get_strategy(
            "challenge"
        )

        strategy["reason"] = (
            "No obvious structural misconception "
            "was detected."
        )

        return strategy


    # Serious syntax problem

    if (
        misconception == "python_syntax"
        or severity == "high"
    ):
        strategy = get_strategy(
            "direct_explanation"
        )

        strategy["reason"] = (
            "A high-severity or syntax-related "
            "problem was detected."
        )

        return strategy


    # Return statement misconception

    if misconception == "return_statement":
        strategy = get_strategy(
            "scaffolding"
        )

        strategy["reason"] = (
            "The learner may need step-by-step support "
            "to distinguish output from returned values."
        )

        return strategy


    # Function definition misconception

    if misconception == "function_definition":
        strategy = get_strategy(
            "socratic"
        )

        strategy["reason"] = (
            "A guiding-question strategy may help the learner "
            "recall the structure of a Python function."
        )

        return strategy


    # Fallback

    strategy = get_strategy(
        "socratic"
    )

    strategy["reason"] = (
        "A misconception was detected, so guided "
        "questioning is used as the default strategy."
    )

    return strategy