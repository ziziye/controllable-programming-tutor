#!/usr/bin/env python3
"""Reproduce the planner parameter-sensitivity analysis reported in the manuscript.

Place this file in backend/scripts/ and run from backend/:

    python scripts/run_parameter_sweep.py

It does not use the database or call an LLM. It writes:

    experiment_results/parameter_sweep/parameter_sweep.csv
    experiment_results/parameter_sweep/parameter_sweep_summary.json

The implementation follows the manuscript's computational sweep:
  sigma  in {0.18, 0.22, 0.26}
  gamma  in {0.50, 0.75, 0.90}
  alpha  in {0.8, 1.0, 1.2}
  beta   in {0.5, 1.0, 1.5}
  lambda in {0.0, 0.1, 0.2}

Each of the 3^5 = 243 configurations is evaluated on the same nine
mastery-confusion states, producing 2187 planner decisions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path


MASTERIES = (0.2, 0.5, 0.8)
CONFUSIONS = (0.14, 0.50, 0.8444444444444442)

SIGMAS = (0.18, 0.22, 0.26)
GAMMAS = (0.50, 0.75, 0.90)
ALPHAS = (0.8, 1.0, 1.2)
BETAS = (0.5, 1.0, 1.5)
LAMBDAS = (0.0, 0.1, 0.2)

NOMINAL = {
    "sigma": 0.22,
    "gamma": 0.75,
    "alpha": 1.0,
    "beta": 1.0,
    "lambda": 0.0,
}

EXPECTED = {
    "configuration_count": 243,
    "decision_count": 2187,
    "nominal_matches": 1984,
    "fully_unchanged_configurations": 70,
    "minimum_configuration_matches": 7,
    "high_confusion_decisions": 729,
    "high_confusion_scaffolding": 729,
    "challenge_decisions": 81,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to backend/ (default: parent of scripts/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: backend/experiment_results/parameter_sweep",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Do not fail if reproduced summary differs from the manuscript values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_dir = args.backend_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else backend_dir / "experiment_results" / "parameter_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import the actual backend planner/reward code.
    sys.path.insert(0, str(backend_dir))

    from pedagogy.belief_state import (  # noqa: E402
        KNOWLEDGE_CENTERS,
        CONFUSION_CENTERS,
    )
    from pedagogy.pomdp import POMDPPlanner, ACTIONS  # noqa: E402
    from pedagogy.reward import RewardModel, ACTION_COSTS  # noqa: E402

    base_planner = POMDPPlanner()
    reward_model = RewardModel()

    def parameterized_belief(mastery: float, confusion: float, sigma: float) -> dict[str, float]:
        """Belief calculation used for the sweep, with sigma exposed as a parameter.

        History, diagnosis and learner-profile adjustments are neutral in this analysis.
        The production estimator rounds normalized masses to six decimals, so the sweep
        does the same before action-value aggregation.
        """
        raw: dict[str, float] = {}
        for knowledge, k_center in KNOWLEDGE_CENTERS.items():
            k_like = math.exp(-((mastery - k_center) ** 2) / (2.0 * sigma * sigma))
            for conf_level, c_center in CONFUSION_CENTERS.items():
                c_like = math.exp(-((confusion - c_center) ** 2) / (2.0 * sigma * sigma))
                raw[f"{knowledge}_{conf_level}"] = k_like * c_like

        total = sum(raw.values())
        return {state: round(value / total, 6) for state, value in raw.items()}

    def base_suitability(state: str, action: str) -> float:
        """Recover B(s,a) from the backend reward model under a neutral profile."""
        # For a neutral profile: RewardModel.reward = B(s,a) - C(a).
        return reward_model.reward(state, action, {}) + ACTION_COSTS[action]

    def mix_transition(row: dict[str, float], lam: float) -> dict[str, float]:
        """T_lambda = (1-lambda) T + lambda/3 for each 3-state marginal row."""
        return {key: (1.0 - lam) * value + lam / 3.0 for key, value in row.items()}

    def transition_distribution(state: str, action: str, lam: float) -> dict[str, float]:
        knowledge, confusion = state.split("_", 1)
        k_row = mix_transition(base_planner._knowledge_transition(knowledge, action), lam)
        c_row = mix_transition(base_planner._confusion_transition(confusion, action), lam)
        return {
            f"{next_k}_{next_c}": p_k * p_c
            for next_k, p_k in k_row.items()
            for next_c, p_c in c_row.items()
        }

    def choose_action(
        mastery: float,
        confusion: float,
        sigma: float,
        gamma: float,
        alpha: float,
        beta: float,
        lam: float,
    ) -> tuple[str, dict[str, float]]:
        belief = parameterized_belief(mastery, confusion, sigma)
        scores: dict[str, float] = {}

        for action in ACTIONS:
            expected_value = 0.0
            for state, state_probability in belief.items():
                # Manuscript sweep definition under a neutral learner profile:
                # R_{alpha,beta}(s,a) = alpha B(s,a) - beta C(a)
                immediate_reward = (
                    alpha * base_suitability(state, action)
                    - beta * ACTION_COSTS[action]
                )

                future_value = sum(
                    transition_probability * reward_model.state_value(future_state)
                    for future_state, transition_probability in transition_distribution(
                        state, action, lam
                    ).items()
                )

                q_value = immediate_reward + gamma * future_value
                expected_value += state_probability * q_value

            # Production planner stores scores at six decimals before max().
            scores[action] = round(expected_value, 6)

        selected = max(scores, key=scores.get)
        return selected, scores

    # Nominal policy on the nine reference states.
    nominal_actions: dict[tuple[float, float], str] = {}
    for mastery in MASTERIES:
        for confusion in CONFUSIONS:
            selected, _ = choose_action(
                mastery,
                confusion,
                NOMINAL["sigma"],
                NOMINAL["gamma"],
                NOMINAL["alpha"],
                NOMINAL["beta"],
                NOMINAL["lambda"],
            )
            nominal_actions[(mastery, confusion)] = selected

    rows: list[dict[str, object]] = []
    configuration_matches: list[int] = []
    config_id = 0

    for sigma, gamma, alpha, beta, lam in itertools.product(
        SIGMAS, GAMMAS, ALPHAS, BETAS, LAMBDAS
    ):
        config_id += 1
        matches_this_config = 0

        for mastery in MASTERIES:
            for confusion in CONFUSIONS:
                selected, scores = choose_action(
                    mastery, confusion, sigma, gamma, alpha, beta, lam
                )
                nominal = nominal_actions[(mastery, confusion)]
                is_match = selected == nominal
                matches_this_config += int(is_match)

                rows.append(
                    {
                        "configuration_id": config_id,
                        "sigma": sigma,
                        "gamma": gamma,
                        "alpha": alpha,
                        "beta": beta,
                        "lambda": lam,
                        "mastery": mastery,
                        "confusion": confusion,
                        "nominal_action": nominal,
                        "selected_action": selected,
                        "action_match": is_match,
                        "high_confusion": math.isclose(confusion, CONFUSIONS[-1]),
                        "challenge_score": scores["challenge"],
                        "socratic_score": scores["socratic"],
                        "scaffolding_score": scores["scaffolding"],
                        "direct_explanation_score": scores["direct_explanation"],
                    }
                )

        configuration_matches.append(matches_this_config)

    csv_path = output_dir / "parameter_sweep.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    nominal_matches = sum(int(row["action_match"]) for row in rows)
    high_confusion_rows = [row for row in rows if row["high_confusion"]]
    challenge_decisions = sum(row["selected_action"] == "challenge" for row in rows)

    # One-at-a-time self-agreement values used by the manuscript table.
    oat: dict[str, dict[str, object]] = {}
    parameter_sets = {
        "sigma": SIGMAS,
        "gamma": GAMMAS,
        "alpha": ALPHAS,
        "beta": BETAS,
        "lambda": LAMBDAS,
    }

    for parameter_name, tested_values in parameter_sets.items():
        agreements = []
        high_conf_scaffold = []

        for value in tested_values:
            params = dict(NOMINAL)
            params[parameter_name] = value
            agreement = 0
            high_scaffold_count = 0

            for mastery in MASTERIES:
                for confusion in CONFUSIONS:
                    selected, _ = choose_action(
                        mastery,
                        confusion,
                        params["sigma"],
                        params["gamma"],
                        params["alpha"],
                        params["beta"],
                        params["lambda"],
                    )
                    agreement += int(selected == nominal_actions[(mastery, confusion)])
                    if math.isclose(confusion, CONFUSIONS[-1]):
                        high_scaffold_count += int(selected == "scaffolding")

            agreements.append(agreement)
            high_conf_scaffold.append(high_scaffold_count)

        oat[parameter_name] = {
            "tested_values": list(tested_values),
            "agreements_out_of_9": agreements,
            "minimum_agreement": min(agreements),
            "high_confusion_scaffold_out_of_9": high_conf_scaffold,
        }

    summary = {
        "configuration_count": len(configuration_matches),
        "decision_count": len(rows),
        "nominal_matches": nominal_matches,
        "nominal_agreement_percent": round(100.0 * nominal_matches / len(rows), 2),
        "fully_unchanged_configurations": sum(x == 9 for x in configuration_matches),
        "minimum_configuration_matches": min(configuration_matches),
        "minimum_configuration_agreement_percent": round(
            100.0 * min(configuration_matches) / 9.0, 2
        ),
        "high_confusion_decisions": len(high_confusion_rows),
        "high_confusion_scaffolding": sum(
            row["selected_action"] == "scaffolding" for row in high_confusion_rows
        ),
        "challenge_decisions": challenge_decisions,
        "nominal_actions": {
            f"mastery={m},confusion={c}": action
            for (m, c), action in nominal_actions.items()
        },
        "one_at_a_time": oat,
    }

    summary_path = output_dir / "parameter_sweep_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_assert:
        for key, expected_value in EXPECTED.items():
            actual_value = summary[key]
            assert actual_value == expected_value, (
                f"Reproduction mismatch for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
