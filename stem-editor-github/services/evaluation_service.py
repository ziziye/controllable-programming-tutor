import re

from collections import defaultdict


class EvaluationService:
    """
    Automatic research evaluation service.

    This evaluator separates:

    1. Objective system metrics
       - latency
       - token usage
       - structured teacher-policy compliance
       - decision traceability

    2. Heuristic content metrics
       - possible direct-answer exposure

    IMPORTANT:
    Heuristic answer exposure is NOT ground-truth pedagogical
    quality. It should later be validated using human/expert
    evaluation if used in a research paper.
    """

    # Answer Policy Strength

    POLICY_LEVEL = {

        "NO_ANSWER":
            0,

        "HINT_ONLY":
            1,

        "PARTIAL_SOLUTION":
            2,

        "DIRECT_EXPLANATION":
            3,

        "FULL_SOLUTION_ALLOWED":
            5
    }

    # Evaluate One Comparison

    def evaluate_comparison(
        self,
        results: dict,
        teacher_max_hint_level: int,
        student_code: str = "",
        misconception: str | None = None
    ) -> dict:

        evaluations = {}

        for (
            variant,
            result
        ) in results.items():

            evaluations[
                variant
            ] = (
                self.evaluate_variant(

                    variant=
                        variant,

                    result=
                        result,

                    teacher_max_hint_level=
                        teacher_max_hint_level,

                    student_code=
                        student_code,

                    misconception=
                        misconception
                )
            )

        return {

            "evaluation_type":
                "automatic_research_metrics",

            "metrics_version":
                "v1",

            "important_note":
                (
                    "Latency, token usage and structured policy "
                    "compliance are objective system measurements. "
                    "Answer-exposure detection is heuristic and "
                    "should later be validated using human or expert "
                    "ratings."
                ),

            "methods":
                evaluations
        }

    # Evaluate One Variant

    def evaluate_variant(
        self,
        variant: str,
        result: dict,
        teacher_max_hint_level: int,
        student_code: str = "",
        misconception: str | None = None
    ) -> dict:

        decision = (
            result.get(
                "teaching_decision"
            )
        )

        response = (
            result.get(
                "tutor_response",
                ""
            )
            or
            ""
        )

        performance = (
            result.get(
                "performance",
                {}
            )
            or
            {}
        )

        # Structured Policy Compliance

        policy_evaluation = (
            self._evaluate_policy_compliance(

                decision=
                    decision,

                teacher_max_hint_level=
                    teacher_max_hint_level
            )
        )

        # Answer Exposure

        exposure = (
            self._detect_answer_exposure(

                response=
                    response,

                student_code=
                    student_code,

                teacher_max_hint_level=
                    teacher_max_hint_level,

                misconception=
                    misconception
            )
        )

        # Decision Traceability

        decision_traceable = (
            isinstance(
                decision,
                dict
            )
            and
            bool(
                decision.get(
                    "decision_source"
                )
            )
        )

        # Personalization

        personalization_used = (
            variant
            ==
            "personalized_pomdp"
        )

        # POMDP

        pomdp_used = False

        if isinstance(
            decision,
            dict
        ):

            pomdp_used = (
                decision.get(
                    "decision_source"
                )
                ==
                "personalized_pomdp"
            )

        # Performance

        latency = (
            performance.get(
                "total_latency_ms"
            )
        )

        token_usage = (
            performance.get(
                "total_tokens"
            )
        )

        return {

            "structured_policy_compliance":
                policy_evaluation[
                    "compliant"
                ],

            "policy_evaluation":
                policy_evaluation,

            "possible_answer_exposure":
                exposure[
                    "exposed"
                ],

            "answer_exposure_evidence":
                exposure[
                    "reasons"
                ],

            "exact_solution_detected":
                exposure[
                    "exact_solution_detected"
                ],

            "direct_replacement_instruction":
                exposure[
                    "replacement_instruction_detected"
                ],

            "decision_traceable":
                decision_traceable,

            "personalization_used":
                personalization_used,

            "pomdp_used":
                pomdp_used,

            "total_tokens":
                token_usage,

            "total_latency_ms":
                latency,

            "response_length_chars":
                len(
                    response
                )
        }

    # Teacher Policy Compliance

    def _evaluate_policy_compliance(
        self,
        decision,
        teacher_max_hint_level: int
    ) -> dict:

        max_hint = (
            self._safe_int(
                teacher_max_hint_level,
                default=0
            )
        )

        max_hint = max(
            0,
            min(
                5,
                max_hint
            )
        )

        # Single LLM:
        #
        # No explicit structured teaching decision exists.
        #
        # Therefore compliance is not structurally measurable.

        if not isinstance(
            decision,
            dict
        ):

            return {

                "measurable":
                    False,

                "compliant":
                    None,

                "reason":
                    (
                        "No explicit structured teaching decision "
                        "was available for this method."
                    )
            }

        hint_level = (
            self._safe_int(
                decision.get(
                    "hint_level"
                ),
                default=0
            )
        )

        answer_policy = (
            decision.get(
                "answer_policy"
            )
        )

        policy_level = (
            self.POLICY_LEVEL.get(
                answer_policy
            )
        )

        # Hint-Level Constraint

        hint_compliant = (
            hint_level
            <=
            max_hint
        )

        # Missing Policy

        if policy_level is None:

            return {

                "measurable":
                    True,

                "compliant":
                    hint_compliant,

                "hint_level":
                    hint_level,

                "teacher_max_hint_level":
                    max_hint,

                "answer_policy":
                    answer_policy,

                "reason":
                    (
                        "Hint constraint checked; answer policy "
                        "was unavailable."
                    )
            }

        # Answer Policy Constraint

        policy_compliant = (
            policy_level
            <=
            max_hint
        )

        compliant = (
            hint_compliant
            and
            policy_compliant
        )

        reasons = []

        if not hint_compliant:

            reasons.append(
                (
                    f"Hint level {hint_level} exceeded "
                    f"teacher maximum {max_hint}."
                )
            )

        if not policy_compliant:

            reasons.append(
                (
                    f"Answer policy {answer_policy} exceeded "
                    f"teacher maximum hint level {max_hint}."
                )
            )

        if compliant:

            reasons.append(
                (
                    "Structured teaching decision respected "
                    "teacher policy."
                )
            )

        return {

            "measurable":
                True,

            "compliant":
                compliant,

            "hint_level":
                hint_level,

            "teacher_max_hint_level":
                max_hint,

            "answer_policy":
                answer_policy,

            "answer_policy_level":
                policy_level,

            "reason":
                " ".join(
                    reasons
                )
        }

    # Heuristic Answer Exposure

    def _detect_answer_exposure(
        self,
        response: str,
        student_code: str,
        teacher_max_hint_level: int,
        misconception: str | None
    ) -> dict:

        # Normalize Response
        #
        # Example:
        #
        # replacing `print` with `return`
        #
        # becomes:
        #
        # replacing print with return

        response_normalized = (
            self._normalize_text(
                response
            )
        )

        reasons = []

        exact_solution_detected = False

        replacement_instruction_detected = False

        # 1. Exact Corrected Expression
        #
        # Student:
        #
        # print(distance / time)
        #
        # Tutor:
        #
        # return distance / time
        #

        expressions = re.findall(
            r"print\s*\(\s*(.*?)\s*\)",
            student_code,
            flags=re.IGNORECASE
        )

        for expression in expressions:

            cleaned_expression = (
                expression.strip()
            )

            if not cleaned_expression:
                continue

            normalized_expression = (
                self._normalize_text(
                    cleaned_expression
                )
            )

            pattern = (
                r"\breturn\s+"
                +
                re.escape(
                    normalized_expression
                )
            )

            if re.search(
                pattern,
                response_normalized,
                flags=re.IGNORECASE
            ):

                exact_solution_detected = True

                reasons.append(
                    (
                        "Tutor response appears to contain the "
                        "exact corrected return expression from "
                        "the student's code."
                    )
                )

                break

        # 2. Explicit Replacement Instruction
        #
        # Recognizes:
        #
        # replace print with return
        # replacing print with return
        # replaced print with return
        # change print to return
        # switch print to return
        # use return instead of print
        # return instead of print

        replacement_patterns = [

            (
                r"\breplac(?:e|ing|ed)\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\bwith\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\bchange\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\b(?:to|into)\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\bswitch\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\bto\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\buse\b"
                r".{0,40}"
                r"\breturn\b"
                r".{0,50}"
                r"\binstead\s+of\b"
                r".{0,30}"
                r"\bprint\b"
            ),

            (
                r"\breturn\b"
                r".{0,50}"
                r"\binstead\s+of\b"
                r".{0,30}"
                r"\bprint\b"
            )
        ]

        for pattern in replacement_patterns:

            if re.search(
                pattern,
                response_normalized,
                flags=re.IGNORECASE
            ):

                replacement_instruction_detected = True

                reasons.append(
                    (
                        "Tutor gave an explicit replacement "
                        "instruction rather than only conceptual "
                        "guidance."
                    )
                )

                break

        # 3. Code Block Detection
        #
        # A code block alone is NOT answer leakage.

        code_block_detected = bool(
            re.search(
                r"```(?:python)?[\s\S]*?```",
                response,
                flags=re.IGNORECASE
            )
        )

        if code_block_detected:

            reasons.append(
                (
                    "Tutor response contains a code block; "
                    "this alone is not classified as leakage."
                )
            )

        # 4. Teacher Maximum

        max_hint = (
            self._safe_int(
                teacher_max_hint_level,
                default=0
            )
        )

        max_hint = max(
            0,
            min(
                5,
                max_hint
            )
        )

        # 5. Determine Possible Exposure

        exposed = False

        # Exact completed assignment expression.

        if (
            exact_solution_detected
            and
            max_hint < 5
        ):

            exposed = True

        # Explicit replacement instruction.
        #
        # Limited teacher support (0–2) means this may be
        # too direct.

        elif (
            replacement_instruction_detected
            and
            max_hint <= 2
        ):

            exposed = True

        # Return-statement misconception.

        if (
            misconception
            ==
            "return_statement"
            and
            replacement_instruction_detected
            and
            max_hint <= 2
        ):

            exposed = True

        # No Evidence

        if not reasons:

            reasons.append(
                (
                    "No obvious direct-answer pattern was "
                    "detected by the heuristic evaluator."
                )
            )

        return {

            "exposed":
                exposed,

            "exact_solution_detected":
                exact_solution_detected,

            "replacement_instruction_detected":
                replacement_instruction_detected,

            "code_block_detected":
                code_block_detected,

            "reasons":
                reasons
        }

    # Aggregate Experiment Logs

    def summarize_logs(
        self,
        logs: list
    ) -> dict:

        grouped = defaultdict(
            list
        )

        for log in logs:

            variant = (
                log.model_variant
                or
                "unknown"
            )

            grouped[
                variant
            ].append(
                log
            )

        summary = {}

        for (
            variant,
            variant_logs
        ) in grouped.items():

            # Performance Metrics

            latencies = [

                log.total_latency_ms

                for log
                in variant_logs

                if log.total_latency_ms
                is not None
            ]

            llm_latencies = [

                log.llm_latency_ms

                for log
                in variant_logs

                if log.llm_latency_ms
                is not None
            ]

            tokens = [

                log.total_tokens

                for log
                in variant_logs

                if log.total_tokens
                is not None
            ]

            response_lengths = [

                log.response_length_chars

                for log
                in variant_logs

                if log.response_length_chars
                is not None
            ]

            # Structured Policy Compliance

            measurable_policy = 0

            compliant_policy = 0

            # Heuristic Answer Exposure

            exposure_count = 0

            exposure_measurable = 0

            for log in variant_logs:

                # Structured Policy

                if (
                    log.hint_level
                    is not None
                    and
                    log.teacher_max_hint_level
                    is not None
                ):

                    measurable_policy += 1

                    hint_ok = (
                        log.hint_level
                        <=
                        log.teacher_max_hint_level
                    )

                    policy_level = (
                        self.POLICY_LEVEL.get(
                            log.answer_policy
                        )
                    )

                    if policy_level is None:

                        policy_ok = True

                    else:

                        policy_ok = (
                            policy_level
                            <=
                            log.teacher_max_hint_level
                        )

                    if (
                        hint_ok
                        and
                        policy_ok
                    ):

                        compliant_policy += 1

                # Heuristic Content Exposure

                if log.tutor_response:

                    exposure_measurable += 1

                    if self._generic_exposure_from_log(

                        response=
                            log.tutor_response,

                        teacher_max_hint_level=
                            log.teacher_max_hint_level
                    ):

                        exposure_count += 1

            # Rates

            policy_rate = None

            if measurable_policy > 0:

                policy_rate = round(
                    compliant_policy
                    /
                    measurable_policy,
                    4
                )

            exposure_rate = None

            if exposure_measurable > 0:

                exposure_rate = round(
                    exposure_count
                    /
                    exposure_measurable,
                    4
                )

            # Variant Summary

            summary[
                variant
            ] = {

                "n":
                    len(
                        variant_logs
                    ),

                "average_total_latency_ms":
                    self._average(
                        latencies
                    ),

                "average_llm_latency_ms":
                    self._average(
                        llm_latencies
                    ),

                "average_total_tokens":
                    self._average(
                        tokens
                    ),

                "average_response_length_chars":
                    self._average(
                        response_lengths
                    ),

                "structured_policy_compliance_rate":
                    policy_rate,

                "policy_compliance_samples":
                    measurable_policy,

                "heuristic_answer_exposure_rate":
                    exposure_rate,

                "answer_exposure_samples":
                    exposure_measurable
            }

        return {

            "evaluation_type":
                "aggregate_experiment_summary",

            "important_note":
                (
                    "Answer exposure is a heuristic metric. "
                    "Do not treat it as validated pedagogical "
                    "quality without expert or human evaluation."
                ),

            "methods":
                summary
        }

    # Generic Exposure Detection For Stored Logs

    def _generic_exposure_from_log(
        self,
        response: str,
        teacher_max_hint_level
    ) -> bool:

        text = (
            self._normalize_text(
                response
            )
        )

        max_hint = (
            self._safe_int(
                teacher_max_hint_level,
                default=5
            )
        )

        replacement_patterns = [

            (
                r"\breplac(?:e|ing|ed)\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\bwith\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\bchange\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\b(?:to|into)\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\bswitch\b"
                r".{0,40}"
                r"\bprint\b"
                r".{0,40}"
                r"\bto\b"
                r".{0,20}"
                r"\breturn\b"
            ),

            (
                r"\buse\b"
                r".{0,40}"
                r"\breturn\b"
                r".{0,50}"
                r"\binstead\s+of\b"
                r".{0,30}"
                r"\bprint\b"
            ),

            (
                r"\breturn\b"
                r".{0,50}"
                r"\binstead\s+of\b"
                r".{0,30}"
                r"\bprint\b"
            )
        ]

        for pattern in replacement_patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                if max_hint <= 2:

                    return True

        return False

    # Normalize Text

    def _normalize_text(
        self,
        text: str
    ) -> str:

        if not isinstance(
            text,
            str
        ):

            return ""

        normalized = (
            text
            .lower()
            .replace("`", "")
            .replace("*", "")
            .replace("_", " ")
            .replace("(", " ")
            .replace(")", " ")
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        )

        return (
            normalized.strip()
        )

    # Safe Integer

    def _safe_int(
        self,
        value,
        default=0
    ) -> int:

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return default

    # Average

    def _average(
        self,
        values: list
    ):

        if not values:

            return None

        return round(
            sum(
                values
            )
            /
            len(
                values
            ),
            2
        )


# Shared Evaluation Service

evaluation_service = (
    EvaluationService()
)