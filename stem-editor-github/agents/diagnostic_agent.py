from __future__ import annotations

import ast
from typing import Any

from services.llm_service import llm_service


class DiagnosticAgent:
    """
    Phase 1 structured diagnostic agent.

    This version is intentionally simple and deterministic.

    It can work in two modes:

    1. Backward-compatible mode:
       analyze_code(code)

       Used by the existing tutor pipeline before code execution
       is connected to tutor.py.

    2. Execution-aware mode:
       analyze_code(
           code=student_code,
           assignment=assignment,
           execution_result=execution_result
       )

       Uses hidden-test evidence from CodeExecutionService.

    Later, an LLM semantic diagnosis can be added on top of this
    evidence without changing the basic output structure.
    """

    # Public API

    def analyze_code(
        self,
        code: str,
        assignment: Any | None = None,
        execution_result: dict[str, Any] | None = None
    ) -> dict[str, Any]:

        # 1. Syntax check

        try:
            tree = ast.parse(code)

        except SyntaxError as error:
            return self._result(
                syntax_valid=False,
                is_correct=False,
                error_type="syntax_error",
                message=(
                    f"{error.msg} "
                    f"(line {error.lineno})"
                ),
                concept="python_syntax",
                misconception="python_syntax_error",
                severity="high",
                evidence=(
                    "The submitted code could not be parsed "
                    "as valid Python."
                ),
                confidence=1.0
            )

        # 2. If execution evidence is available, use it.

        if execution_result is not None:
            return self._diagnose_from_execution(
                code=code,
                tree=tree,
                assignment=assignment,
                execution_result=execution_result
            )

        # 3. Structural diagnosis first

        structural_diagnosis = (
            self._diagnose_structure(
                tree=tree,
                assignment=assignment
            )
        )

        # Keep deterministic findings when a concrete structural
        # problem is already known.
        if (
            structural_diagnosis.get("misconception")
            is not None
            or
            structural_diagnosis.get("is_correct")
            is False
        ):
            return structural_diagnosis

        # 4. Assignment-aware semantic fallback
        #
        # For code-oriented tasks without executable hidden
        # tests, "no structural error" is NOT success evidence.
        # We may still identify a likely task-specific issue
        # semantically, while keeping is_correct=None.

        if assignment is not None:

            semantic_diagnosis = (
                self._semantic_fallback_without_execution(
                    assignment=assignment,
                    code=code
                )
            )

            if semantic_diagnosis:

                return self._result(
                    syntax_valid=True,
                    is_correct=None,
                    error_type="semantic_issue",
                    message=(
                        "The code is structurally valid, but an "
                        "assignment-specific issue was identified "
                        "from semantic analysis."
                    ),
                    concept=semantic_diagnosis["concept"],
                    misconception=semantic_diagnosis["misconception"],
                    severity=semantic_diagnosis["severity"],
                    evidence=semantic_diagnosis["evidence"],
                    confidence=semantic_diagnosis["confidence"],
                    diagnosis_source="llm_semantic_fallback"
                )

        # No objective execution evidence and no sufficiently
        # supported semantic issue: remain neutral.
        return structural_diagnosis

    # Execution-aware diagnosis

    def _diagnose_from_execution(
        self,
        code: str,
        tree: ast.AST,
        assignment: Any | None,
        execution_result: dict[str, Any]
    ) -> dict[str, Any]:

        error_type = execution_result.get(
            "error_type"
        )

        # A. Execution service reported syntax error

        if (
            execution_result.get("syntax_valid")
            is False
        ):
            return self._result(
                syntax_valid=False,
                is_correct=False,
                error_type="syntax_error",
                message=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "The code contains a syntax error."
                ),
                concept="python_syntax",
                misconception="python_syntax_error",
                severity="high",
                evidence=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "Python could not parse the submitted code."
                ),
                confidence=1.0
            )

        # B. Assignment configuration problem

        if error_type in {
            "missing_function_name",
            "missing_tests",
            "invalid_test_format"
        }:
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="assignment_configuration_error",
                message=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "The assignment configuration is incomplete."
                ),
                concept=None,
                misconception=None,
                severity="high",
                evidence=(
                    execution_result.get(
                        "error_message"
                    )
                ),
                confidence=1.0
            )

        # C. Required function was not found

        if error_type == "function_not_found":
            function_name = self._get_assignment_value(
                assignment,
                "function_name"
            )

            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="structural_error",
                message=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "The required function was not found."
                ),
                concept="function_definition",
                misconception="missing_required_function",
                severity="high",
                evidence=(
                    f"The assignment expects the function "
                    f"'{function_name}', but it was not found "
                    f"in the submission."
                    if function_name
                    else
                    "The required assignment function was not found."
                ),
                confidence=1.0
            )

        # D. Timeout

        if error_type == "timeout":
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="runtime_error",
                message=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "The code exceeded the execution time limit."
                ),
                concept=self._choose_concept(
                    assignment
                ),
                misconception="non_terminating_or_inefficient_logic",
                severity="high",
                evidence=(
                    "The code did not finish within the "
                    "allowed execution time."
                ),
                confidence=0.95
            )

        # E. Runtime error

        if error_type in {
            "runtime_error",
            "runner_error"
        }:
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="runtime_error",
                message=(
                    execution_result.get(
                        "error_message"
                    )
                    or
                    "The code raised an error during execution."
                ),
                concept=self._choose_concept(
                    assignment
                ),
                misconception="runtime_failure",
                severity="high",
                evidence=(
                    execution_result.get(
                        "error_message"
                    )
                ),
                confidence=0.95
            )

        # F. All hidden tests passed

        if execution_result.get(
            "all_tests_passed"
        ):
            tests_run = execution_result.get(
                "tests_run",
                0
            )

            return self._result(
                syntax_valid=True,
                is_correct=True,
                error_type=None,
                message=(
                    "The submission passed all hidden tests."
                ),
                concept=self._choose_concept(
                    assignment
                ),
                misconception=None,
                severity="low",
                evidence=(
                    f"Passed {tests_run} of "
                    f"{tests_run} hidden tests."
                ),
                confidence=0.99
            )

        # G. Hidden-test failures
        #
        # Try a few transparent AST rules first.
        # Otherwise return a generic logical-error diagnosis.

        test_results = (
            execution_result.get(
                "test_results"
            )
            or
            []
        )

        # Rule 1:
        # Student prints a result but does not return it.
        if self._prints_without_return(
            tree
        ):
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="logical_error",
                message=(
                    "The function prints a result but does not "
                    "return the required value."
                ),
                concept="return_statement",
                misconception="print_instead_of_return",
                severity="medium",
                evidence=self._build_test_evidence(
                    test_results
                ),
                confidence=0.95
            )

        # Rule 2:
        # Common parity error:
        #
        #     number % 2 != 0
        #
        # when the assignment is about even numbers.
        if (
            self._assignment_mentions_even(
                assignment
            )
            and
            self._uses_odd_parity_condition(
                tree
            )
        ):
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="logical_error",
                message=(
                    "The parity condition selects odd numbers "
                    "instead of even numbers."
                ),
                concept="modulo_operator",
                misconception="incorrect_even_number_condition",
                severity="medium",
                evidence=self._build_test_evidence(
                    test_results
                ),
                confidence=0.97
            )

        # Hybrid semantic fallback
        #
        # Only use the LLM when deterministic rules cannot identify
        # a specific misconception.

        semantic_diagnosis = (
            self._semantic_fallback(
                assignment=assignment,
                code=code,
                execution_result=execution_result
            )
        )

        if semantic_diagnosis:

            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="logical_error",
                message=(
                    "The submission contains a logical error "
                    "identified from execution evidence."
                ),
                concept=semantic_diagnosis[
                    "concept"
                ],
                misconception=semantic_diagnosis[
                    "misconception"
                ],
                severity=semantic_diagnosis[
                    "severity"
                ],
                evidence=semantic_diagnosis[
                    "evidence"
                ],
                confidence=semantic_diagnosis[
                    "confidence"
                ],
                diagnosis_source="llm_semantic_fallback"
            )

        # Generic deterministic fallback if the LLM is unavailable
        # or its structured response cannot be validated.
        failed_count = execution_result.get(
            "tests_failed",
            0
        )

        tests_run = execution_result.get(
            "tests_run",
            0
        )

        return self._result(
            syntax_valid=True,
            is_correct=False,
            error_type="logical_error",
            message=(
                f"The submission failed "
                f"{failed_count} of {tests_run} hidden tests."
            ),
            concept=self._choose_concept(
                assignment
            ),
            misconception="incorrect_program_logic",
            severity=self._severity_from_test_failures(
                failed_count=failed_count,
                tests_run=tests_run
            ),
            evidence=self._build_test_evidence(
                test_results
            ),
            confidence=0.65,
            diagnosis_source="deterministic_fallback"
        )

    # Backward-compatible structural diagnosis

    def _diagnose_structure(
        self,
        tree: ast.AST,
        assignment: Any | None
    ) -> dict[str, Any]:

        function_name = self._get_assignment_value(
            assignment,
            "function_name"
        )

        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            )
        ]

        if not functions:
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="structural_error",
                message=(
                    "No function definition was detected."
                ),
                concept="function_definition",
                misconception="missing_function_definition",
                severity="medium",
                evidence=(
                    "The code contains no Python function definition."
                ),
                confidence=1.0
            )

        if function_name:
            required_function_found = any(
                function.name == function_name
                for function in functions
            )

            if not required_function_found:
                return self._result(
                    syntax_valid=True,
                    is_correct=False,
                    error_type="structural_error",
                    message=(
                        f"The required function "
                        f"'{function_name}' was not detected."
                    ),
                    concept="function_definition",
                    misconception="incorrect_function_name",
                    severity="medium",
                    evidence=(
                        f"The assignment expects "
                        f"'{function_name}'."
                    ),
                    confidence=1.0
                )

        has_return = any(
            isinstance(
                node,
                ast.Return
            )
            for node in ast.walk(tree)
        )

        if not has_return:
            return self._result(
                syntax_valid=True,
                is_correct=False,
                error_type="structural_error",
                message=(
                    "The function does not contain "
                    "a return statement."
                ),
                concept="return_statement",
                misconception="missing_return_statement",
                severity="medium",
                evidence=(
                    "No return statement was detected "
                    "in the submitted code."
                ),
                confidence=1.0
            )

        return self._result(
            syntax_valid=True,
            is_correct=None,
            error_type=None,
            message=(
                "No obvious structural errors detected. "
                "Execution evidence is required to verify correctness."
            ),
            concept=self._choose_concept(
                assignment
            ),
            misconception=None,
            severity="low",
            evidence=(
                "The code is syntactically valid and "
                "contains a function and return statement."
            ),
            confidence=0.60
        )

    # Small transparent rules

    @staticmethod
    def _prints_without_return(
        tree: ast.AST
    ) -> bool:

        has_print = False
        has_return = False

        for node in ast.walk(
            tree
        ):
            if isinstance(
                node,
                ast.Return
            ):
                has_return = True

            if (
                isinstance(
                    node,
                    ast.Call
                )
                and
                isinstance(
                    node.func,
                    ast.Name
                )
                and
                node.func.id == "print"
            ):
                has_print = True

        return (
            has_print
            and
            not has_return
        )

    @staticmethod
    def _uses_odd_parity_condition(
        tree: ast.AST
    ) -> bool:
        """
        Detect patterns such as:

            x % 2 != 0

        This is deliberately narrow and transparent.
        """

        for node in ast.walk(
            tree
        ):
            if not isinstance(
                node,
                ast.Compare
            ):
                continue

            if len(node.ops) != 1:
                continue

            if not isinstance(
                node.ops[0],
                ast.NotEq
            ):
                continue

            if len(node.comparators) != 1:
                continue

            left = node.left
            right = node.comparators[0]

            if not isinstance(
                left,
                ast.BinOp
            ):
                continue

            if not isinstance(
                left.op,
                ast.Mod
            ):
                continue

            if not (
                isinstance(
                    left.right,
                    ast.Constant
                )
                and
                left.right.value == 2
            ):
                continue

            if not (
                isinstance(
                    right,
                    ast.Constant
                )
                and
                right.value == 0
            ):
                continue

            return True

        return False

    def _assignment_mentions_even(
        self,
        assignment: Any | None
    ) -> bool:

        pieces: list[str] = []

        for field_name in (
            "title",
            "description",
            "learning_objective"
        ):
            value = self._get_assignment_value(
                assignment,
                field_name
            )

            if value:
                pieces.append(
                    str(value)
                )

        for field_name in (
            "learning_objectives",
            "concepts",
            "constraints"
        ):
            value = self._get_assignment_value(
                assignment,
                field_name
            )

            if isinstance(
                value,
                list
            ):
                pieces.extend(
                    str(item)
                    for item in value
                )

        text = " ".join(
            pieces
        ).lower()

        return (
            "even"
            in text
        )

    # Semantic LLM Fallback

    def _semantic_fallback_without_execution(
        self,
        *,
        assignment: Any | None,
        code: str
    ) -> dict[str, Any] | None:
        """
        Assignment-aware semantic diagnosis for code-oriented
        tasks that do not have executable hidden-test evidence.

        A likely misconception may be identified, but objective
        correctness is deliberately left unknown by the caller.
        """

        assignment_context = {
            "title": self._get_assignment_value(assignment, "title"),
            "description": self._get_assignment_value(
                assignment,
                "description"
            ),
            "question_type": self._get_assignment_value(
                assignment,
                "question_type"
            ),
            "function_name": self._get_assignment_value(
                assignment,
                "function_name"
            ),
            "learning_objective": self._get_assignment_value(
                assignment,
                "learning_objective"
            ),
            "learning_objectives": (
                self._get_assignment_value(
                    assignment,
                    "learning_objectives"
                )
                or
                []
            ),
            "concepts": (
                self._get_assignment_value(
                    assignment,
                    "concepts"
                )
                or
                []
            ),
            "constraints": (
                self._get_assignment_value(
                    assignment,
                    "constraints"
                )
                or
                []
            )
        }

        allowed_concepts = (
            assignment_context.get("concepts")
            or
            []
        )

        prompt = f"""
You are the semantic diagnostic component of a programming tutoring system.

The learner code has already passed basic Python syntax and structural checks.
There are NO executable hidden-test results for this interaction.

Determine whether the learner code appears to miss a clear,
task-specific requirement stated in the assignment.

Assignment:
{assignment_context}

Student code:
```python
{code}
```

Allowed concepts:
{allowed_concepts}

Return ONLY one JSON object with this exact structure:

{{
  "issue_detected": true,
  "concept": "one concept",
  "misconception": "short_snake_case_label",
  "severity": "low|medium|high",
  "evidence": "brief explanation grounded in the assignment and code",
  "confidence": 0.0
}}

Rules:
1. Do not claim that hidden tests passed or failed.
2. Do not claim objective correctness.
3. Diagnose only a clear assignment-specific issue visible from the code.
4. Prefer one of the allowed concepts when possible.
5. If evidence is ambiguous, set "issue_detected" to false.
6. Do not teach the solution.
7. Confidence must be between 0 and 1.
"""

        try:
            result = llm_service.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=350
            )
        except Exception as error:
            print(
                "[DiagnosticAgent] "
                "Semantic fallback without execution failed: "
                f"{error}"
            )
            return None

        if not isinstance(result, dict):
            return None

        if not bool(result.get("issue_detected", False)):
            return None

        concept = result.get("concept")
        misconception = result.get("misconception")
        severity = result.get("severity")
        evidence = result.get("evidence")
        confidence = result.get("confidence")

        if not concept or not misconception:
            return None

        if severity not in {"low", "medium", "high"}:
            severity = "medium"

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))

        # Avoid recording weak semantic guesses as learner memory.
        if confidence < 0.60:
            return None

        if allowed_concepts:
            normalized_allowed = {
                str(item).strip().lower(): str(item).strip()
                for item in allowed_concepts
                if str(item).strip()
            }

            key = str(concept).strip().lower()

            if key in normalized_allowed:
                concept = normalized_allowed[key]
            else:
                non_generic = [
                    str(item).strip()
                    for item in allowed_concepts
                    if str(item).strip().lower()
                    not in {
                        "python",
                        "programming",
                        "general_python"
                    }
                ]
                concept = (
                    non_generic[0]
                    if non_generic
                    else str(allowed_concepts[0])
                )

        return {
            "concept": str(concept),
            "misconception": str(misconception),
            "severity": severity,
            "evidence": (
                str(evidence)
                if evidence
                else (
                    "The learner code appears to miss an "
                    "assignment-specific requirement."
                )
            ),
            "confidence": round(confidence, 2)
        }


    def _semantic_fallback(
        self,
        *,
        assignment: Any | None,
        code: str,
        execution_result: dict[str, Any]
    ) -> dict[str, Any] | None:

        assignment_context = {
            "title":
                self._get_assignment_value(
                    assignment,
                    "title"
                ),

            "description":
                self._get_assignment_value(
                    assignment,
                    "description"
                ),

            "function_name":
                self._get_assignment_value(
                    assignment,
                    "function_name"
                ),

            "learning_objective":
                self._get_assignment_value(
                    assignment,
                    "learning_objective"
                ),

            "learning_objectives":
                (
                    self._get_assignment_value(
                        assignment,
                        "learning_objectives"
                    )
                    or
                    []
                ),

            "concepts":
                (
                    self._get_assignment_value(
                        assignment,
                        "concepts"
                    )
                    or
                    []
                ),

            "constraints":
                (
                    self._get_assignment_value(
                        assignment,
                        "constraints"
                    )
                    or
                    []
                )
        }

        try:
            return llm_service.diagnose_code(
                assignment=assignment_context,
                student_code=code,
                execution_result=execution_result
            )

        except Exception as error:
            print(
                "[DiagnosticAgent] "
                f"Semantic fallback failed: {error}"
            )

            return None


    # Evidence / concept helpers

    @staticmethod
    def _build_test_evidence(
        test_results: list[dict[str, Any]]
    ) -> str:

        failed = [
            result
            for result in test_results
            if not result.get(
                "passed",
                False
            )
        ]

        if not failed:
            return (
                "No failed test evidence was available."
            )

        examples: list[str] = []

        for result in failed[:3]:
            runtime_error = result.get(
                "runtime_error"
            )

            if runtime_error:
                examples.append(
                    f"test {result.get('test_index')}: "
                    f"{runtime_error}"
                )

            else:
                examples.append(
                    f"test {result.get('test_index')}: "
                    f"expected {result.get('expected')!r}, "
                    f"got {result.get('actual')!r}"
                )

        return "; ".join(
            examples
        )

    def _choose_concept(
        self,
        assignment: Any | None
    ) -> str:

        concepts = self._get_assignment_value(
            assignment,
            "concepts"
        )

        if (
            isinstance(
                concepts,
                list
            )
            and
            concepts
        ):

            normalized = [
                str(concept).strip()
                for concept in concepts
                if str(concept).strip()
            ]

            generic = {
                "python",
                "programming",
                "general_python"
            }

            for concept in normalized:
                if concept.lower() not in generic:
                    return concept

            if normalized:
                return normalized[0]

        return "general_python"

    @staticmethod
    def _severity_from_test_failures(
        failed_count: int,
        tests_run: int
    ) -> str:

        if tests_run <= 0:
            return "medium"

        failure_ratio = (
            failed_count
            /
            tests_run
        )

        if failure_ratio >= 0.75:
            return "high"

        if failure_ratio >= 0.25:
            return "medium"

        return "low"

    @staticmethod
    def _get_assignment_value(
        assignment: Any | None,
        field_name: str
    ) -> Any:

        if assignment is None:
            return None

        if isinstance(
            assignment,
            dict
        ):
            return assignment.get(
                field_name
            )

        return getattr(
            assignment,
            field_name,
            None
        )

    # Standard result format

    @staticmethod
    def _result(
        *,
        syntax_valid: bool,
        is_correct: bool | None,
        error_type: str | None,
        message: str,
        concept: str | None,
        misconception: str | None,
        severity: str,
        evidence: str | None,
        confidence: float,
        diagnosis_source: str = "deterministic"
    ) -> dict[str, Any]:

        return {
            "syntax_valid":
                syntax_valid,

            "is_correct":
                is_correct,

            "error_type":
                error_type,

            "message":
                message,

            "concept":
                concept,

            "misconception":
                misconception,

            "severity":
                severity,

            "evidence":
                evidence,

            "confidence":
                round(
                    max(
                        0.0,
                        min(
                            1.0,
                            confidence
                        )
                    ),
                    2
                ),

            "diagnosis_source":
                diagnosis_source
        }
