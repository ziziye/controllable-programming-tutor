from __future__ import annotations

import ast
import json
import subprocess
import sys
from typing import Any


RUNNER_CODE = """
import contextlib
import io
import json
import sys


def safe_value(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def main():
    payload = json.loads(sys.stdin.read())

    code = payload["code"]
    function_name = payload["function_name"]
    hidden_tests = payload["hidden_tests"]

    namespace = {
        "__name__": "__student_code__"
    }

    setup_output = io.StringIO()

    try:
        with contextlib.redirect_stdout(setup_output):
            exec(code, namespace, namespace)

    except BaseException as error:
        print(json.dumps({
            "execution_success": False,
            "syntax_valid": True,
            "function_found": False,
            "all_tests_passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": "runtime_error",
            "error_message": f"{type(error).__name__}: {error}",
            "captured_output": setup_output.getvalue(),
            "test_results": []
        }))
        return

    target = namespace.get(function_name)

    if not callable(target):
        print(json.dumps({
            "execution_success": False,
            "syntax_valid": True,
            "function_found": False,
            "all_tests_passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": "function_not_found",
            "error_message": (
                f"Required function '{function_name}' was not found."
            ),
            "captured_output": setup_output.getvalue(),
            "test_results": []
        }))
        return

    test_results = []
    passed_count = 0

    for index, test_case in enumerate(hidden_tests, start=1):
        arguments = test_case["input"]
        expected = test_case["expected"]

        call_output = io.StringIO()

        try:
            with contextlib.redirect_stdout(call_output):
                actual = target(*arguments)

            passed = actual == expected

            if passed:
                passed_count += 1

            test_results.append({
                "test_index": index,
                "input": safe_value(arguments),
                "expected": safe_value(expected),
                "actual": safe_value(actual),
                "passed": passed,
                "runtime_error": None,
                "printed_output": call_output.getvalue()
            })

        except BaseException as error:
            test_results.append({
                "test_index": index,
                "input": safe_value(arguments),
                "expected": safe_value(expected),
                "actual": None,
                "passed": False,
                "runtime_error": f"{type(error).__name__}: {error}",
                "printed_output": call_output.getvalue()
            })

    tests_run = len(hidden_tests)
    tests_failed = tests_run - passed_count

    print(json.dumps({
        "execution_success": True,
        "syntax_valid": True,
        "function_found": True,
        "all_tests_passed": tests_failed == 0,
        "tests_run": tests_run,
        "tests_passed": passed_count,
        "tests_failed": tests_failed,
        "error_type": None if tests_failed == 0 else "test_failure",
        "error_message": None,
        "captured_output": setup_output.getvalue(),
        "test_results": test_results
    }))


if __name__ == "__main__":
    main()
"""


class CodeExecutionService:
    """
    Simple local execution service for the research prototype.

    It checks Python syntax and then executes the student's code in a
    separate Python subprocess with a timeout.

    This is useful for local research/testing, but it is not a
    production-grade security sandbox.
    """

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        code: str,
        function_name: str | None,
        hidden_tests: list[dict[str, Any]] | None
    ) -> dict[str, Any]:

        # 1. Syntax check
        try:
            ast.parse(code)

        except SyntaxError as error:
            return {
                "execution_success": False,
                "syntax_valid": False,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "syntax_error",
                "error_message": (
                    f"{error.msg} (line {error.lineno})"
                ),
                "test_results": []
            }

        # 2. Assignment configuration checks
        if not function_name:
            return {
                "execution_success": False,
                "syntax_valid": True,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "missing_function_name",
                "error_message": (
                    "The assignment does not define function_name."
                ),
                "test_results": []
            }

        tests = hidden_tests or []

        if not tests:
            return {
                "execution_success": False,
                "syntax_valid": True,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "missing_tests",
                "error_message": (
                    "The assignment has no hidden tests."
                ),
                "test_results": []
            }

        # 3. Validate hidden-test format
        for index, test_case in enumerate(tests, start=1):
            if not isinstance(test_case, dict):
                return self._invalid_test_result(
                    f"Test case {index} must be a JSON object."
                )

            if "input" not in test_case:
                return self._invalid_test_result(
                    f"Test case {index} is missing 'input'."
                )

            if "expected" not in test_case:
                return self._invalid_test_result(
                    f"Test case {index} is missing 'expected'."
                )

            if not isinstance(test_case["input"], list):
                return self._invalid_test_result(
                    f"Test case {index} 'input' must be a list."
                )

        # 4. Run in a separate Python process
        payload = {
            "code": code,
            "function_name": function_name,
            "hidden_tests": tests
        }

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    RUNNER_CODE
                ],
                input=json.dumps(
                    payload,
                    ensure_ascii=False
                ),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )

        except subprocess.TimeoutExpired:
            return {
                "execution_success": False,
                "syntax_valid": True,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "timeout",
                "error_message": (
                    "Student code exceeded the "
                    f"{self.timeout_seconds}-second limit."
                ),
                "test_results": []
            }

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if not stdout:
            return {
                "execution_success": False,
                "syntax_valid": True,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "runner_error",
                "error_message": (
                    stderr
                    or
                    "Execution runner returned no output."
                ),
                "test_results": []
            }

        try:
            return json.loads(stdout)

        except json.JSONDecodeError:
            return {
                "execution_success": False,
                "syntax_valid": True,
                "function_found": False,
                "all_tests_passed": False,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "error_type": "runner_error",
                "error_message": (
                    "Execution runner returned invalid JSON."
                ),
                "runner_stdout": stdout,
                "runner_stderr": stderr,
                "test_results": []
            }

    @staticmethod
    def _invalid_test_result(
        message: str
    ) -> dict[str, Any]:
        return {
            "execution_success": False,
            "syntax_valid": True,
            "function_found": False,
            "all_tests_passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "error_type": "invalid_test_format",
            "error_message": message,
            "test_results": []
        }


code_execution_service = CodeExecutionService()