import json
import os
import re
import time

from pathlib import Path

import requests

from dotenv import load_dotenv


# Environment

ENV_PATH = (
    Path(__file__).resolve().parents[1]
    /
    ".env"
)

load_dotenv(
    ENV_PATH
)


LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "qwen"
)


DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY"
)


QWEN_MODEL = os.getenv(
    "QWEN_MODEL",
    "qwen-plus"
)


DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# LLM Service

class LLMService:

    def __init__(self):

        self.provider = (
            LLM_PROVIDER
        )

        self.api_key = (
            DASHSCOPE_API_KEY
        )

        self.model = (
            QWEN_MODEL
        )

        self.base_url = (
            DASHSCOPE_BASE_URL
        )

        self.available = bool(
            self.api_key
        )

        if not self.available:

            print(
                "[LLMService] "
                "DASHSCOPE_API_KEY is missing."
            )

    # Standard Text API
    #
    # Backward-compatible method.

    def generate(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 700
    ) -> str | None:

        result = (
            self.generate_with_metadata(

                prompt=
                    prompt,

                temperature=
                    temperature,

                max_tokens=
                    max_tokens
            )
        )

        if not result:
            return None

        return result.get(
            "text"
        )

    # Text + Research Metadata

    def generate_with_metadata(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 700
    ) -> dict | None:

        if not self.available:

            print(
                "[LLMService] "
                "LLM unavailable because API key is missing."
            )

            return None

        if self.provider == "qwen":

            return (
                self._generate_qwen_with_metadata(

                    prompt=
                        prompt,

                    temperature=
                        temperature,

                    max_tokens=
                        max_tokens
                )
            )

        print(
            "[LLMService] "
            f"Unsupported provider: {self.provider}"
        )

        return None

    # Structured JSON

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 600
    ) -> dict | None:

        text = self.generate(

            prompt=
                prompt,

            temperature=
                temperature,

            max_tokens=
                max_tokens
        )

        if not text:
            return None

        # Attempt 1: direct JSON

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        # Attempt 2: remove Markdown fences

        cleaned = (
            text
            .replace(
                "```json",
                ""
            )
            .replace(
                "```JSON",
                ""
            )
            .replace(
                "```",
                ""
            )
            .strip()
        )

        try:

            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        # Attempt 3: extract first JSON object

        match = re.search(
            r"\{.*\}",
            cleaned,
            re.DOTALL
        )

        if match:

            try:

                parsed = json.loads(
                    match.group(0)
                )

                if isinstance(
                    parsed,
                    dict
                ):
                    return parsed

            except json.JSONDecodeError:
                pass

        print(
            "[LLMService] "
            "Could not parse JSON response."
        )

        return None

    # Semantic Code Diagnosis

    def diagnose_code(
        self,
        *,
        assignment: dict,
        student_code: str,
        execution_result: dict
    ) -> dict | None:
        """
        LLM fallback for semantic diagnosis.

        This method is intentionally used only when deterministic
        execution/AST rules cannot identify a specific misconception.
        It returns structured JSON or None.
        """

        if not self.available:
            return None

        failed_tests = [
            result
            for result in (
                execution_result.get(
                    "test_results",
                    []
                )
                or
                []
            )
            if not result.get(
                "passed",
                False
            )
        ]

        # Keep the prompt compact: at most 3 failed cases.
        compact_failures = []

        for result in failed_tests[:3]:
            compact_failures.append({
                "input":
                    result.get(
                        "input"
                    ),

                "expected":
                    result.get(
                        "expected"
                    ),

                "actual":
                    result.get(
                        "actual"
                    ),

                "runtime_error":
                    result.get(
                        "runtime_error"
                    )
            })

        allowed_concepts = (
            assignment.get(
                "concepts",
                []
            )
            or
            []
        )

        prompt = f"""
You are a diagnostic component in a programming tutoring system.

Your task is NOT to teach the learner and NOT to write a tutoring response.
Your task is only to identify the most likely programming misconception
supported by the execution evidence.

Assignment:
{json.dumps(assignment, ensure_ascii=False)}

Student code:
```python
{student_code}
```

Failed execution evidence:
{json.dumps(compact_failures, ensure_ascii=False)}

Allowed concepts:
{json.dumps(allowed_concepts, ensure_ascii=False)}

Return ONLY one JSON object with this exact structure:

{{
  "concept": "one concept",
  "misconception": "short_snake_case_label",
  "severity": "low|medium|high",
  "evidence": "brief evidence-based explanation",
  "confidence": 0.0
}}

Rules:
1. Base the diagnosis on the code and failed tests.
2. Prefer one of the allowed concepts when possible.
3. Do not invent learner traits, emotions, or history.
4. Do not give the learner the solution.
5. Use confidence between 0 and 1.
6. If evidence is ambiguous, use a broader misconception label and
   lower confidence.
"""

        result = self.generate_json(
            prompt=prompt,
            temperature=0.1,
            max_tokens=350
        )

        if not isinstance(
            result,
            dict
        ):
            return None

        concept = result.get(
            "concept"
        )

        misconception = result.get(
            "misconception"
        )

        severity = result.get(
            "severity"
        )

        evidence = result.get(
            "evidence"
        )

        confidence = result.get(
            "confidence"
        )

        if not concept:
            return None

        if not misconception:
            return None

        if severity not in {
            "low",
            "medium",
            "high"
        }:
            severity = "medium"

        try:
            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):
            confidence = 0.6

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        # Avoid arbitrary concept drift when the assignment already
        # defines a concept vocabulary.
        if (
            allowed_concepts
            and
            concept not in allowed_concepts
        ):
            concept = (
                allowed_concepts[0]
            )

        return {
            "concept":
                str(
                    concept
                ),

            "misconception":
                str(
                    misconception
                ),

            "severity":
                severity,

            "evidence":
                (
                    str(
                        evidence
                    )
                    if evidence
                    else
                    "The diagnosis was inferred from failed execution evidence."
                ),

            "confidence":
                round(
                    confidence,
                    2
                )
        }


    # Qwen

    def _generate_qwen_with_metadata(
    self,
    prompt: str,
    temperature: float,
    max_tokens: int
) -> dict | None:

        url = (
            f"{self.base_url.rstrip('/')}"
            "/chat/completions"
        )

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json"
        }

        payload = {
            "model":
                self.model,

            "messages": [
                {
                    "role":
                        "system",

                    "content":
                        (
                            "You are part of an adaptive "
                            "educational tutoring system. "
                            "Follow the requested instructional "
                            "constraints and output format."
                        )
                },
                {
                    "role":
                        "user",

                    "content":
                        prompt
                }
            ],

            "temperature":
                temperature,

            "max_tokens":
                max_tokens,

            "enable_thinking":
                False
        }

        start_time = (
            time.perf_counter()
        )

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(10, 90)
            )

            latency_ms = (
                (
                    time.perf_counter()
                    -
                    start_time
                )
                *
                1000
            )

            if not response.ok:

                print(
                    "[LLMService] "
                    "Qwen API error"
                )

                print(
                    "Status:",
                    response.status_code
                )

                print(
                    "Response:",
                    response.text
                )

                return None

            data = (
                response.json()
            )

            choices = data.get(
                "choices",
                []
            )

            if not choices:

                print(
                    "[LLMService] "
                    "No choices returned."
                )

                return None

            message = (
                choices[0]
                .get(
                    "message",
                    {}
                )
            )

            content = (
                message.get(
                    "content"
                )
            )

            if not content:

                print(
                    "[LLMService] "
                    "Empty LLM response."
                )

                return None

            usage = (
                data.get(
                    "usage",
                    {}
                )
                or
                {}
            )

            prompt_tokens = (
                usage.get(
                    "prompt_tokens"
                )
            )

            completion_tokens = (
                usage.get(
                    "completion_tokens"
                )
            )

            total_tokens = (
                usage.get(
                    "total_tokens"
                )
            )

            return {
                "text":
                    content.strip(),

                "provider":
                    self.provider,

                "model":
                    data.get(
                        "model",
                        self.model
                    ),

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    total_tokens,

                "latency_ms":
                    round(
                        latency_ms,
                        2
                    )
            }

        except requests.Timeout as error:

            elapsed_ms = (
                (
                    time.perf_counter()
                    -
                    start_time
                )
                *
                1000
            )

            print(
                "[LLMService] "
                "Qwen request timed out."
            )

            print(
                "Elapsed:",
                round(
                    elapsed_ms,
                    2
                ),
                "ms"
            )

            print(
                "Error:",
                error
            )

            return None

        except requests.RequestException as error:

            print(
                "[LLMService] "
                f"HTTP request failed: {error}"
            )

            return None

        except Exception as error:

            print(
                "[LLMService] "
                f"Unexpected error: {error}"
            )

            return None


# Shared Instance

llm_service = (
    LLMService()
)