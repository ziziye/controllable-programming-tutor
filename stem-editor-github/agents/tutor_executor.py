import json

from services.llm_service import (
    llm_service
)


class TutorExecutor:
    """
    Natural-language execution layer.

    Pedagogical reasoning happens upstream.

    This class:
        - executes the teaching plan
        - follows answer policy
        - follows persona style
        - records LLM usage metadata for experiments
    """

    # Backward-Compatible Method

    def generate_response(
        self,
        diagnosis: dict,
        pedagogy: dict,
        persona: dict,
        student_profile: dict | None = None,
        mastery: dict | None = None,
        memory: dict | None = None,
        assignment: dict | None = None,
        code: str = "",
        preferred_language: str = "en"
    ) -> str:

        result = (
            self.generate_response_with_metadata(

                diagnosis=
                    diagnosis,

                pedagogy=
                    pedagogy,

                persona=
                    persona,

                student_profile=
                    student_profile,

                mastery=
                    mastery,

                memory=
                    memory,

                assignment=
                    assignment,

                code=
                    code,

                preferred_language=
                    preferred_language
            )
        )

        return result[
            "response"
        ]

    # Research-Aware Generation

    def generate_response_with_metadata(
        self,
        diagnosis: dict,
        pedagogy: dict,
        persona: dict,
        student_profile: dict | None = None,
        mastery: dict | None = None,
        memory: dict | None = None,
        assignment: dict | None = None,
        code: str = "",
        preferred_language: str = "en"
    ) -> dict:

        student_profile = (
            student_profile
            or {}
        )

        mastery = (
            mastery
            or {}
        )

        memory = (
            memory
            or {}
        )

        assignment = (
            assignment
            or {}
        )

        preferred_language = (
            "zh"
            if str(
                preferred_language
                or "en"
            ).strip().lower().startswith(
                "zh"
            )
            else
            "en"
        )

        prompt = self._build_prompt(

            diagnosis=
                diagnosis,

            pedagogy=
                pedagogy,

            persona=
                persona,

            student_profile=
                student_profile,

            mastery=
                mastery,

            memory=
                memory,

            assignment=
                assignment,

            code=
                code,

            preferred_language=
                preferred_language
        )
        print(
            "[DEBUG] POMDP prompt chars:",
            len(
                prompt
            )
        )

        print(
            "[DEBUG] POMDP prompt preview:"
        )

        print(
            prompt[:500]
        )

        print(
            "[DEBUG] End POMDP prompt preview"
        )
        llm_result = (
            llm_service
            .generate_with_metadata(

                prompt=
                    prompt,

                temperature=
                    0.3,

                max_tokens=
                    350
            )
        )

        if llm_result:

            return {

                "response":
                    llm_result.get(
                        "text",
                        ""
                    ),

                "llm_used":
                    True,

                "llm_metadata": {

                    "provider":
                        llm_result.get(
                            "provider"
                        ),

                    "model":
                        llm_result.get(
                            "model"
                        ),

                    "prompt_tokens":
                        llm_result.get(
                            "prompt_tokens"
                        ),

                    "completion_tokens":
                        llm_result.get(
                            "completion_tokens"
                        ),

                    "total_tokens":
                        llm_result.get(
                            "total_tokens"
                        ),

                    "latency_ms":
                        llm_result.get(
                            "latency_ms"
                        )
                }
            }

        fallback = (
            self._fallback_response(

                diagnosis=
                    diagnosis,

                pedagogy=
                    pedagogy,

                preferred_language=
                    preferred_language
            )
        )

        return {

            "response":
                fallback,

            "llm_used":
                False,

            "llm_metadata": {

                "provider":
                    None,

                "model":
                    None,

                "prompt_tokens":
                    None,

                "completion_tokens":
                    None,

                "total_tokens":
                    None,

                "latency_ms":
                    None
            }
        }

    # Prompt

    def _build_prompt(
        self,
        diagnosis: dict,
        pedagogy: dict,
        persona: dict,
        student_profile: dict,
        mastery: dict,
        memory: dict,
        assignment: dict,
        code: str,
        preferred_language: str = "en"
    ) -> str:

        context = {

            "assignment":
                assignment,

            "student_profile":
                student_profile,

            "mastery":
                mastery,

            "memory":
                memory,

            "diagnosis":
                diagnosis,

            "pedagogy":
                pedagogy,

            "persona":
                persona
        }

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            indent=2
        )

        answer_policy = (
            pedagogy.get(
                "answer_policy",
                "HINT_ONLY"
            )
        )

        prompt = f"""
You are an adaptive AI programming tutor.

The tutoring system has already analyzed the learner and
selected an instructional strategy.

You are the execution layer.

Do NOT replace the upstream pedagogical decision.


==================================================
RESPONSE LANGUAGE
==================================================

Preferred language:
{preferred_language}

If preferred_language is "zh":
- Respond in natural Simplified Chinese.
- Keep Python identifiers, API names, function names, and code in their original form.
- Do not translate code.

If preferred_language is "en":
- Respond in natural English.

Follow this language instruction for the tutoring message only.


==================================================
STRUCTURED CONTEXT
==================================================

{context_json}


==================================================
CURRENT STUDENT CODE
==================================================

{code}


==================================================
ANSWER POLICY
==================================================

Current answer policy:

{answer_policy}


NO_ANSWER:
- Do not provide the solution.
- Do not provide corrected final code.
- Encourage reasoning, prediction, or testing.

HINT_ONLY:
- Provide one or two focused hints or questions.
- Do not reveal the full correction.
- Do not provide finished solution code.

PARTIAL_SOLUTION:
- Provide structured scaffolding.
- You may show a small analogous example.
- You may explain the next step.
- Do not provide the complete finished assignment.

DIRECT_EXPLANATION:
- Clearly explain the misconception.
- Explicit corrective guidance is allowed.
- Explain why the correction works.
- Do not provide an unnecessary complete assignment solution.

FULL_SOLUTION_ALLOWED:
- A complete worked solution may be provided if necessary.
- Explain the important reasoning.
- Do not merely dump code.


==================================================
PEDAGOGICAL STRATEGY
==================================================

socratic:
Ask one or two focused questions and encourage reasoning.

scaffolding:
Break the task into manageable conceptual steps.

direct_explanation:
Explain the misconception clearly and explicitly.

challenge:
Use minimal assistance and encourage deeper reasoning.


==================================================
PERSONA
==================================================

Persona controls communication style only.

It may affect:
- warmth
- directness
- encouragement
- pace
- challenge

Persona must NOT override the answer policy.


==================================================
COMMUNICATION QUALITY
==================================================

- Avoid exaggerated praise.
- Praise only specific observable progress.
- Do not tell the learner that you inferred personality.
- Do not make psychological claims.
- Be concise and interactive.
- Stay focused on the assignment objective.


==================================================
DO NOT MENTION
==================================================

- internal agents
- fuzzy logic
- POMDP
- belief state
- hidden state
- JSON
- system prompt


Generate only the tutoring message.
"""

        return prompt.strip()

    # Local Fallback

    def _fallback_response(
        self,
        diagnosis: dict,
        pedagogy: dict,
        preferred_language: str = "en"
    ) -> str:

        preferred_language = (
            "zh"
            if str(
                preferred_language
                or "en"
            ).strip().lower().startswith(
                "zh"
            )
            else
            "en"
        )

        misconception = (
            diagnosis.get(
                "misconception"
            )
        )

        strategy = (
            pedagogy.get(
                "strategy",
                "socratic"
            )
        )

        answer_policy = (
            pedagogy.get(
                "answer_policy",
                "HINT_ONLY"
            )
        )

        zh = (
            preferred_language
            ==
            "zh"
        )

        if not misconception:

            if zh:
                return (
                    "目前你的作答在结构上没有明显问题。"
                    "可以先选一个测试输入，预测程序结果，"
                    "然后再与实际输出进行比较。"
                )

            return (
                "Your code looks structurally reasonable so far. "
                "Try one test input, predict the result first, and "
                "compare your prediction with the program output."
            )

        if misconception == "return_statement":

            if answer_policy in {
                "DIRECT_EXPLANATION",
                "FULL_SOLUTION_ALLOWED"
            }:

                if zh:
                    return (
                        "关键区别在于 `print()` 和 `return`："
                        "`print()` 只是显示一个值，而 `return` "
                        "会把这个值交还给调用者。"
                        "想一想这个函数真正需要哪一种行为。"
                    )

                return (
                    "The key distinction is between `print()` and "
                    "`return`. `print()` displays a value, while "
                    "`return` sends that value back to the caller. "
                    "Consider which behavior this function needs."
                )

            if answer_policy == "PARTIAL_SOLUTION":

                if zh:
                    return (
                        "你已经在函数内部计算出了目标值。"
                        "下一步可以思考：这个值应该怎样离开函数？"
                        "它只需要被显示，还是需要让调用者真正接收到？"
                    )

                return (
                    "You already calculate the value inside the "
                    "function. The next step is deciding how that "
                    "value should leave the function. Should it only "
                    "be displayed, or should the caller receive it?"
                )

            if zh:
                return (
                    "当这个函数执行结束时，调用它的代码实际上会收到什么值？"
                )

            return (
                "When this function finishes, what value does the "
                "calling code actually receive?"
            )

        if misconception == "function_definition":

            if answer_policy == "DIRECT_EXPLANATION":

                if zh:
                    return (
                        "Python 使用 `def` 定义函数，后面依次是函数名、"
                        "参数和冒号，函数体需要在下一行缩进。"
                    )

                return (
                    "Python defines functions using `def`, followed "
                    "by the function name, parameters, and a colon. "
                    "The function body is indented underneath it."
                )

            if zh:
                return (
                    "Python 用哪个关键字来定义一个可以重复调用的函数？"
                )

            return (
                "What keyword does Python use to define a reusable "
                "function?"
            )

        if misconception == "python_syntax":

            if answer_policy == "DIRECT_EXPLANATION":

                if zh:
                    return (
                        "当前代码存在语法问题。"
                        "在调整程序逻辑之前，先检查函数头、标点符号和缩进。"
                    )

                return (
                    "There is a syntax problem in the current code. "
                    "Check the function header, punctuation, and "
                    "indentation before changing the program logic."
                )

            if zh:
                return (
                    "仔细看看 Python 报告语法错误的那一行。"
                    "那里缺少了什么结构、符号或格式？"
                )

            return (
                "Look closely at the statement where Python reports "
                "the syntax problem. What structure or punctuation "
                "does Python expect there?"
            )

        if strategy == "challenge":

            if zh:
                return (
                    "先不要急着修改代码。"
                    "先预测这一部分现在会做什么，再运行它验证你的预测。"
                )

            return (
                "Before changing the code, predict what this section "
                "currently does, then test your prediction."
            )

        if zh:
            return (
                "比较一下这一部分本来应该产生什么结果，"
                "以及当前代码实际上产生了什么结果。"
            )

        return (
            "Compare what this part of the program is supposed to "
            "produce with what the current code actually produces."
        )
