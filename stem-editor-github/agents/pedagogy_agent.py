from pedagogy.planner import (
    PedagogyPlanner
)


class PedagogyAgent:

    def __init__(self):

        self.planner = (
            PedagogyPlanner()
        )

    def choose_strategy(
        self,
        diagnosis: dict,
        confusion_level: float = 0.3,
        learning_preference: str = "guided",
        teacher_max_hint_level: int = 5,
        dynamic_profile: dict | None = None,
        mastery_score: float = 0.5,
        recent_episodes: list | None = None
    ) -> dict:

        return (
            self.planner
            .create_plan(

                diagnosis=
                    diagnosis,

                confusion_level=
                    confusion_level,

                learning_preference=
                    learning_preference,

                teacher_max_hint_level=
                    teacher_max_hint_level,

                dynamic_profile=
                    dynamic_profile,

                mastery_score=
                    mastery_score,

                recent_episodes=
                    recent_episodes
            )
        )