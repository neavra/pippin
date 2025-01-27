"""Activity for generating daily workout using OpenAI."""

import logging
from datetime import timedelta
from framework.activity_decorator import activity, ActivityBase, ActivityResult
from skills.skill_chat import chat_skill

logger = logging.getLogger(__name__)


@activity(
    name="daily_workout_plan",
    energy_cost=0.4,
    cooldown=1800,  # 30 minutes
    required_skills=["openai_chat"],
)
class DailyWorkoutPlan(ActivityBase):
    """Generates daily workout plans using OpenAI."""

    def __init__(self):
        super().__init__()
        self.system_prompt = """You are an Amatuer person trying to learn how to box as a way to keep fit. Keep responses concise (2-3 sentences) and 
        focused on how to improve on your boxing technique, cardio, or learn about new ways to train."""

    async def execute(self, shared_data) -> ActivityResult:
        """Execute the daily workout plan activity."""
        try:
            logger.info("Starting daily workout plan generation")

            # Initialize required skills
            if not await chat_skill.initialize():
                return ActivityResult.error_result("Failed to initialize chat skill")

            # Generate the thought
            result = await chat_skill.get_chat_completion(
                prompt="Generate a workout plan for today. Focus on boxing technique, cardio, or learn about new ways to train. Think out loud and in first person",
                system_prompt=self.system_prompt,
                max_tokens=100,
            )

            if not result["success"]:
                return ActivityResult.error_result(result["error"])

            return ActivityResult.success_result(
                data={"thought": result["data"]["content"]},
                metadata={
                    "model": result["data"]["model"],
                    "finish_reason": result["data"]["finish_reason"],
                },
            )

        except Exception as e:
            logger.error(f"Error in daily workout plan activity: {e}")
            return ActivityResult.error_result(str(e))
