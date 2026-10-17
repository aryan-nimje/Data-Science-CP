"""NeMo Guardrails integration using the LLMRails API."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from nemoguardrails import LLMRails, RailsConfig

from app.config import GUARDRAILS_DIR


class GuardedLLM:
    def __init__(self, llm: BaseChatModel):
        config = RailsConfig.from_path(str(GUARDRAILS_DIR))
        self.rails = LLMRails(config, llm=llm)

    def generate(self, messages: list[BaseMessage]) -> str:
        payload = []
        for m in messages:
            if m.type == "human":
                payload.append({"role": "user", "content": str(m.content)})
            elif m.type == "ai":
                payload.append({"role": "assistant", "content": str(m.content)})

        result = self.rails.generate(messages=payload)

        if isinstance(result, dict):
            return result.get("content", "")
        # Newer versions may return a GenerationResponse object
        response = getattr(result, "response", None)
        if isinstance(response, list) and response:
            return response[-1].get("content", "")
        return str(result)