from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def ask(self, content: str, prompt: str) -> str:
        """Sends a text prompt and retrieves the response."""
        pass