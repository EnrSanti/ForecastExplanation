from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    """Simplifies the rules of a learned model, after learning and before it
    is scored, using only the model's training rows."""

    @abstractmethod
    def clean(self, model, data: list[list]) -> None:
        """
        Simplifies the rules of `model` in place.

        Args:
            model: The learned model, as returned by the learning framework.
            data: The rows the model was trained on (label last).
        """
