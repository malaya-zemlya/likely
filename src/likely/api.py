"""Public API for the likely package."""

import logging

from typesafe_sdk import Noul, TypeSafeClient

logger = logging.getLogger(__name__)


class Likely:
    """Estimate how likely a question is true given some state, via TypeSafe."""

    def __init__(self, client: TypeSafeClient) -> None:
        self._client = client

    def __call__(self, question: str, state: str) -> float:
        """Return how likely ``question`` is true given ``state``, in [0, 1].

        Args:
            question: A natural-language yes/no question.
            state: Context the question should be evaluated against.

        Returns:
            A float in [0.0, 1.0]: the probability of a "yes"/true answer.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not state or not state.strip():
            raise ValueError("state must be a non-empty string")

        logger.debug("Likely called question=%r state=%r", question, state)
        response = self._client.system_one(
            state=state,
            questions={"question": Noul(instructions=question)},
        )
        return response.answers["question"].noul
