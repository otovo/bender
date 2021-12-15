from typing import Generic, TypeVar

from bender.evaluator.interface import Evaluator

EvaluableType = TypeVar('EvaluableType')


class Evaluable(Generic[EvaluableType]):
    def evaluate(self, evaluators: list[Evaluator]) -> EvaluableType:
        raise NotImplementedError()
