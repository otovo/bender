from enum import Enum, EnumMeta
from typing import Any, Callable, Optional, TypeVar

from pandas.core.frame import DataFrame, Series


class MetaEnum(EnumMeta):
    """
    Overwrites __contains__ so we can do `"string" in Enum`.
    """

    def __contains__(cls, item: Any) -> bool:
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class StrEnum(str, BaseEnum):
    def __str__(self) -> str:
        return self.value  # type: ignore

    @classmethod
    def choices(cls) -> list['StrEnum']:
        return list(cls)


class PredictionOutput(StrEnum):

    PROBABILITY = 'probability'
    CLASSIFICATION = 'classification'


class PredictionExtractor:
    @property
    def output(self) -> tuple[str, PredictionOutput]:
        raise NotImplementedError()

    @property
    def needed_features(self) -> set[str]:
        raise NotImplementedError()

    async def extract(self, output: DataFrame) -> None:
        raise NotImplementedError()


PredictableType = TypeVar('PredictableType')


class Predictable:
    def predict(self, on: Optional[Callable[[DataFrame], Series]] = None) -> PredictableType:
        raise NotImplementedError()


ProbPredictableType = TypeVar('ProbPredictableType')


class ProbabilisticPredictable:
    def predict_proba(self, on: Optional[Callable[[DataFrame], Series]] = None) -> ProbPredictableType:
        raise NotImplementedError()
