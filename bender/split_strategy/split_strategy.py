from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from pandas import DataFrame, Series


@dataclass
class TrainingDataSet:

    x_features: list[str]
    y_feature: str

    train: DataFrame
    validation: DataFrame

    @property
    def y_train(self) -> Series:
        return self.train[self.y_feature]

    @property
    def y_validate(self) -> Series:
        return self.validation[self.y_feature]

    @property
    def x_train(self) -> DataFrame:
        return self.train[self.x_features]

    @property
    def x_validate(self) -> DataFrame:
        return self.validation[self.x_features]


class SplitStrategy:
    async def split(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        raise NotImplementedError()


class RandomRatioSplitter(SplitStrategy):

    ratio: float

    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    async def split(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        split_index = int(len(df) * self.ratio)

        train = df[:split_index]
        validate = df[split_index:]

        return train, validate


class SortedRatioSplitter(SplitStrategy):

    sort_key: str
    ratio: float

    def __init__(self, ratio: float, sort_key: str) -> None:
        self.ratio = ratio
        self.sort_key = sort_key

    async def split(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:
        split_index = int(len(df) * self.ratio)
        dates = df[self.sort_key]
        sorted_index = [x for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])]

        train_index = sorted_index[:split_index]
        test_index = sorted_index[split_index:]

        train: DataFrame = df.iloc[train_index]
        validate: DataFrame = df.iloc[test_index]

        return train, validate


SplitableType = TypeVar('SplitableType')


class Splitable(Generic[SplitableType]):
    def split(self, split_strategy: SplitStrategy) -> SplitableType:
        raise NotImplementedError()
