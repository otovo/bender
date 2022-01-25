from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

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


class UniformSplitRatio(SplitStrategy):

    start_offset: float
    ratio: float
    group_by: str

    def __init__(self, ratio: float, group_by: str, start_offset: float) -> None:
        self.ratio = ratio
        self.group_by = group_by
        self.start_offset = start_offset

    async def split(self, df: DataFrame) -> tuple[DataFrame, DataFrame]:

        train: Optional[DataFrame] = None
        validate: Optional[DataFrame] = None

        for index, group_value in enumerate(df[self.group_by].unique()):
            rows = df.loc[df[self.group_by] == group_value]

            train_ranges: list[tuple[float, float]]
            validate_ranges: list[tuple[float, float]]
            if self.start_offset + self.ratio > 1:
                start = self.start_offset + self.ratio - 1
                train_ranges = [(0, start), (self.start_offset, 1)]
                validate_ranges = [(start, self.start_offset)]
            else:
                train_ranges = [(self.start_offset, self.start_offset + self.ratio)]
                validate_ranges = [(0, self.start_offset), (self.start_offset + self.ratio, 1)]

            for train_range in train_ranges:

                split_start_index = int(round(len(rows) * train_range[0]))
                split_end_index = int(round(len(rows) * train_range[1]))

                if split_end_index == split_start_index:
                    sub_train = rows[split_start_index : split_end_index + 1]
                elif split_start_index == 0:
                    sub_train = rows[:split_end_index]
                elif split_end_index == len(rows):
                    sub_train = rows[split_start_index:]
                else:
                    sub_train = rows[split_start_index:split_end_index]

                if index != 0:
                    train = train.append(sub_train)  # type: ignore
                else:
                    train = sub_train

            for validate_range in validate_ranges:

                split_start_index = int(round(len(rows) * validate_range[0]))
                split_end_index = int(round(len(rows) * validate_range[1]))

                if split_end_index == split_start_index:
                    sub_validate = rows[split_start_index : split_end_index + 1]
                elif split_start_index == 0:
                    sub_validate = rows[:split_end_index]
                elif split_end_index == len(rows):
                    sub_validate = rows[split_start_index:]
                else:
                    sub_validate = rows[split_start_index:split_end_index]

                if index != 0:
                    validate = validate.append(sub_validate)  # type: ignore
                else:
                    validate = sub_validate

        return train, validate


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
        sorted_index = [
            x
            for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])  # type: ignore
        ]

        train_index = sorted_index[:split_index]
        test_index = sorted_index[split_index:]

        train: DataFrame = df.iloc[train_index]
        validate: DataFrame = df.iloc[test_index]

        return train, validate


SplitableType = TypeVar('SplitableType')


class Splitable(Generic[SplitableType]):
    def split(self, split_strategy: SplitStrategy) -> SplitableType:
        raise NotImplementedError()
