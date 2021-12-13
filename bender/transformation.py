from __future__ import annotations
from typing import Optional, Callable, TypeVar, Any
from pandas import DataFrame
from numpy import log1p, log2
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.series import Series
import pandas as pd
import logging
import numpy as np
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


class Transformation:

    async def transform(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def only_if(self, should_run: bool) -> Transformation:
        return OptionalTransformation(self, should_run=should_run)

ProcessResultType = TypeVar('ProcessResultType')

class Processable:
    def process(self, transformations: list[Transformation]) -> ProcessResultType:
        raise NotImplementedError()


class OptionalTransformation(Transformation):

    should_run: bool
    transformation: Transformation

    def __init__(self, transformation: Transformation, should_run: bool) -> None:
        self.transformation = transformation
        self.should_run = should_run

    async def transform(self, df: DataFrame) -> DataFrame:
        if self.should_run:
            return await self.transformation.transform(df)
        return df


class RemoveDuplicateColumns(Transformation):
    async def transform(self, df: DataFrame) -> DataFrame:
        s = df.columns.to_series()
        df.columns = s + s.groupby(s).cumcount().astype(str).replace({'0': ''})
        return df


class DescribeData(Transformation):

    features: Optional[list[str]]

    def __init__(self, features: Optional[list[str]] = None) -> None:
        self.features = features

    async def transform(self, df: DataFrame) -> DataFrame:
        display_features = list(df.columns)
        if self.features:
            display_features = self.features
        logger.info(df[display_features].describe())
        return df


class LogDataFeatures(Transformation):
    async def transform(self, df: DataFrame) -> DataFrame:
        logger.info(df.columns)
        return df


class LogDataHead(Transformation):

    features: Optional[list[str]]

    def __init__(self, features: Optional[list[str]] = None) -> None:
        self.features = features

    async def transform(self, df: DataFrame) -> DataFrame:
        display_features = list(df.columns)
        if self.features:
            display_features = self.features
        logger.info(df[display_features[0]].iloc[0])
        logger.info(df[display_features].head())
        return df


class LogDataInfo(Transformation):
    async def transform(self, df: DataFrame) -> DataFrame:
        logger.info(df.info())
        return df


class Filter(Transformation):

    lambda_function: Callable[[DataFrame], Series]

    def __init__(self, lambda_function: Callable[[DataFrame], Series]) -> None:
        self.lambda_function = lambda_function

    async def transform(self, df: DataFrame) -> DataFrame:
        return df[self.lambda_function(df)]


class LogNormalDistributionShift(Transformation):

    input_feature: str
    output_feature: str
    input_has_zeros: bool

    def __init__(self, input_feature: str, output_feature: Optional[str] = None, input_has_zeros: bool = True) -> None:
        self.input_feature = input_feature
        self.input_has_zeros = input_has_zeros
        if output_feature:
            self.output_feature = output_feature
        else:
            self.output_feature = f'{input_feature}_log'

    async def transform(self, df: DataFrame) -> DataFrame:
        if self.input_has_zeros:
            df[self.output_feature] = log1p(df[self.input_feature])
        else:
            df[self.output_feature] = log2(df[self.input_feature]) / log2(np.e)
        return df


class DateComponent(Transformation):

    input_feature: str
    output_feature: str
    component: str

    def __init__(self, component: str, input_feature: str, output_feature: Optional[str] = None) -> None:

        self.component = component
        self.input_feature = input_feature
        if output_feature:
            self.output_feature = output_feature
        else:
            self.output_feature = f'{self.input_feature}_{component}'

    async def transform(self, df: DataFrame) -> DataFrame:
        # if df[self.input_feature].dtype not in [datetime, datetime64, pd.Timestamp]:
        df[self.input_feature] = pd.to_datetime(df[self.input_feature], infer_datetime_format=True)
        logger.info(f'dtype for {self.input_feature}: {df[self.input_feature].dtype}')
        df[self.output_feature] = getattr(df[self.input_feature].dt, self.component)
        return df


class BinaryTransform(Transformation):

    lambda_function: Callable[[DataFrame], Series]
    output_feature: str

    def __init__(self, output_feature: str, lambda_function: Callable[[DataFrame], Series]) -> None:
        self.output_feature = output_feature
        self.lambda_function = lambda_function

    async def transform(self, df: DataFrame) -> DataFrame:
        df[self.output_feature] = self.lambda_function(df)
        return df


class SelectFeatures(Transformation):

    features: list[str]

    def __init__(self, features: list[str]) -> None:
        self.features = features

    async def transform(self, df: DataFrame) -> DataFrame:
        return df[self.features]


class DiscardFeatures(Transformation):

    features: list[str]

    def __init__(self, features: list[str]) -> None:
        self.features = features

    async def transform(self, df: DataFrame) -> DataFrame:
        all_features = list(df.columns)
        for remove_feature in self.features:
            if remove_feature in all_features:
                all_features.remove(remove_feature)
        return df[all_features]


class UnpackList(Transformation):

    input_feature: str
    index: int
    output_feature: str

    def __init__(self, input_feature: str, index: int, output_feature: str) -> None:
        self.input_feature = input_feature
        self.index = index
        self.output_feature = output_feature

    async def transform(self, df: DataFrame) -> DataFrame:
        df[self.output_feature] = df[self.input_feature].apply(pd.Series).loc[:, self.index]
        return df


class UnpackTypePolicy:
    def unpack(self, column: Series, key: str) -> Series:
        raise NotImplementedError()

    @staticmethod
    def min_number() -> UnpackTypePolicy:
        return UnpackNumber(lambda df: df.min())

    @staticmethod
    def max_number() -> UnpackTypePolicy:
        return UnpackNumber(lambda df: df.max())

    @staticmethod
    def median_number() -> UnpackTypePolicy:
        return UnpackNumber(lambda df: df.median())

    @staticmethod
    def mean_number() -> UnpackTypePolicy:
        return UnpackNumber(lambda df: df.mean())

    @staticmethod
    def first_string() -> UnpackTypePolicy:
        return UnpackString()


class UnpackNumber(UnpackTypePolicy):

    metric: Callable[[SeriesGroupBy], Series]

    def __init__(self, metric: Callable[[SeriesGroupBy], float]) -> None:
        self.metric = metric

    def unpack(self, column: Series, key: str) -> Series:
        bracket = '}'
        regex_str = rf'"{key}"[\s:]+(\d+)[{bracket},]'
        grouped = column.astype(str).str.extractall(regex_str).reset_index(level='match', drop=True).reset_index()
        grouped.columns = ['index', 'value']
        # Is needed to set type to float, as it can otherwise lead to bugs
        # Like incorrect mean values
        grouped['value'] = grouped['value'].astype(float)
        grouped = grouped.groupby(by='index')

        return self.metric(grouped).astype(float)


class UnpackString(UnpackTypePolicy):
    def unpack(self, column: Series, key: str) -> Series:
        bracket = '}'
        regex_str = rf'"{key}"[\s:]+"([\w ]+)["{bracket},]'
        return column.astype(str).str.extract(regex_str).astype(str)


class UnpackJson(Transformation):

    input_feature: str
    key: str
    output_feature: str
    policy: UnpackTypePolicy

    def __init__(self, input_feature: str, key: str, output_feature: str, policy: UnpackTypePolicy) -> None:
        self.input_feature = input_feature
        self.key = key
        self.output_feature = output_feature
        self.policy = policy

    async def transform(self, df: DataFrame) -> DataFrame:
        df[self.output_feature] = self.policy.unpack(df[self.input_feature], self.key)
        return df


class NeighbourDistance(Transformation):

    to: Optional[Callable[[DataFrame], Series]]
    latitude_key: str
    longitude_key: str
    number_of_neighbours: int

    def __init__(
        self,
        number_of_neighbours: int,
        latitude: str = 'latitude',
        longitude: str = 'longitude',
        to: Optional[Callable[[DataFrame], Series]] = None,
    ) -> None:
        self.latitude_key = latitude
        self.longitude_key = longitude
        self.number_of_neighbours = number_of_neighbours
        self.to = to

    async def transform(self, df: DataFrame) -> DataFrame:
        for column in df[[self.latitude_key, self.longitude_key]]:
            if not isinstance(df[column].values[0], float):
                df[column] = df[column].astype(float)
            rad = np.deg2rad(df[column].values)
            df[f'{column}_rad'] = rad

        if self.to:
            to_points = df[self.to(df)]
        else:
            to_points = df

        ball = BallTree(to_points[[f'{self.latitude_key}_rad', f'{self.longitude_key}_rad']].values, metric='haversine')
        distances, _ = ball.query(
            df[[f'{self.longitude_key}_rad', f'{self.longitude_key}_rad']].values, k=self.number_of_neighbours + 1
        )

        # To km
        distances *= 6371

        distances[distances == 0] = np.nan
        df['distance_neighbor'] = distances.mean(axis=1)

        return df


class DropNaN(Transformation):
    async def transform(self, df: DataFrame) -> DataFrame:
        df.dropna()
        return df


class Relation(Transformation):

    value_key: str
    per_key: str
    output_key: str

    def __init__(self, key: str, per: str, output: str) -> None:
        self.value_key = key
        self.per_key = per
        self.output_key = output

    async def transform(self, df: DataFrame) -> DataFrame:
        df[self.output_key] = df[self.value_key] / df[self.per_key]
        return df


class FillMissingValuePolicy:
    def fill_value(self, feature: str, df: DataFrame) -> Any:
        raise NotImplementedError()

    @staticmethod
    def constant(value: Any) -> ConstantFillValue:
        return ConstantFillValue(value)

    @staticmethod
    def mean(similar_to: Optional[list[str]] = None) -> MeanFillValue:
        return MeanFillValue(similar_to if similar_to is not None else [])


class ConstantFillValue(FillMissingValuePolicy):

    value: Any

    def __init__(self, value: Any) -> None:
        self.value = value

    def fill_value(self, feature: str, df: DataFrame) -> Any:
        return self.value


class MeanFillValue(FillMissingValuePolicy):

    similar_to_features: list[str]

    def __init__(self, similar_to_features: list[str]) -> None:
        self.similar_to_features = similar_to_features

    def fill_value(self, feature: str, df: DataFrame) -> Any:
        if not self.similar_to_features:
            # Find rows most similar to the features
            # and return the mean here
            logger.info('Using based_on_features is not supported yet')
        return df[feature].mean()


class FillMissingValue(Transformation):

    value_key: str
    policy: FillMissingValuePolicy

    def __init__(self, key: str, policy: FillMissingValuePolicy) -> None:
        self.value_key = key
        self.policy = policy

    async def transform(self, df: DataFrame) -> DataFrame:
        df.loc[df[self.value_key].isnull(), self.value_key] = self.policy.fill_value(self.value_key, df)
        return df


class CustomCodeTransformation(Transformation):

    transformation: Callable[[DataFrame], DataFrame]

    def __init__(self, transformation: Callable[[DataFrame], DataFrame]) -> None:
        self.transformation = transformation

    async def transform(self, df: DataFrame) -> DataFrame:
        return self.transformation(df)


class CombineToMean(Transformation):

    output_key: str
    features: list[str]

    def __init__(self, output_key: str, features: list[str]) -> None:
        self.features = features
        self.output_key = output_key

    async def transform(self, df: DataFrame) -> DataFrame:
        df[self.output_key] = df[self.features].mean(axis=1)
        return df
