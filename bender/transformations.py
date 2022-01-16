from typing import Any, Callable, Optional, Union

from pandas import DataFrame, Series

from bender.transformation.schema import SchemaTransformation, SchemaType
from bender.transformation.transformation import (
    BinaryTransform,
    BinFeature,
    CombineToMean,
    CustomCodeTransformation,
    DateComponent,
    ExponentialShift,
    FillMissingValue,
    FillPolicy,
    Filter,
    LogNormalDistributionShift,
    LogToConsole,
    NeighbourDistance,
    Relation,
    SetIndex,
    SplitString,
    ToCatagorical,
    UnpackJson,
    UnpackPolicy,
)


class Transformations:
    @staticmethod
    def log_normal_shift(input: str, output: str, input_has_zeros: bool = True) -> LogNormalDistributionShift:
        return LogNormalDistributionShift(input, output, input_has_zeros)

    @staticmethod
    def date_component(component: str, input: str, output: str) -> DateComponent:
        return DateComponent(component, input, output)

    @staticmethod
    def unpack_json(input: str, key: str, output: str, policy: UnpackPolicy) -> UnpackJson:
        return UnpackJson(input, key, output, policy)

    @staticmethod
    def neighour_distance(
        number_of_neighbours: int,
        latitude: str = 'latitude',
        longitude: str = 'longitude',
        to: Optional[Callable[[DataFrame], Series]] = None,
    ) -> NeighbourDistance:
        return NeighbourDistance(number_of_neighbours, latitude, longitude, to)

    @staticmethod
    def compute(output: str, computation: Callable[[DataFrame], Series]) -> BinaryTransform:
        return BinaryTransform(output, computation)

    @staticmethod
    def fill_missing(feature: str, policy: FillPolicy) -> FillMissingValue:
        return FillMissingValue(feature, policy)

    @staticmethod
    def to_catigorical(feature: str, output: Optional[str] = None) -> ToCatagorical:
        return ToCatagorical(feature, feature if output is None else output)

    @staticmethod
    def log_to_console(data: Callable[[DataFrame], Any]) -> LogToConsole:
        return LogToConsole(data)

    @staticmethod
    def filter(lambda_function: Callable[[DataFrame], Series]) -> Filter:
        return Filter(lambda_function)

    @staticmethod
    def ratio(numerator: str, denumirator: str, output: str) -> Relation:
        return Relation(numerator, denumirator, output)

    @staticmethod
    def exp_shift(value: str, output: str) -> ExponentialShift:
        return ExponentialShift(value, output)

    @staticmethod
    def custom(code: Callable[[DataFrame], DataFrame]) -> CustomCodeTransformation:
        return CustomCodeTransformation(code)

    @staticmethod
    def combined_mean(features: set[str], output: str) -> CombineToMean:
        return CombineToMean(output, list(features))

    @staticmethod
    def bin(feature: str, n_bins: int, output: str) -> BinFeature:
        return BinFeature(feature, output, n_bins)

    @staticmethod
    def set_index(feature: str) -> SetIndex:
        return SetIndex(feature)

    @staticmethod
    def split_string(
        feature: str, seperator: str, output: Union[list[str], str], select_number: int = 1
    ) -> SplitString:
        if isinstance(output, str):
            output = [output]
        return SplitString(feature, output, seperator, select_number)

    @staticmethod
    def schema(schema: dict[str, SchemaType]) -> SchemaTransformation:
        return SchemaTransformation(schema)
