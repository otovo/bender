from typing import Callable, Optional

from pandas import DataFrame, Series

from bender.transformation.transformation import (
    BinaryTransform,
    CombineToMean,
    CustomCodeTransformation,
    DateComponent,
    ExponentialShift,
    FillMissingValue,
    FillPolicy,
    Filter,
    LogNormalDistributionShift,
    NeighbourDistance,
    Relation,
    UnpackJson,
    UnpackPolicy,
)


class Transformations:
    @staticmethod
    def log_normal_shift(
        input_feature: str, output_feature: str, input_has_zeros: bool = True
    ) -> LogNormalDistributionShift:
        return LogNormalDistributionShift(input_feature, output_feature, input_has_zeros)

    @staticmethod
    def date_component(component: str, input_feature: str, output_feature: str) -> DateComponent:
        return DateComponent(component, input_feature, output_feature)

    @staticmethod
    def unpack_json(input_feature: str, key: str, output_feature: str, policy: UnpackPolicy) -> UnpackJson:
        return UnpackJson(input_feature, key, output_feature, policy)

    @staticmethod
    def neighour_distance(
        number_of_neighbours: int,
        latitude: str = 'latitude',
        longitude: str = 'longitude',
        to: Optional[Callable[[DataFrame], Series]] = None,
    ) -> NeighbourDistance:
        return NeighbourDistance(number_of_neighbours, latitude, longitude, to)

    @staticmethod
    def binary(output_feature: str, lambda_function: Callable[[DataFrame], Series]) -> BinaryTransform:
        return BinaryTransform(output_feature, lambda_function)

    @staticmethod
    def fill_missing(feature_name: str, policy: FillPolicy) -> FillMissingValue:
        return FillMissingValue(feature_name, policy)

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
