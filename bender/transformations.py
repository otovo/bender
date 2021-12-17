from typing import Callable, Optional

from pandas import DataFrame, Series

from bender.transformation.transformation import (
    BinaryTransform,
    DateComponent,
    LogNormalDistributionShift,
    NeighbourDistance,
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
