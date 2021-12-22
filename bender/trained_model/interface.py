from __future__ import annotations

from typing import Any

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline


class TrainedModel:

    input_features: list[str]

    def _valid_data(self, data: DataFrame) -> DataFrame:
        feature_set = set(self.input_features)
        intersection = set(data.columns).intersection(feature_set)
        if len(intersection) != len(feature_set):
            raise Exception(f'Missing the following features: {feature_set - intersection}')
        return data[self.input_features]

    def predict(self, data: DataFrame) -> Series:
        return self._predict_on_valid(self._valid_data(data))

    def _predict_on_valid(self, data: DataFrame) -> Series:
        raise NotImplementedError()

    def to_json(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainedModel:
        raise NotImplementedError()


class TrainedEstimatorModel(TrainedModel):
    def estimator(self) -> Pipeline:
        raise NotImplementedError()


class TrainedRegressionModel(TrainedModel):
    """A trained model that preduces regression outputs

    Therefor representing a transformation from
    Vector[float | int] -> float
    """

    pass


class TrainedProbabilisticModel(TrainedModel):
    def predict_proba(self, data: DataFrame) -> DataFrame:
        return self._predict_proba_on_valid(self._valid_data(data))

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        raise NotImplementedError()


class TrainedClassificationModel(TrainedModel):
    """A trained model that preduces classification outputs

    Therefor representing a transformation from
    Vector[float | int | bool | str | date | datetime] -> int | bool | str
    """

    def class_names(self) -> list[Any]:
        raise NotImplementedError()


class TrainedProbabilisticClassificationModel(TrainedProbabilisticModel, TrainedClassificationModel):
    def predict_proba(self, data: DataFrame) -> DataFrame:
        label_indicies = self.class_names()
        predictions = self._predict_proba_on_valid(self._valid_data(data))
        return DataFrame(data=predictions, columns=label_indicies)

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        raise NotImplementedError()
