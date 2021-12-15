from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

import numpy as np
from pandas.core.frame import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from bender.split_strategy.split_strategy import TrainingDataSet


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


class TrainedRegressionModel(TrainedModel):
    """A trained model that preduces regression outputs

    Therefor representing a transformation from
    Vector[float | int] -> float
    """

    pass


class TrainedClassificationModel(TrainedModel):
    """A trained model that preduces classification outputs

    Therefor representing a transformation from
    Vector[float | int | bool | str | date | datetime] -> int | bool | str
    """

    def classification_indicies(self) -> list[str]:
        raise NotImplementedError()

    def predict_proba(self, data: DataFrame) -> DataFrame:
        label_indicies = self.classification_indicies()
        predictions = self._predict_proba_on_valid(self._valid_data(data))
        return DataFrame(data=predictions, columns=label_indicies)

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        raise NotImplementedError()


class ModelTrainer:
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        raise NotImplementedError()


TrainableType = TypeVar('TrainableType')


class Trainable(Generic[TrainableType]):
    def train(self, model: ModelTrainer, input_features: set[str], target_feature: str) -> TrainableType:
        raise NotImplementedError()


class TrainedDecisionTreeClassifier(TrainedClassificationModel):

    model: DecisionTreeClassifier

    def __init__(self, model: DecisionTreeClassifier, input_features: list[str]) -> None:
        self.model = model
        self.input_features = input_features

    def classification_indicies(self) -> list[str]:
        return [str(label) for label in self.model.classes_]

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        return self.model.predict_proba(data)

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)


class DecisionTreeClassifierTrainer(ModelTrainer):
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        model = DecisionTreeClassifier()
        model.fit(data_split.x_train, data_split.y_train)
        return TrainedDecisionTreeClassifier(model, data_split.x_features)


class TrainedXGBoostModel(TrainedClassificationModel):
    """A Trained XGBoosted tree

    Takes a Vector -> Int
    """

    model: XGBClassifier
    used_features: list[str]

    def __init__(self, model: XGBClassifier, used_features: list[str]) -> None:
        self.model = model
        self.used_features = used_features

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        return self.model.predict_proba(data)

    def loss(self, data: DataFrame) -> float:
        return self.model.evals_result()['validation_0']['logloss'][-1]

    def estimator(self) -> Pipeline:
        return self.model

    def to_json(self) -> str:
        params = self.model.get_xgb_params()
        return json.dumps({'used_features': self.used_features, 'params': params})

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainedModel:

        if 'model' not in data:
            raise Exception('Unsupported format')

        temp_file_path = 'tmp-xgboost.json'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(data['model'])

        model = XGBClassifier()
        model.load_model(temp_file_path)
        return TrainedXGBoostModel(model, used_features=data['used_features'])


class XGBoostTrainer(ModelTrainer):

    xgboost_parmas: dict[str, Any]

    def __init__(
        self,
        use_label_encoder=False,
        learning_rate=0.01,
        max_depth=5,
        n_estimators=400,
        verbosity=0,
        scale_pos_weight=1.0,
        gamma=0,
        min_child_weight=1,
        colsample_bytree=1,
        reg_lambda=1,
        alpha=0,
    ) -> None:
        self.xgboost_parmas = {
            'use_label_encoder': use_label_encoder,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'verbosity': verbosity,
            'scale_pos_weight': scale_pos_weight,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'alpha': alpha,
        }

    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        if data_split.y_train.dtype not in [int, bool, str]:
            raise Exception('Training classification model on continuse values. Maybe you want a regression model?')
        model = XGBClassifier(**self.xgboost_parmas)
        if isinstance(data_split.y_train, DataFrame):
            model.scale_pos_weight = int(
                np.round(data_split.x_train.shape[0] / data_split.y_train[data_split.y_train.columns[0]].sum() - 1)
            )
        elif isinstance(data_split.y_train, Series):
            model.scale_pos_weight = int(np.round(data_split.x_train.shape[0] / data_split.y_train.sum() - 1))
        model.fit(data_split.x_train, data_split.y_train, eval_set=[(data_split.x_validate, data_split.y_validate)])
        return TrainedXGBoostModel(model, data_split.x_features)


# class TrainedKMeansClustering(TrainedModel):
#     """A trained kmeans model
#     Takes a Vector -> Int (cluster number)
#     """

#     model: KMeans
#     used_features: list[str]

#     def predict(self, data: DataFrame) -> DataFrame:
#         self.validate_data(data)
#         self.model.predict()
#         raise NotImplementedError()

#     def predict_proba(self, data: DataFrame) -> DataFrame:
#         raise NotImplementedError()

#     def loss(self, data: DataFrame) -> float:
#         raise NotImplementedError()

#     def estimator(self) -> Pipeline:
#         raise NotImplementedError()

#     def validate_data(self, data: DataFrame):
#         feature_set = set(self.used_features)
#         intersection = set(data.columns).intersection(feature_set)
#         if len(intersection) != len(feature_set):
#             raise Exception(f'Missing the following features: {feature_set - intersection}')

#     def to_json(self) -> str:
#         raise NotImplementedError()

#     @staticmethod
#     def from_dict(data: dict[str, Any]) -> TrainedModel:
#         raise NotImplementedError()
