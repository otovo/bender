from __future__ import annotations

import json
from typing import Any, TypeVar

import numpy as np
from pandas.core.frame import DataFrame, Series
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from bender.split_strategy import TrainingDataSet


class TrainedModel:

    used_features: list[str]

    def predict(self, data: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def predict_proba(self, data: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def loss(self, data: DataFrame) -> float:
        raise NotImplementedError()

    def estimator(self) -> Pipeline:
        raise NotImplementedError()

    def validate_data(self, data: DataFrame):
        feature_set = set(self.used_features)
        intersection = set(data.columns).intersection(feature_set)
        if len(intersection) != len(feature_set):
            raise Exception(f'Missing the following features: {feature_set - intersection}')

    def to_json(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainedModel:
        raise NotImplementedError()


class ModelTrainer:
    async def train(self, data_split: TrainingDataSet) -> TrainedModel:
        raise NotImplementedError()

    @staticmethod
    def xgboost() -> XGBoostTrainer:
        return XGBoostTrainer()


TrainableType = TypeVar('TrainableType')

class Trainable:

    def train(self, model: ModelTrainer, input_features: set[str], target_feature: str) -> TrainableType:
        raise NotImplementedError()

class TrainedXGBoostModel(TrainedModel):

    model: XGBClassifier
    used_features: list[str]

    def __init__(self, model: XGBClassifier, used_features: list[str]) -> None:
        self.model = model
        self.used_features = used_features

    def predict(self, data: DataFrame) -> DataFrame:
        self.validate_data(data)
        return self.model.predict(data)

    def predict_proba(self, data: DataFrame) -> DataFrame:
        self.validate_data(data)
        return self.model.predict_proba(data)[:, 1]

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
        model = XGBClassifier(**self.xgboost_parmas)
        if isinstance(data_split.y_train, DataFrame):
            model.scale_pos_weight = int(
                np.round(data_split.x_train.shape[0] / data_split.y_train[data_split.y_train.columns[0]].sum() - 1)
            )
        elif isinstance(data_split.y_train, Series):
            model.scale_pos_weight = int(np.round(data_split.x_train.shape[0] / data_split.y_train.sum() - 1))
        model.fit(data_split.x_train, data_split.y_train, eval_set=[(data_split.x_validate, data_split.y_validate)])
        return TrainedXGBoostModel(model, data_split.x_features)
