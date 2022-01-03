import json
from typing import Any
from uuid import uuid4

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from bender.trained_model.interface import TrainedEstimatorModel, TrainedModel, TrainedProbabilisticClassificationModel


class TrainedXGBoostModel(TrainedProbabilisticClassificationModel, TrainedEstimatorModel):
    """A Trained XGBoosted tree

    Takes a Vector -> Int
    """

    model: XGBClassifier
    input_features: list[str]

    def __init__(self, model: XGBClassifier, input_features: list[str]) -> None:
        self.model = model
        self.input_features = input_features

    def class_names(self) -> list[Any]:
        return list(self.model.classes_)

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        return self.model.predict_proba(data)

    def estimator(self) -> Pipeline:
        return self.model

    def to_json(self) -> str:
        path = f'tmp-xgboost-{uuid4()}.json'
        self.model.save_model(path)
        with open(path) as file:
            model_json = file.read()
        return json.dumps({'input_features': self.input_features, 'model': model_json, 'name': 'xgboost'})

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainedModel:

        if 'model' not in data:
            raise Exception('Unsupported format')

        temp_file_path = f'tmp-xgboost-{uuid4()}.json'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(data['model'])

        model = XGBClassifier()
        model.load_model(temp_file_path)
        return TrainedXGBoostModel(model, data['input_features'])
