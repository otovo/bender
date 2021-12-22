from typing import Any

from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier

from bender.trained_model.interface import TrainedModel, TrainedProbabilisticClassificationModel


class TrainedKNeighboursClassifier(TrainedProbabilisticClassificationModel):

    model: KNeighborsClassifier
    input_features: list[str]

    def __init__(self, model: KNeighborsClassifier, input_features: list[str]) -> None:
        self.model = model
        self.input_features = input_features

    def class_names(self) -> list[Any]:
        return list(self.model.classes_)

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)

    def _predict_proba_on_valid(self, data: DataFrame) -> DataFrame:
        return self.model.predict_proba(data)

    def to_json(self) -> str:
        # path = f'tmp-xgboost-{uuid4()}.json'
        # self.model.save_model(path)
        # with open(path) as file:
        #     model_json = file.read()
        # return json.dumps({'input_features': self.input_features, 'model': model_json})
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainedModel:
        raise NotImplementedError()
        # if 'model' not in data:
        #     raise Exception('Unsupported format')

        # temp_file_path = f'tmp-xgboost-{uuid4()}.json'
        # with open(temp_file_path, 'w') as temp_file:
        #     temp_file.write(data['model'])

        # model = XGBClassifier()
        # model.load_model(temp_file_path)
        # return TrainedXGBoostModel(model, data['input_features'])
