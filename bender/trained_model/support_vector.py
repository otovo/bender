from pandas import DataFrame, Series
from sklearn.svm import SVR

from bender.trained_model.interface import TrainedRegressionModel


class SupportVectorRegression(TrainedRegressionModel):

    model: SVR
    input_features: list[str]

    def __init__(self, model: SVR, input_features: list[str]) -> None:
        self.input_features = input_features
        self.model = model

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)
