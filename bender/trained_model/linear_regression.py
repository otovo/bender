from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from bender.trained_model.interface import TrainedRegressionModel


class TrainedLinearRegression(TrainedRegressionModel):

    model: LinearRegression
    input_features: list[str]

    def __init__(self, model: LinearRegression, input_features: list[str]) -> None:
        self.model = model
        self.input_features = input_features

    def _predict_on_valid(self, data: DataFrame) -> Series:
        return self.model.predict(data)
