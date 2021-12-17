from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier

from bender.trained_model.interface import TrainedClassificationModel


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

    def to_json(self) -> str:
        # print("Printing tree")
        # object = self.model.tree_
        # methods = [method_name for method_name in dir(object) if callable(getattr(object, method_name))]
        # print(methods)
        # print(object.decision_path(np.array([[0, 1]], dtype=np.float32)))
        raise NotImplementedError()
