from bender.evaluator import RocCurve, ConfusionMatrix, CorrelationMatrix, PredictProbability, XGBoostFeatureImportance, PrecisionRecall
from bender.exporter import Exporter


class Evaluators:

    @staticmethod
    def roc_curve(exporter: Exporter) -> RocCurve:
        return RocCurve(exporter)

    @staticmethod
    def confusion_matrix(exporter: Exporter) -> ConfusionMatrix:
        return ConfusionMatrix(exporter)

    @staticmethod
    def correlation_matrix(exporter: Exporter) -> CorrelationMatrix:
        return CorrelationMatrix(exporter)

    @staticmethod
    def predict_probability(exporter: Exporter) -> PredictProbability:
        return PredictProbability(exporter)

    @staticmethod
    def feature_importance(exporter: Exporter) -> XGBoostFeatureImportance:
        return XGBoostFeatureImportance(exporter)

    @staticmethod
    def precision_recall(exporter: Exporter) -> PrecisionRecall:
        return PrecisionRecall(exporter)