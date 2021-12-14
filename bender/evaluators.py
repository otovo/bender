from bender.evaluator import RocCurve, ConfusionMatrix, CorrelationMatrix, PredictProbability, XGBoostFeatureImportance, PrecisionRecall
from bender.exporter import Exporter


class Evaluators:

    @staticmethod
    def roc_curve(exporter: Exporter = Exporter.in_memory()) -> RocCurve:
        return RocCurve(exporter)

    @staticmethod
    def confusion_matrix(exporter: Exporter = Exporter.in_memory()) -> ConfusionMatrix:
        return ConfusionMatrix(exporter)

    @staticmethod
    def correlation_matrix(exporter: Exporter = Exporter.in_memory()) -> CorrelationMatrix:
        return CorrelationMatrix(exporter)

    @staticmethod
    def predict_probability(exporter: Exporter = Exporter.in_memory()) -> PredictProbability:
        return PredictProbability(exporter)

    @staticmethod
    def feature_importance(exporter: Exporter = Exporter.in_memory()) -> XGBoostFeatureImportance:
        return XGBoostFeatureImportance(exporter)

    @staticmethod
    def precision_recall(exporter: Exporter = Exporter.in_memory()) -> PrecisionRecall:
        return PrecisionRecall(exporter)