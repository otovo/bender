from bender.evaluator.confusion_matrix import ConfusionMatrix
from bender.evaluator.correlation_matrix import CorrelationMatrix
from bender.evaluator.feature_importance import XGBoostFeatureImportance
from bender.evaluator.precision_recall import PrecisionRecall
from bender.evaluator.predict_probability import PredictProbability
from bender.evaluator.roc import RocCurve
from bender.exporter.exporter import Exporter


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
