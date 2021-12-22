from typing import Any, Optional

from bender.evaluator.confusion_matrix import ConfusionMatrix
from bender.evaluator.difference_distribution import DifferenceDistribution, DifferenceMetric
from bender.evaluator.feature_importance import XGBoostFeatureImportance
from bender.evaluator.precision_recall import PrecisionRecall
from bender.evaluator.predict_probability import ProbabilityForClassification
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
    def probability_for(
        classification: Optional[Any] = None, num_bins: Optional[int] = None, exporter: Exporter = Exporter.in_memory()
    ) -> ProbabilityForClassification:
        return ProbabilityForClassification(exporter, classification, num_bins)

    @staticmethod
    def feature_importance(exporter: Exporter = Exporter.in_memory()) -> XGBoostFeatureImportance:
        return XGBoostFeatureImportance(exporter)

    @staticmethod
    def precision_recall(exporter: Exporter = Exporter.in_memory()) -> PrecisionRecall:
        return PrecisionRecall(exporter)

    @staticmethod
    def error_distribution(
        metric: DifferenceMetric, exporter: Exporter = Exporter.in_memory()
    ) -> DifferenceDistribution:
        return DifferenceDistribution(metric, exporter)
