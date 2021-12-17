import matplotlib.pyplot as plt
from xgboost import plot_importance

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel
from bender.trained_model.xgboosted_tree import TrainedXGBoostModel


class XGBoostFeatureImportance(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        if not isinstance(model, TrainedXGBoostModel):
            raise Exception('Only supporting feature importance for XGBoost models')

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        plot_importance(model.model, ax=ax)
        await self.exporter.store_figure(fig)
