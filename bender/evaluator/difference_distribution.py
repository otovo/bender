import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series
from pandas.core.frame import DataFrame

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel


class DifferenceMetric:
    def difference(self, predict: Series, true_value: Series) -> Series:
        raise NotImplementedError()


class SqareError(DifferenceMetric):
    def difference(self, predict: Series, true_value: Series) -> Series:
        return (true_value - predict) ** 2


class AbsoluteError(DifferenceMetric):
    def difference(self, predict: Series, true_value: Series) -> Series:
        return (true_value - predict).abs()


class AbsolutePercentageError(DifferenceMetric):
    def difference(self, predict: Series, true_value: Series) -> Series:
        return ((true_value - predict) / true_value).abs()


class DifferenceDistribution(Evaluator):

    diff_metrics: DifferenceMetric
    exporter: Exporter

    def __init__(self, diff_metrics: DifferenceMetric, exporter: Exporter) -> None:
        self.diff_metrics = diff_metrics
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        pred = model.predict(data_set.x_validate)
        df = DataFrame(data=[], columns=[])
        df['diff'] = self.diff_metrics.difference(pred, data_set.y_validate)

        fig, ax = plt.subplots()
        sns.histplot(df, x='diff', multiple='stack', ax=ax)
        await self.exporter.store_figure(fig)
