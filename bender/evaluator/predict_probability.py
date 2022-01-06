import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trained_model.interface import TrainedModel, TrainedProbabilisticClassificationModel

logger = logging.getLogger(__name__)


class ProbabilityForClassification(Evaluator):

    num_bins: Optional[int]
    classification_of_interest: Optional[Any]
    exporter: Exporter

    def __init__(
        self, exporter: Exporter, classification_of_interest: Optional[Any] = None, num_bins: Optional[int] = None
    ) -> None:
        self.exporter = exporter
        self.classification_of_interest = classification_of_interest
        self.num_bins = num_bins

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        if not isinstance(model, TrainedProbabilisticClassificationModel):
            logger.info(
                'ProbabilityForClassification Evaluator will only work for TrainedProbabilisticClassificationModel'
            )
            return
        pred_result = model.predict_proba(data_set.x_validate)
        true_results = data_set.y_validate.reset_index(drop=True)
        if set(true_results.unique().tolist()) == {0, 1}:
            true_results = true_results.astype(bool)

        labels = true_results.unique()
        # Scores compared to true labels
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        n_bins = 50 if self.num_bins is None else self.num_bins
        bin_width = 1 / n_bins
        plt.xlim(0, 1)
        sns.histplot(
            data=pred_result,
            x=self.classification_of_interest,
            hue=true_results,
            multiple='stack',
            binwidth=bin_width,
            legend=False,
            ax=ax,
        )
        plt.legend(title='True Values', labels=[str(label) for label in labels])
        await self.exporter.store_figure(fig)
