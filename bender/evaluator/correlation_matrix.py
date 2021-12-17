import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bender.evaluator.interface import Evaluator
from bender.exporter.exporter import Exporter
from bender.split_strategy.split_strategy import TrainingDataSet
from bender.trainer.model_trainer import TrainedModel

logger = logging.getLogger(__name__)


class CorrelationMatrix(Evaluator):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def evaluate(self, model: TrainedModel, data_set: TrainingDataSet) -> None:
        corr_heatmap = data_set.x_train.append(data_set.x_validate).corr()
        corr_threshold = 0.9
        for feature in data_set.x_features:
            is_feature_mask = corr_heatmap.columns == feature
            column_values = corr_heatmap[corr_heatmap.columns == feature]
            heatmap_mask = ((column_values > corr_threshold) | (column_values < -corr_threshold)) & (~is_feature_mask)
            mask = heatmap_mask.iloc[0]
            correlated_featrues = corr_heatmap.columns[mask]
            if len(correlated_featrues) == 0:
                continue
            new_mask = column_values.columns.isin(correlated_featrues)
            logger.info('Warning: Correlated features should be considered to be removed')
            logger.info(
                f'{feature} is related to {correlated_featrues}, corr values: {column_values.iloc[0][new_mask]}'
            )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_heatmap,
            mask=np.zeros_like(corr_heatmap, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax,
        )
        ax.set_title('Correlation Matrix')
        await self.exporter.store_figure(fig)
