import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from bender.explorer.interface import Explorer
from bender.exporter.exporter import Exporter

logger = logging.getLogger(__name__)


class CorrelationMatrix(Explorer):

    exporter: Exporter
    features: Optional[list[str]]

    def __init__(self, features: Optional[list[str]], exporter: Exporter) -> None:
        self.exporter = exporter
        self.features = features

    async def explore(self, df: DataFrame) -> None:
        if self.features:
            corr_heatmap = df[self.features].corr()
        else:
            corr_heatmap = df.corr()
        corr_threshold = 0.9
        for feature in corr_heatmap.columns:
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
