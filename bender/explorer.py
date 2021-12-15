from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from bender.exporter.exporter import Exporter

logger = logging.getLogger(__name__)


class Explorer:
    async def explore(self, df: DataFrame) -> None:
        raise NotImplementedError()

    @staticmethod
    def feature_correlation(exporter: Exporter = Exporter.in_memory()) -> Explorer:
        return FeatureCorrelationMatrix(exporter)

    @staticmethod
    def histogram(
        key: Optional[str] = None, bins: Optional[int] = None, exporter: Exporter = Exporter.in_memory()
    ) -> Explorer:
        return HistogramExplorer(key, bins, exporter)

    @staticmethod
    def scatter_plot(x_key: str, y_key: str, exporter: Exporter = Exporter.in_memory()) -> Explorer:
        return ScatterChartExplorer(x_key, y_key, exporter)

    @staticmethod
    def violin_plot(x_key: str, y_key: Optional[str] = None, exporter: Exporter = Exporter.in_memory()) -> Explorer:
        return ViolinPlotExplorer(x_key, y_key, exporter)


class FeatureCorrelationMatrix(Explorer):

    exporter: Exporter

    def __init__(self, exporter: Exporter) -> None:
        self.exporter = exporter

    async def explore(self, df: DataFrame) -> None:
        corr_heatmap = df.corr()
        corr_threshold = 0.9
        for feature in df.columns:
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


class HistogramExplorer(Explorer):

    key: Optional[str]
    bins: Optional[int]
    exporter: Exporter

    def __init__(self, key: Optional[str], bins: Optional[int], exporter: Explorer) -> None:
        self.key = key
        self.bins = bins
        self.exporter = exporter

    async def explore(self, df: DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = df.hist(self.key, bins=self.bins, ax=ax)
        await self.exporter.store_figure(fig)


class ScatterChartExplorer(Explorer):

    x_key: str
    y_key: str
    exporter: Exporter

    def __init__(self, x_key: str, y_key: str, exporter: Exporter) -> None:
        self.x_key = x_key
        self.y_key = y_key
        self.exporter = exporter

    async def explore(self, df: DataFrame) -> None:
        g = sns.jointplot(x=self.x_key, y=self.y_key, data=df)
        await self.exporter.store_figure(g)


class ViolinPlotExplorer(Explorer):

    x_key: str
    y_key: Optional[str]
    exporter: Exporter

    def __init__(self, x_key: str, y_key: Optional[str], exporter: Exporter) -> None:
        self.x_key = x_key
        self.y_key = y_key
        self.exporter = exporter

    async def explore(self, df: DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.violinplot(x=self.x_key, y=self.y_key, ax=ax)
        await self.exporter.store_figure(fig)
