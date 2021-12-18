from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ceil, floor, sqrt
from pandas import DataFrame

from bender.explorer.interface import Explorer
from bender.exporter.exporter import Exporter


class Histogram(Explorer):

    features: Optional[list[str]]
    target: Optional[str]
    exporter: Exporter

    def __init__(self, features: Optional[list[str]], target: Optional[str], exporter: Exporter) -> None:
        self.features = features
        self.exporter = exporter
        self.target = target

    async def explore(self, df: DataFrame) -> None:
        features = list(df.columns)
        if self.features:
            features = self.features
        if self.target:
            x_axs = int(ceil(sqrt(len(features))))
            y_axs = int(ceil(len(features) / x_axs))
            fig, axs = plt.subplots(x_axs, y_axs, figsize=(7, 5))
            for index, feature in enumerate(features):
                sns.histplot(
                    df, x=feature, hue=self.target, multiple='stack', ax=axs[index % x_axs, int(floor(index / x_axs))]
                )
            await self.exporter.store_figure(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            df.hist(features, ax=ax)
            await self.exporter.store_figure(fig)
