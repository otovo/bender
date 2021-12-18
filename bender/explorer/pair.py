from typing import Optional

import seaborn as sns
from pandas import DataFrame

from bender.explorer.interface import Explorer
from bender.exporter.exporter import Exporter


class PairPlot(Explorer):

    features: Optional[list[str]]
    target: str
    exporter: Exporter

    def __init__(self, target: str, exporter: Exporter, features: Optional[list[str]]) -> None:
        self.exporter = exporter
        self.target = target
        self.features = features

    async def explore(self, df: DataFrame) -> None:
        plot = sns.pairplot(df, hue=self.target, vars=self.features)
        await self.exporter.store_figure(plot.fig)
