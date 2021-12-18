from typing import Optional

import seaborn as sns
from pandas import DataFrame

from bender.explorer.interface import Explorer
from bender.exporter.exporter import Exporter


class ScatterChartExplorer(Explorer):

    x_key: str
    y_key: str
    target: Optional[str]
    exporter: Exporter

    def __init__(self, x_key: str, y_key: str, target: Optional[str], exporter: Exporter) -> None:
        self.x_key = x_key
        self.y_key = y_key
        self.target = target
        self.exporter = exporter

    async def explore(self, df: DataFrame) -> None:
        g = sns.jointplot(x=self.x_key, y=self.y_key, data=df, hue=self.target)
        await self.exporter.store_figure(g.fig)
