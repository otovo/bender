import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from bender.explorer.interface import Explorer
from bender.exporter.exporter import Exporter


class ViolinPlot(Explorer):

    target: str
    y_feature: str
    exporter: Exporter

    def __init__(self, target: str, y_feature: str, exporter: Exporter) -> None:
        self.exporter = exporter
        self.y_feature = y_feature
        self.target = target

    async def explore(self, df: DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=df,
            x=self.target,
            y=self.y_feature,
            ax=ax,
        )
        await self.exporter.store_figure(fig)
