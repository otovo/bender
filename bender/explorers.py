from typing import Optional

from bender.explorer.correlation import CorrelationMatrix
from bender.explorer.histogram import Histogram, HistogramConfig
from bender.explorer.pair import PairPlot
from bender.explorer.scatter import ScatterChartExplorer
from bender.explorer.violin import ViolinPlot
from bender.exporter.exporter import Exporter, MemoryExporter


class Explorers:
    @staticmethod
    def histogram(
        features: Optional[list[str]] = None,
        target: Optional[str] = None,
        config: HistogramConfig = HistogramConfig(),
        exporter: Exporter = MemoryExporter(),
    ) -> Histogram:
        return Histogram(features, target, config, exporter)

    @staticmethod
    def correlation(features: Optional[list[str]] = None, exporter: Exporter = MemoryExporter()) -> CorrelationMatrix:
        return CorrelationMatrix(features, exporter)

    @staticmethod
    def pair_plot(target: str, features: Optional[list[str]] = None, exporter: Exporter = MemoryExporter()) -> PairPlot:
        return PairPlot(target, exporter, features)

    @staticmethod
    def distribution(feature: str, target: str, exporter: Exporter = MemoryExporter()) -> ViolinPlot:
        return ViolinPlot(target, feature, exporter)

    @staticmethod
    def scatter(
        x_feature: str, y_feature: str, target: Optional[str] = None, exporter: Exporter = MemoryExporter()
    ) -> ScatterChartExplorer:
        return ScatterChartExplorer(x_feature, y_feature, target, exporter)
