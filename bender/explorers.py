from typing import Optional

from bender.explorer.correlation import CorrelationMatrix
from bender.explorer.histogram import Histogram
from bender.explorer.pair import PairPlot
from bender.explorer.violin import ViolinPlot
from bender.exporter.exporter import Exporter, MemoryExporter


class Explorers:
    @staticmethod
    def histogram(
        features: Optional[list[str]] = None, target: Optional[str] = None, exporter: Exporter = MemoryExporter()
    ) -> Histogram:
        return Histogram(features, target, exporter)

    @staticmethod
    def correlation(exporter: Exporter = MemoryExporter()) -> CorrelationMatrix:
        return CorrelationMatrix(exporter)

    @staticmethod
    def pair_plot(target: str, features: Optional[list[str]] = None, exporter: Exporter = MemoryExporter()) -> PairPlot:
        return PairPlot(target, exporter, features)

    @staticmethod
    def distribution(feature: str, target: str, exporter: Exporter = MemoryExporter()) -> ViolinPlot:
        return ViolinPlot(target, feature, exporter)
