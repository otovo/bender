from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Exporter:
    async def store_figure(self, figure: Figure) -> None:
        raise NotImplementedError()

    # @staticmethod
    # def clearml(logger=Logger) -> ClearmlExporter:
    #     return ClearmlExporter(logger)

    @staticmethod
    def disk(path: str) -> LocalDiskExporter:
        return LocalDiskExporter(path)

    @staticmethod
    def in_memory() -> MemoryExporter:
        return MemoryExporter()


class ChainedExporter(Exporter):

    first: Exporter
    second: Exporter

    def __init__(self, first: Exporter, second: Exporter) -> None:
        self.first = first
        self.second = second

    async def store_figure(self, figure: Figure) -> None:
        await self.first.store_figure(figure)
        await self.second.store_figure(figure)


class LocalDiskExporter(Exporter):

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    async def store_figure(self, figure: Figure) -> None:
        used_path = self.path + '.png'
        figure.savefig(used_path)


class MemoryExporter(Exporter):
    async def store_figure(self, figure: Figure) -> None:
        plt.show()


# class ClearmlExporter(Exporter):

#     logger: Logger

#     def __init__(self, logger: Logger) -> None:
#         self.logger = logger

#     async def store_data_frame(self, df: DataFrame):
#         raise NotImplementedError()

#     async def store_figure(self, figure: Figure):
#         if isinstance(figure, PltFigure):
#             plt.show(block=False)
#             plt.close()
#         elif isinstance(figure, PlotFigure):
#             self.logger.report_plotly(title="plotly plot", series="plotly", figure=figure)
#         else:
#             raise NotImplementedError()
