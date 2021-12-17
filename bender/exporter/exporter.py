from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as PltFigure
from plotly.graph_objects import Figure as PlotFigure

Figure = Union[PltFigure, PlotFigure]


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
        if isinstance(figure, PltFigure):
            figure.savefig(used_path)
        elif isinstance(figure, PlotFigure):
            image_data = figure.to_image('png')
            with open(used_path, 'wb') as file:
                file.write(image_data)
        else:
            raise NotImplementedError()


class MemoryExporter(Exporter):
    async def store_figure(self, figure: Figure) -> None:
        if isinstance(figure, (PlotFigure, PltFigure)):
            plt.show()
        else:
            raise NotImplementedError()


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
