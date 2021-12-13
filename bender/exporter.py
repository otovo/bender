from typing import Union
from matplotlib.figure import Figure as PltFigure
from plotly.graph_objects import Figure as PlotFigure
from pandas.core.frame import DataFrame
import gspread

Figure = Union[PltFigure, PlotFigure]

class Exporter:
    async def store_figure(self, figure: Figure):
        raise NotImplementedError()

    async def store_data_frame(self, df: DataFrame):
        raise NotImplementedError()


class LocalDiskExporter(Exporter):

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    async def store_data_frame(self, df: DataFrame):
        used_path = self.path + ".csv"
        df.to_csv(used_path)

    async def store_figure(self, figure: Figure):
        used_path = self.path + ".png"
        if isinstance(figure, PltFigure):
            figure.savefig(used_path)
        elif isinstance(figure, PlotFigure):
            image_data = figure.to_image("png")
            with open(used_path, "wb") as file:
                file.write(image_data)
        else:
            raise NotImplementedError()


class GoogleDriveExporter(Exporter):

    credentials_path: str
    file_name: str

    async def store_data_frame(self, df: DataFrame):
        # Should store csv using the Docs API
        # https://developers.google.com/docs/api/reference/rest/v1/documents/create
        sheet = self._service().create(self.file_name)
        sheet.update([df.columns.values.tolist()] + df.values.tolist())

    async def store_figure(self, figure: Figure):
        raise NotImplementedError()


    def _service(self):
        return gspread.service_account()

