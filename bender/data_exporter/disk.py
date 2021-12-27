from pandas import DataFrame

from bender.data_exporter.interface import DataExporter


class DiskDataExporter(DataExporter):

    path: str

    def __init__(self, path: str) -> None:
        self.path = path

    async def export(self, data_frame: DataFrame) -> None:
        data_frame.to_csv(self.path, index=False)
