from pandas import DataFrame

from bender.data_importer.interface import DataImporter
from bender.transformation.transformation import Transformation


class DataExplorer:

    data: DataFrame
    importer: DataImporter

    def __init__(self, importer: DataImporter) -> None:
        self.importer = importer
        self.data = DataFrame()

    def add(self, transformation: Transformation) -> None:
        raise NotImplementedError()

    async def import_with(self, importer: DataImporter) -> None:
        raise NotImplementedError()
