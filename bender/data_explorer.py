from pandas import DataFrame
from bender.transformation import Transformation
from bender.importer import DataImporter

class DataExplorer:

    data: DataFrame
    importer: DataImporter

    def __init__(self, importer: DataImporter) -> None:
        self.importer = importer
        self.data = DataFrame()

    def add(self, transformation: Transformation):
        raise NotImplementedError()

    async def import_with(self, importer: DataImporter):
        raise NotImplementedError()