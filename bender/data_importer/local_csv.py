from typing import Optional

import pandas
from pandas import DataFrame

from bender.data_importer.interface import DataImporter


class LocalCsvImporter(DataImporter):

    file: str
    seperator: Optional[str]

    def __init__(self, file: str, seperator: Optional[str]) -> None:
        self.file = file
        self.seperator = seperator

    async def import_data(self) -> DataFrame:
        sep = self.seperator if self.seperator is not None else ','
        return pandas.read_csv(self.file, sep=sep)
