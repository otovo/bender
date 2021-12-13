from typing import Optional
from pandas import DataFrame
from bender.factory_states import LoadedData
from bender.importer import SqlImporter, LiteralImporter


class DataImporters:

    @staticmethod
    def sql(url: str, query: str, values: Optional[dict] = None) -> LoadedData:
        return LoadedData(SqlImporter(url, query, values=values))

    @staticmethod
    def literal(df: DataFrame) -> LoadedData:
        return LoadedData(LiteralImporter(df))