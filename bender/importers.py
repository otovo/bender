from typing import Any, Optional

from pandas import DataFrame

from bender.data_importer.importer import LiteralImporter, SqlImporter
from bender.pipeline.factory_states import LoadedData


class DataImporters:
    @staticmethod
    def sql(url: str, query: str, values: Optional[dict[str, Any]] = None) -> LoadedData:
        return LoadedData(SqlImporter(url, query, values=values), [])

    @staticmethod
    def literal(df: DataFrame) -> LoadedData:
        return LoadedData(LiteralImporter(df), [])
