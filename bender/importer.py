from __future__ import annotations
import datetime
import logging
from typing import Optional, TypeVar
from databases.core import DatabaseURL
from pandas import DataFrame
import pandas
from databases import Database
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataImporter:
    async def import_data(self) -> DataFrame:
        raise NotImplementedError()

    def join_import(self, importer: DataImporter, join_key: str) -> DataImporter:
        return JoinedImporter(first_import=self, second_import=importer, join_key=join_key)

    def cached(self, path: str, from_now: Optional[timedelta] = None, timestamp: Optional[datetime] = None) -> DataImporter:
        if timestamp:
            return CachedImporter(self, path, timestamp)
        elif from_now:
            return CachedImporter(self, path, datetime.now() + from_now)
        else:
            return CachedImporter(self, path, datetime.now() + timedelta(days=1))


DataImporterType = TypeVar('DataImporterType')

class DataImportable:
    def import_data(self, importer: DataImporter) -> DataImporterType:
        raise NotImplementedError()


class CachedImporter(DataImporter):

    importer: DataImporter
    path: str
    expiration_date: datetime

    def __init__(self, importer: DataImporter, path: str, expiration_date: datetime) -> None:
        self.importer = importer
        self.path = path
        self.expiration_date = expiration_date

    async def import_data(self) -> DataFrame:
        expration_path = self.path + "expiration.csv"
        file_path = self.path + ".csv"
        try:
            logger.info("Trying to load csv")
            saved_expiration_date = pandas.read_csv(expration_path)
            if pandas.to_datetime(saved_expiration_date["date"].iloc[0]) < datetime.now():
                logger.info("Refreshing source")
                raise Exception("Out of date cache")
            return pandas.read_csv(file_path)
        except Exception as error:
            logger.info(f"Error loading file, so loading from source: {error}")
            expiration = DataFrame({"date": [self.expiration_date]})
            df = await self.importer.import_data()
            df.to_csv(file_path)
            expiration.to_csv(expration_path)
            return df


class JoinedImporter(DataImporter):

    first_import: DataImporter
    second_import: DataImporter
    join_key: str

    def __init__(self, first_import: DataImporter, second_import: DataImporter, join_key: str) -> None:
        self.first_import = first_import
        self.second_import = second_import
        self.join_key = join_key

    async def import_data(self) -> DataFrame:
        first_frame = await self.first_import.import_data()
        second_frame = await self.second_import.import_data()
        return first_frame.join(second_frame, on=self.join_key, how="inner")

        
class LiteralImporter(DataImporter):

    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    async def import_data(self) -> DataFrame:
        return self.df


class SqlImporter(DataImporter):

    query: str
    values: Optional[dict]
    url: DatabaseURL

    def __init__(self, url: DatabaseURL, query: str, values: Optional[dict]) -> None:
        self.query = query
        self.url = url
        self.values = values

    async def import_data(self) -> DataFrame:
        database = Database(self.url)
        await database.connect()
        records = await database.fetch_all(self.query, values=self.values)
        await database.disconnect()
        return DataFrame.from_records(records)
