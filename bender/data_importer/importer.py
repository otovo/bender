from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

import pandas
from databases import Database
from databases.core import DatabaseURL
from pandas import DataFrame

from bender.data_importer.interface import DataImporter

logger = logging.getLogger(__name__)


DataImporterType = TypeVar('DataImporterType')


class DataImportable(Generic[DataImporterType]):
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
        expration_path = self.path + 'expiration.csv'
        file_path = self.path + '.csv'
        try:
            logger.info('Trying to load csv')
            saved_expiration_date = pandas.read_csv(expration_path)
            if pandas.to_datetime(saved_expiration_date['date'].iloc[0]) < datetime.now():
                logger.info('Refreshing source')
                raise Exception('Out of date cache')
            return pandas.read_csv(file_path)
        except Exception as error:
            logger.info(f'Error loading file, so loading from source: {error}')
            expiration = DataFrame({'date': [self.expiration_date]})
            df = await self.importer.import_data()
            df.to_csv(file_path, index=False)
            expiration.to_csv(expration_path)
            return df


class AppendImporter(DataImporter):

    first_importer: DataImporter
    second_importer: DataImporter

    def __init__(self, first_importer: DataImporter, second_importer: DataImporter) -> None:
        self.first_importer = first_importer
        self.second_importer = second_importer

    async def import_data(self) -> DataFrame:
        first, second = await asyncio.gather(self.first_importer.import_data(), self.second_importer.import_data())
        return first.append(second)


class JoinedImporter(DataImporter):

    first_import: DataImporter
    second_import: DataImporter
    join_key: str

    def __init__(self, first_import: DataImporter, second_import: DataImporter, join_key: str) -> None:
        self.first_import = first_import
        self.second_import = second_import
        self.join_key = join_key

    async def import_data(self) -> DataFrame:
        first_frame, second_frame = await asyncio.gather(
            self.first_import.import_data(), self.second_import.import_data()
        )
        return first_frame.join(second_frame, on=self.join_key, how='inner')


class LiteralImporter(DataImporter):

    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    async def import_data(self) -> DataFrame:
        return self.df


class SqlImporter(DataImporter):

    query: str
    values: Optional[dict[str, Any]]
    url: DatabaseURL

    def __init__(self, url: DatabaseURL, query: str, values: Optional[dict[str, Any]]) -> None:
        self.query = query
        self.url = url
        self.values = values

    async def import_data(self) -> DataFrame:
        database = Database(self.url)
        await database.connect()
        records = await database.fetch_all(self.query, values=self.values)
        await database.disconnect()
        return DataFrame.from_records(records)
