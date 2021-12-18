import logging
from enum import Enum
from typing import Any, Optional

from pandas import DataFrame

from bender.data_importer.importer import LiteralImporter, SqlImporter
from bender.pipeline.factory_states import LoadedData  # type: ignore


class DataSets(Enum):
    IRIS = 'iris'
    DIABETES = 'diabets'
    BREAST_CANCER = 'breast_cancer'
    DIGITS = 'digits'
    WINE = 'wine'


logger = logging.getLogger(__name__)


class DataImporters:
    def __combine(data) -> DataFrame:
        df = data.data  # type: ignore
        df['target'] = data.target  # type: ignore
        logger.info(f'Loaded data set with target names: {data.target_names}')  # type: ignore
        return df

    @staticmethod
    def sql(url: str, query: str, values: Optional[dict[str, Any]] = None) -> LoadedData:
        return LoadedData(SqlImporter(url, query, values=values), [])

    @staticmethod
    def literal(df: DataFrame) -> LoadedData:
        return LoadedData(LiteralImporter(df), [])

    @staticmethod
    def data_set(data_set: DataSets) -> LoadedData:
        df: DataFrame
        if data_set == DataSets.IRIS:
            from sklearn.datasets import load_iris

            df = DataImporters.__combine(load_iris(as_frame=True))
        elif data_set == DataSets.BREAST_CANCER:
            from sklearn.datasets import load_breast_cancer

            df = DataImporters.__combine(load_breast_cancer(as_frame=True))
        elif data_set == DataSets.DIABETES:
            from sklearn.datasets import load_diabetes

            df = DataImporters.__combine(load_diabetes(as_frame=True))
        elif data_set == DataSets.DIGITS:
            from sklearn.datasets import load_digits

            df = DataImporters.__combine(load_digits(as_frame=True))
        elif data_set == DataSets.WINE:
            from sklearn.datasets import load_wine

            df = DataImporters.__combine(load_wine(as_frame=True))
        else:
            raise Exception('Unable to find the correct data set')
        return DataImporters.literal(df)
