from __future__ import annotations

from pandas import DataFrame, Series, Timestamp, to_datetime

from bender.transformation.transformation import Transformation


class SchemaType:
    @property
    def data_type(self) -> type:
        raise NotImplementedError()

    def convert(self, series: Series) -> Series:
        raise NotImplementedError()

    @staticmethod
    def integer() -> IntegerType:
        return IntegerType()

    @staticmethod
    def float() -> FloatType:
        return FloatType()

    @staticmethod
    def string() -> StringType:
        return StringType()

    @staticmethod
    def datetime() -> DatetimeType:
        return DatetimeType()


class IntegerType(SchemaType):
    @property
    def data_type(self) -> type:
        return int

    def convert(self, series: Series) -> Series:
        return series.astype(int)


class StringType(SchemaType):
    @property
    def data_type(self) -> type:
        return str

    def convert(self, series: Series) -> Series:
        return series.astype(str)


class FloatType(SchemaType):
    @property
    def data_type(self) -> type:
        return float

    def convert(self, series: Series) -> Series:
        return series.astype(float)


class DatetimeType(SchemaType):
    @property
    def data_type(self) -> type:
        return type(Timestamp)

    def convert(self, series: Series) -> Series:
        if series.dtype == str:
            return to_datetime(series, infer_datetime_format=True)
        return to_datetime(series.astype(str), infer_datetime_format=True)


class SchemaTransformation(Transformation):  # type: ignore

    schemas: dict[str, SchemaType]

    def __init__(self, schemas: dict[str, SchemaType]) -> None:
        self.schemas = schemas

    async def transform(self, df: DataFrame) -> DataFrame:
        for (key, value) in self.schemas.items():
            if df[key].dtype == value.data_type:
                continue
            df[key] = value.convert(df[key])
        return df
