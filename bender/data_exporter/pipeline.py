from typing import Generic, TypeVar

from bender.data_exporter.interface import DataExporter

T = TypeVar('T')


class Extractable(Generic[T]):
    def extract(self, prediction_as: str, metadata: list[str], exporter: DataExporter) -> T:
        raise NotImplementedError()
