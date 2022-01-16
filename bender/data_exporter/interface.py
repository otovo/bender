from __future__ import annotations

import asyncio

from pandas import DataFrame


class DataExporter:
    async def export(self, data_frame: DataFrame) -> None:
        raise NotImplementedError()

    def and_export(self, exporter: DataExporter) -> ChainedExporter:
        return ChainedExporter(self, exporter)


class ChainedExporter(DataExporter):

    first_exporter: DataExporter
    second_exporter: DataExporter

    def __init__(self, first: DataExporter, second: DataExporter) -> None:
        self.first_exporter = first
        self.second_exporter = second

    async def export(self, data_frame: DataFrame) -> None:
        await asyncio.gather(self.first_exporter.export(data_frame), self.second_exporter.export(data_frame))
