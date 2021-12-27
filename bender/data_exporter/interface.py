from pandas import DataFrame


class DataExporter:
    async def export(self, data_frame: DataFrame) -> None:
        raise NotImplementedError()
