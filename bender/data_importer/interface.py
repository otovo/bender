from pandas import DataFrame


class DataImporter:
    async def import_data(self) -> DataFrame:
        raise NotImplementedError()
