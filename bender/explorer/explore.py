from pandas import DataFrame


class Explorer:
    async def explor(self, df: DataFrame) -> None:
        raise NotImplementedError()


class SingleHistogram(Explorer):
    async def explor(self, df: DataFrame) -> None:
        pass
