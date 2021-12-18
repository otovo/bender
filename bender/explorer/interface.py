from pandas import DataFrame


class Explorer:
    async def explore(self, df: DataFrame) -> None:
        raise NotImplementedError()
