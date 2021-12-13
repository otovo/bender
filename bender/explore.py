from pandas import DataFrame

class Explorer:

    async def explor(self, df: DataFrame):
        raise NotImplementedError()


class SingleHistogram(Explorer):

    async def explor(self, df: DataFrame):
        pass