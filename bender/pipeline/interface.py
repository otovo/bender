from typing import Generic, TypeVar

Output = TypeVar('Output')


class RunnablePipeline(Generic[Output]):
    async def run(self) -> Output:
        raise NotImplementedError()
