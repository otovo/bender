from typing import Generic, TypeVar

from bender.explorer.interface import Explorer

T = TypeVar('T')


class Explorable(Generic[T]):
    def explore(self, explorers: list[Explorer]) -> T:
        raise NotImplementedError()
