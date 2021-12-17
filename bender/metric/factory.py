from typing import Generic, TypeVar

from bender.metric.interface import Metric

T = TypeVar('T')


class Metricable(Generic[T]):
    def metric(self, metric: Metric) -> T:
        raise NotImplementedError()
