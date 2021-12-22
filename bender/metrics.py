from bender.metric.log_loss import LogLoss
from bender.metric.mean_square_error import MeanSquareError


class Metrics:
    @staticmethod
    def log_loss() -> LogLoss:
        return LogLoss()

    @staticmethod
    def mean_sqare_error() -> MeanSquareError:
        return MeanSquareError()

    @staticmethod
    def mean_absolute_error() -> MeanSquareError:
        return MeanSquareError()

    @staticmethod
    def mean_absolute_percentage_error() -> MeanSquareError:
        return MeanSquareError()
