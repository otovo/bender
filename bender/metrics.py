from bender.metric.log_loss import LogLoss


class Metrics:
    @staticmethod
    def log_loss() -> LogLoss:
        return LogLoss()
