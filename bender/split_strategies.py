from bender.split_strategy.split_strategy import RandomRatioSplitter, SortedRatioSplitter


class SplitStrategies:
    @staticmethod
    def ratio(ratio: float) -> RandomRatioSplitter:
        return RandomRatioSplitter(min(max(ratio, 0), 1))

    @staticmethod
    def sorted_ratio(sort_key: str, ratio: float) -> SortedRatioSplitter:
        return SortedRatioSplitter(min(max(ratio, 0), 1), sort_key)
