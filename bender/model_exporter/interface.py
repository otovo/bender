from __future__ import annotations

from bender.trained_model.interface import TrainedModel


class ModelExporter:
    async def export(self, model: TrainedModel) -> None:
        raise NotImplementedError()
