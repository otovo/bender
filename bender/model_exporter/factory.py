from typing import Generic, TypeVar

from bender.model_exporter.interface import ModelExporter

ExportedType = TypeVar('ExportedType')


class ModelExportable(Generic[ExportedType]):
    def export_model(self, exporter: ModelExporter) -> ExportedType:
        raise NotImplementedError()
