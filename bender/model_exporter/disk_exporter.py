from bender.model_exporter.interface import ModelExporter
from bender.trained_model.interface import TrainedModel


class DiskExporter(ModelExporter):

    file_url: str

    def __init__(self, file_url: str) -> None:
        self.file_url = file_url

    async def export(self, model: TrainedModel) -> None:
        json_data = model.to_json()
        with open(self.file_url, 'w') as file:
            file.write(json_data)
