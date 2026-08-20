from pathlib import Path

from fire import Fire

from brain_segmentation_tools.model_manager import ModelManager


class ModelConversionCLI:
    def __init__(self, dev_mode: bool | None = None):
        self.manager = ModelManager(dev_mode=dev_mode)

    def list_models(self):
        return [spec.key for spec in self.manager.configured_models]

    def convert_h5_to_pt(self, *, model_name: str, model_type: str, version: str, output_path: str):
        out_path = self.manager.convert_h5_to_pt(
            model_name=model_name,
            model_type=model_type,
            version=version,
            output_path=output_path,
        )
        return out_path.as_posix()

    def save_all_converted(self, *, output_dir: str):
        converted = self.manager.save_all_converted(Path(output_dir))
        return {k: v.as_posix() for k, v in converted.items()}


def main():
    Fire(ModelConversionCLI)


if __name__ == "__main__":
    main()
