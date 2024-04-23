from dataclasses import dataclass
from pathlib import Path

#Entity is return type of a function, it returns the configuration from data ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weight: str
    params_classes: int
