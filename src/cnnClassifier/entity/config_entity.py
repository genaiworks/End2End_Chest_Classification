from dataclasses import dataclass
from pathlib import Path

#Entity is return type of a function, it returns the configuration from data ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str
    local_data_file: Path
    unzip_dir: Path