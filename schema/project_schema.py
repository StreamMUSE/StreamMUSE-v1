from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator
from schema.model_schema import ModelSchema
from schema.dataset_schema import DataModuleSchema
from typing import Union, Literal
import yaml
import os
import re


def get_next_version(base_dir: str, project_name: Optional[str] = None, version_prefix: Optional[str] = None) -> str:
    """
    Detects existing version directories (e.g., 'version_0', 'version_1')
    under base_dir/(project_name if exists) and returns the next available version string.
    If version_prefix is provided (e.g., "1.0"), it tries to find the max 'x' for "1.0.x"
    otherwise, it auto-increments "version_x".

    Args:
        base_dir: The base directory where logs/versions are saved.
        project_name: Optional, the name of the project, often used as a sub-directory.
        version_prefix: Optional, a version string like "1.0" to find "1.0.x".
                        If None, it looks for "version_x" directories.
    Returns:
        The next version string (e.g., "1.0.5" or "version_3").
    """
    target_dir = os.path.join(base_dir, project_name) if project_name else base_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists for scanning
        if version_prefix:
            return f"{version_prefix}.0"
        return "version_0"

    max_version_num = -1
    for item in os.listdir(target_dir):
        if os.path.isdir(os.path.join(target_dir, item)):
            if version_prefix:
                # Regex for "1.0.x" or similar
                match = re.match(rf"^{re.escape(version_prefix)}\.(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)
            else:
                # Regex for "version_x"
                match = re.match(r"^version_(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)

    next_version_num = max_version_num + 1
    if version_prefix:
        return f"{version_prefix}.{next_version_num}"
    return f"version_{next_version_num}"


class TrainerSchema(BaseModel):
    """
    Schema for the PyTorch Lightning Trainer configuration.
    """

    max_epochs: int = Field(10, description="Maximum number of epochs for training. Default is 10.")
    accelerator: Optional[str] = Field("auto", description="Accelerator to use for training (e.g., 'cpu', 'gpu'). Default is 'auto'.")
    devices: Optional[Union[int, list[int], tuple[int]]] = Field(
        None, description="Number of devices to use for training. Default is None (use all available)."
    )

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass


class CSVLoggerSchema(BaseModel):
    """
    Schema for the CSV logger configuration.
    """

    type: Literal["csv"] = Field("csv", description="Type of logger. Default is 'csv'.")
    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass

    @model_validator(mode="after")
    def auto_increment_version(self) -> "CSVLoggerSchema":
        if self.version is None:
            run_name_for_versioning = self.name if self.name else "default_run"
            self.version = get_next_version(self.save_dir, run_name_for_versioning)
        elif re.match(r"^\d+\.\d+$", self.version):  # If user provides "1.0" or "2.1"
            run_name_for_versioning = self.name if self.name else "default_run"
            self.version = get_next_version(self.save_dir, run_name_for_versioning, self.version)
        return self


class WandbLoggerSchema(BaseModel):
    """
    Schema for the Weights & Biases logger configuration.
    """

    type: Literal["wandb"] = Field("wandb", description="Type of logger. Default is 'wandb'.")
    project: str = Field(..., description="Name of the Weights & Biases project.")
    entity: Optional[str] = Field(None, description="Entity name for the Weights & Biases project. Optional if not needed.")
    log_model: bool = Field(True, description="Whether to log the model to Weights & Biases. Default is True.")

    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass

    @model_validator(mode="after")
    def auto_increment_version(self) -> "WandbLoggerSchema":
        # This validator runs after initial validation, so self.name, self.save_dir are ready
        if self.version is None:
            # Use self.name as the 'project_name' for versioning within save_dir
            run_name_for_versioning = self.name if self.name else "default_run"  # Use W&B project name if run name is None
            self.version = get_next_version(self.save_dir, run_name_for_versioning)
        elif re.match(r"^\d+\.\d+$", self.version):  # If user provides "1.0" or "2.1"
            run_name_for_versioning = self.name if self.name else "default_run"
            self.version = get_next_version(self.save_dir, run_name_for_versioning, self.version)
        return self


class TensorBoardLoggerSchema(BaseModel):
    """
    Schema for the TensorBoard logger configuration.
    """

    type: Literal["tensorboard"] = Field("tensorboard", description="Type of logger. Default is 'tensorboard'.")
    save_dir: str = Field("./logs", description="Directory where CSV logs will be saved.")
    name: Optional[str] = Field(None, description="Name of the CSV log file. Optional if not needed.")
    version: Optional[str] = Field(None, description="Version of the CSV log file. Optional if not needed.")

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass

    @model_validator(mode="after")
    def auto_increment_version(self) -> "TensorBoardLoggerSchema":
        if self.version is None:
            run_name_for_versioning = self.name if self.name else "default_run"
            self.version = get_next_version(self.save_dir, run_name_for_versioning)
        elif re.match(r"^\d+\.\d+$", self.version):  # If user provides "1.0" or "2.1"
            run_name_for_versioning = self.name if self.name else "default_run"
            self.version = get_next_version(self.save_dir, run_name_for_versioning, self.version)
        return self


LoggerSchema = Union[CSVLoggerSchema, WandbLoggerSchema, TensorBoardLoggerSchema]


class ProjectSchema(BaseModel):
    """
    Schema for the M2A Transformer project configuration.
    """

    project_name: str = Field(..., description="Name of the project.")
    version: str = Field("1.0.0", description="Version of the project. Default is '1.0.0'.")
    description: Optional[str] = Field(None, description="Description of the project.")
    loggers: Optional[dict[str, LoggerSchema]] = Field(None, description="List of logger configurations.")
    model: ModelSchema = Field(..., description="Model configuration.")
    dataset: DataModuleSchema = Field(..., description="Data module configuration.")
    trainer: TrainerSchema = Field(default_factory=TrainerSchema, description="Trainer configuration.")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility.")

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectSchema":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
