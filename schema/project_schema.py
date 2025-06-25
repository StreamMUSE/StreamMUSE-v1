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



LoggerSchema = Union[CSVLoggerSchema, WandbLoggerSchema, TensorBoardLoggerSchema]


class ProjectSchema(BaseModel):
    """
    Schema for the M2A Transformer project configuration.
    """

    project: str = Field(..., description="Name of the project.")
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

    @model_validator(mode="after")
    def unify_logger_versions_and_names(self) -> "ProjectSchema":
        if not self.loggers:
            return self

        # 统一所有 logger 的 save_dir (可选，但推荐)
        # 这里我们假设所有 logger 都应保存在同一个基础目录下
        base_save_dir = next(iter(self.loggers.values())).save_dir

        # 使用 project 的 name 和 version 作为版本控制的基准
        # project.name 是实验名, project.version 是版本前缀
        unified_version = get_next_version(base_dir=base_save_dir, project_name=self.name, version_prefix=self.version)

        # 应用统一的 name 和 version 到所有 logger
        for logger_config in self.loggers.values():
            # 如果 logger 没有指定 name，则默认使用 project.name
            if logger_config.name is None:
                logger_config.name = self.name

            # 强制所有 logger 使用统一计算出的版本号
            logger_config.version = unified_version

        return self