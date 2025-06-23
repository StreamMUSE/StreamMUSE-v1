from pydantic import BaseModel, Field
from typing import Optional, Any, Union
import miditok


# Abstract (parent) dataset schema
class BaseDatasetSchema(BaseModel):
    batch_size: int = Field(32, description="Batch size for data loaders.")
    num_workers: int = Field(4, description="Number of workers for data loading.")
    transform: Optional[Any] = Field(None, description="Optional transform to apply to data.")
    max_seq_len: Optional[int] = Field(384, description="Maximum sequence length for data. Default is 384.")
    tokenization_type: Optional[str] = Field("REMI", description="Type of tokenization to use. Default is 'REMI'.")
    data_range: Optional[Union[list[float], tuple[float]]] = Field(
        (0.0, 1.0), description="Optional range for data normalization. Default is None (no normalization)."
    )
    stage: Optional[str] = Field("train", description="Stage of the dataset (e.g., 'train', 'val', 'test'). Optional if not needed.")


# Abstract (parent) datamodule schema
class BaseDataModuleSchema(BaseModel):

    tokenizer: Optional[Any] = Field(miditok.REMI, description="Tokenizer for processing MIDI data. Optional if not needed.")
    train_config: Optional[BaseDatasetSchema] = Field(None, description="Configuration for the training dataset.")
    val_config: Optional[BaseDatasetSchema] = Field(None, description="Configuration for the validation dataset.")
    test_config: Optional[BaseDatasetSchema] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[BaseDatasetSchema] = Field(None, description="Optional configuration for the prediction dataset.")

    def model_post_init(self, context: Any) -> None:
        if self.train_config:
            self.train_config.stage = "train"
        if self.val_config:
            self.val_config.stage = "val"
        if self.test_config:
            self.test_config.stage = "test"
        if self.predict_config:
            self.predict_config.stage = "predict"


class MelAccRemiJsonDatasetSchema(BaseDatasetSchema):
    mel_dir: str = Field(..., description="Directory containing melody JSONs.")
    acc_dir: str = Field(..., description="Directory containing accompaniment JSONs.")
    file_pattern: str = Field("*.json", description="Glob pattern for finding JSON files (e.g., '*.json').")

class MelAccRemiJsonDataModuleSchema(BaseDataModuleSchema):
    train_config: MelAccRemiJsonDatasetSchema = Field(..., description="Configuration for the training dataset.")
    val_config: MelAccRemiJsonDatasetSchema = Field(..., description="Configuration for the validation dataset.")
    test_config: Optional[MelAccRemiJsonDatasetSchema] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[MelAccRemiJsonDatasetSchema] = Field(None, description="Configuration for the prediction dataset.")
    batch_size: int = Field(1, description="Batch size for data loaders.")
    num_workers: int = Field(0, description="Number of workers for data loaders.")

DataModuleSchema = Union[MelAccRemiJsonDataModuleSchema]
