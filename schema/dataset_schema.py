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


# Now inherit for your specific dataset/datamodule schemas
class Pop909DatasetSchema(BaseDatasetSchema):
    mel_dir: str = Field(..., description="Directory containing melody REMI json files.")
    acc_dir: str = Field(..., description="Directory containing accompaniment REMI json files.")


class Pop909DataModuleSchema(BaseDataModuleSchema):
    train_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the training dataset.")
    val_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the validation dataset.")
    test_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the test dataset.")
    predict_config: Optional[Pop909DatasetSchema] = Field(None, description="Optional configuration for the prediction dataset.")

DataModuleSchema = Union[Pop909DataModuleSchema]
