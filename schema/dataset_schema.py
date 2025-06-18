from pydantic import BaseModel, Field
from typing import Optional, Any


class Pop909DatasetSchema(BaseModel):
    root_dir: str = Field(..., description="Root directory containing the POP909 dataset.")
    transform: Optional[Any] = Field(None, description="Optional transform to apply to MIDI data.")


class Pop909DataModuleSchema(BaseModel):
    batch_size: int = Field(32, description="Batch size for data loaders.")
    num_workers: int = Field(4, description="Number of workers for data loading.")
    train_dataset_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the training dataset.")
    val_dataset_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the validation dataset.")
    test_dataset_config: Optional[Pop909DatasetSchema] = Field(None, description="Configuration for the test dataset.")
    predict_dataset_config: Optional[Pop909DatasetSchema] = Field(None, description="Optional configuration for the prediction dataset.")
