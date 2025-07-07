from pydantic import BaseModel, Field
from typing import Optional, Any, Union, Iterator
import torch
import numpy as np
from collections.abc import Mapping


class BaseModelInputData(BaseModel):
    """
    Schema for model input.
    """

    mel_data: Optional[Union[torch.Tensor, np.ndarray]] = Field(..., description="Melody data in REMI format.  to the tensor conversion.")
    acc_data: Optional[Union[torch.Tensor, np.ndarray]] = Field(..., description="Accompaniment data in REMI format. to the tensor conversion.")

    model_config = {"arbitrary_types_allowed": True}


class BaseModelOutputData(Mapping, BaseModel):
    """
    Schema for model output.
    """

    logits: Optional[torch.Tensor] = Field(None, description="Logits from the model output.")
    loss: Optional[torch.Tensor] = Field(None, description="Loss value computed from the model output.")
    predicted_ids: Optional[torch.Tensor] = Field(None, description="Predicted IDs from the model output.")
    metadata: Optional[Any] = Field(None, description="Optional metadata associated with the model output.")

    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, key: str) -> Any:
        if key in self.model_fields.keys():
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in model fields.")

    def __len__(self) -> int:
        return len(self.model_fields)

    def __iter__(self) -> Iterator[str]:
        yield from self.model_fields.keys()


class M2AModelInputData(BaseModelInputData):
    """
    Schema for M2A Transformer model input.
    """

    pitch_shift: Optional[torch.Tensor] = Field(None, description="Pitch shift values for the melody and accompaniment data.")


class M2AModelOutputData(BaseModelOutputData):
    """
    Schema for M2A Transformer model output.
    """


ModelOutputData = Union[M2AModelOutputData]

ModelInputData = Union[M2AModelInputData]
