from pydantic import BaseModel, Field
from typing import Optional, Any, Union
import torch
import numpy as np


class ModelInputData(BaseModel):
    """
    Schema for model input.
    """

    mel_data: Optional[Union[torch.Tensor,np.ndarray]] = Field(..., description="Melody data in REMI format.  to the tensor conversion.")
    acc_data: Optional[Union[torch.Tensor,np.ndarray]] = Field(..., description="Accompaniment data in REMI format. to the tensor conversion.")

    model_config = {"arbitrary_types_allowed": True}


class ModelOutputData(BaseModel):
    """
    Schema for model output.
    """
    logits: Optional[torch.Tensor] = Field(None, description="Logits from the model output.")
    loss: Optional[torch.Tensor] = Field(None, description="Loss value computed from the model output.")
    predicted_ids: Optional[torch.Tensor] = Field(None, description="Predicted IDs from the model output.")
    metadata: Optional[Any] = Field(None, description="Optional metadata associated with the model output.")

    model_config = {"arbitrary_types_allowed": True}