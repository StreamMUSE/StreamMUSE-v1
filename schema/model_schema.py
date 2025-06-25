from typing import Any
from pydantic import BaseModel, Field
from roformer import RoFormerConfig
from typing import Union, Optional
from typing import Literal


class OptimizerSchema(BaseModel):
    optimizer_type: Literal["adam", "sgd", "adamw"] = Field("adam", description="Type of optimizer.")
    learning_rate: float = Field(1e-4, description="Learning rate for the optimizer.")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    momentum: Optional[float] = Field(None, description="Momentum factor (for SGD).")


class BaseModelSchema(BaseModel):
    """
    Schema for the M2A Transformer model configuration.
    """

    model_name: str = Field("", description="Name of the model.")
    model_type: str = Field("roformer", description="Type of the model. Default is 'roformer'.")
    network_schema: Optional[Any] = Field(None, description="Configuration for the RoFormer model.")
    optimizer_schema: OptimizerSchema = Field(default_factory=lambda: OptimizerSchema(), description="Configuration for the optimizer.")

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass


class M2AModelSchema(BaseModelSchema):
    """
    Schema for the M2A Transformer model configuration.
    """

    model_name: str = Field("REMI-RoFormer", description="Name of the M2A Transformer model.")
    model_type: Literal["REMI-RoFormer"] = Field("REMI-RoFormer", description="Type of the model.")
    network_schema: RoFormerConfig = Field(
        default_factory=lambda: RoFormerConfig(
            vocab_size=500,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            hidden_size=768,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            type_vocab_size=2,
            initializer_range=0.02,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            is_decoder=True,
            return_dict=True,
        ),
        description="Configuration for the RoFormer model. Default values are set for a typical transformer architecture.",
    )

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass


class OldM2ATransformerSchema(BaseModelSchema):
    """
    Configuration schema for RoFormerSymbolicTransformer hyperparameters.
    """
    model_name: str = Field("Old-M2A-Transformer", description="Name of the M2A Transformer model.")
    model_type: Literal["Old-M2A-Transformer"] = Field("Old-M2A-Transformer", description="Type of the model.")
    large: bool = Field(False, description="If True, use larger model configuration (hidden_size=768, num_layers=12, etc.).")

    # Global Model (Main Transformer) Hyperparameters
    hidden_size: Optional[int] = Field(None, description="Hidden layer dimension of the global model. Overrides 'large' setting if specified.")
    num_layers: Optional[int] = Field(
        None, description="Number of Transformer encoder layers in the global model. Overrides 'large' setting if specified."
    )
    num_attention_heads: Optional[int] = Field(
        None, description="Number of attention heads in the global model. Overrides 'large' setting if specified."
    )
    intermediate_size: Optional[int] = Field(
        None, description="Intermediate size of the feed-forward network in the global model. Overrides 'large' setting if specified."
    )

    # Local Model (Encoder/Decoder) Hyperparameters
    local_model_num_layers: int = Field(3, description="Number of layers in the local encoder/decoder.")
    local_model_num_attention_heads: int = Field(8, description="Number of attention heads in the local encoder/decoder.")
    local_model_intermediate_size: int = Field(768, description="Intermediate size of the feed-forward network in the local encoder/decoder.")

    # Dropout probabilities
    hidden_dropout_prob: float = Field(0.1, description="Dropout probability for hidden layers.")
    attention_probs_dropout_prob: float = Field(0.1, description="Dropout probability for attention weights.")

    def model_post_init(self, __context: Any) -> None:
        if self.large:
            # Set default large model parameters IF they haven't been explicitly set
            if self.hidden_size is None:
                self.hidden_size = 768
            if self.num_layers is None:
                self.num_layers = 12
            if self.num_attention_heads is None:
                self.num_attention_heads = 12
            if self.intermediate_size is None:
                self.intermediate_size = 3072
        else:
            # Set default small model parameters IF they haven't been explicitly set
            if self.hidden_size is None:
                self.hidden_size = 512
            if self.num_layers is None:
                self.num_layers = 6
            if self.num_attention_heads is None:
                self.num_attention_heads = 8
            if self.intermediate_size is None:
                self.intermediate_size = 1024


ModelSchema = Union[M2AModelSchema, OldM2ATransformerSchema]
# ModelSchema = OldM2ATransformerSchema