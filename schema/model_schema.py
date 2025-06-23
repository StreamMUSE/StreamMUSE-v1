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
    network_schema: Optional[RoFormerConfig] = Field(None, description="Configuration for the RoFormer model.")
    optimizer_schema: OptimizerSchema = Field(default_factory=lambda: OptimizerSchema(), description="Configuration for the optimizer.")

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, context: Any) -> None:
        # Additional initialization logic can be added here if needed
        pass


class M2AModelSchema(BaseModelSchema):
    """
    Schema for the M2A Transformer model configuration.
    """

    model_name: str = Field("M2A Transformer", description="Name of the M2A Transformer model.")
    model_type: Literal["M2A-Transformer"] = Field("M2A-Transformer", description="Type of the model.")
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


ModelSchema = Union[M2AModelSchema]
