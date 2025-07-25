from typing import Any
from pydantic import BaseModel, Field
from roformer import RoFormerConfig
from typing import Union, Optional
from typing import Literal


class TrainingProbingLoggerSchema(BaseModel):
    """TrainingProbingLoggerSchema."""

    loss_jump_threshold_X: float = Field(1.5, description="Threshold for loss jump")
    loss_avg_window_Y: int = Field(10, description="Window size for loss average")
    recording_window_N: int = Field(5, description="Window size for recording")


class OptimizerSchema(BaseModel):
    optimizer_type: Literal["adam", "sgd", "adamw"] = Field("adam", description="Type of optimizer.")
    params: dict[str, Any] = Field(
        default_factory=lambda: {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8}, description="Parameters for the optimizer."
    )
    # learning_rate: float = Field(1e-4, description="Learning rate for the optimizer.")
    # weight_decay: float = Field(0.0, description="Weight decay (L2 penalty).")
    # momentum: Optional[float] = Field(None, description="Momentum factor (for SGD).")


class LRSchedulerSchema(BaseModel):
    """Schema for learning rate scheduler configuration."""

    type: str = Field("CosineAnnealingLR", description="Type of the LR scheduler, e.g., 'CosineAnnealingLR', 'StepLR'.")
    params: dict[str, Any] = Field(default_factory=lambda: {"T_max": 10000, "eta_min": 1e-6}, description="Parameters for the LR scheduler.")


class BaseModelSchema(BaseModel):
    """
    Schema for the M2A Transformer model configuration.
    """

    model_name: str = Field("", description="Name of the model.")
    model_type: str = Field("roformer", description="Type of the model. Default is 'roformer'.")
    network_schema: Optional[Union[Any, dict[str, Any]]] = Field(None, description="Configuration for the RoFormer model.")
    optimizer_schema: OptimizerSchema = Field(default_factory=lambda: OptimizerSchema(), description="Configuration for the optimizer.")
    training_probing_training_logger_schema: TrainingProbingLoggerSchema = Field(
        default_factory=lambda: TrainingProbingLoggerSchema(), description="Configuration for the training probing logger."
    )

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

    ckpt_path: Optional[str] = Field(
        None, description="Path to the checkpoint file. If provided, the model will be initialized from this checkpoint."
    )

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

class OldM2ANomaskTransformerSchema(BaseModelSchema):
    """
    Configuration schema for RoFormerSymbolicTransformer hyperparameters.
    """

    model_name: str = Field("XinYue's + nomask + customed RoFormer", description="Name of the M2A Transformer model.")
    model_type: Literal["Old-M2A-Transformer-Nomask"] = Field("Old-M2A-Transformer-Nomask", description="Type of the model.")
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

    ckpt_path: Optional[str] = Field(
        None, description="Path to the checkpoint file. If provided, the model will be initialized from this checkpoint."
    )
    
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
                
class NewM2ATransformerSchema(OldM2ATransformerSchema):
    model_name: str = Field("XinYue's + customed RoFormer", description="Name of the M2A Transformer model.")
    model_type: Literal["New-M2A-Transformer"] = Field("New-M2A-Transformer", description="Type of the model.")
    frame_shift: int = Field(4, description="Number of frame shift.")
    

ModelSchema = Union[M2AModelSchema, OldM2ATransformerSchema, OldM2ANomaskTransformerSchema,NewM2ATransformerSchema]
# ModelSchema = OldM2ATransformerSchema


# class OldM2ATransformerNetworkSchema(BaseModel):
#     """Network-specific hyperparameters for Old-M2A-Transformer."""

#     hidden_size: int = Field(1536, description="Hidden layer dimension of the global model.")
#     num_layers: int = Field(12, description="Number of Transformer encoder layers in the global model.")
#     num_attention_heads: int = Field(16, description="Number of attention heads in the global model.")
#     intermediate_size: int = Field(6144, description="Intermediate size of the feed-forward network.")
#     local_model_num_layers: int = Field(3, description="Number of layers in the local encoder/decoder.")
#     local_model_num_attention_heads: int = Field(16, description="Number of attention heads in the local encoder/decoder.")
#     local_model_intermediate_size: int = Field(3072, description="Intermediate size of the feed-forward network in the local encoder/decoder.")


# class REMIRoformerNetworkSchema(BaseModel):
#     """Network-specific hyperparameters for REMI-Roformer."""

#     # 在这里为 REMI-Roformer 添加具体参数
#     hidden_size: int = Field(512, description="Hidden layer dimension.")
#     # ... 其他参数


# # 使用 Union 来支持多种网络配置
# NetworkSchema = Union[OldM2ATransformerNetworkSchema, REMIRoformerNetworkSchema]


# # 3. 创建一个通用的、层次清晰的顶层 ModelSchema
# class ModelSchema(BaseModel):
#     """
#     Generic schema for model configuration, including network, optimizer, and scheduler.
#     """

#     model_type: Literal["Old-M2A-Transformer", "REMI-Roformer"] = Field(..., description="The type of the model to be used.")

#     # 将网络配置嵌套进来
#     network: NetworkSchema = Field(..., description="The network-specific architecture configuration.")

#     # 将优化器和调度器配置嵌套进来
#     optimizer: OptimizerSchema = Field(default_factory=OptimizerSchema, description="Optimizer configuration.")
#     lr_scheduler: LRSchedulerSchema = Field(default_factory=LRSchedulerSchema, description="Learning rate scheduler configuration.")
