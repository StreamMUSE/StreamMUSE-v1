# config_factory.py

import inspect
from typing import Type, Any, Dict, Optional
from pydantic import BaseModel, Field, create_model

def create_base_schema_from_class_init(
    schema_name: str,
    target_class: Type,
) -> Type[BaseModel]:
    """
    根据目标类的 __init__ 方法签名，动态创建一个基础 Pydantic BaseModel。
    这个基础Schema只包含目标类的__init__参数。
    """
    sig = inspect.signature(target_class.__init__)
    fields: Dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name == 'self' or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        ):
            continue

        param_type = Any
        if param.annotation is not inspect.Parameter.empty:
            param_type = param.annotation

        if param.default is not inspect.Parameter.empty:
            fields[name] = (param_type, param.default)
        else:
            fields[name] = (param_type, ...)

    # 动态创建这个基础 Schema
    BaseInitSchema = create_model(schema_name, **fields, __base__=BaseModel)
    return BaseInitSchema