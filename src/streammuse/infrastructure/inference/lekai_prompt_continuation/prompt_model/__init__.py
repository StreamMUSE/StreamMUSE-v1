"""Vendored Lekai prompt-model implementation.

Source: /data/home/yuanxin/lekai/real_time/prompt_model
"""

# Keep this package init lightweight. Importing config/model pulls in
# transformers, which is only needed when a prompt checkpoint is actually used.

__all__: list[str] = []
