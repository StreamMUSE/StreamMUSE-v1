import torch
import pytorch_lightning as L
from collections import deque
import logging
import os
from datetime import datetime
from schema.model_schema import ModelSchema
from schema.model_io_schema import ModelInputData
from schema.model_schema import TrainingProbingLoggerSchema
from typing import Any, Optional
from collections import OrderedDict
import sys
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for better initial visibility
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # All processes print to console initially
)

logger = logging.getLogger(__name__)


class BasePyTorchLightningModel(L.LightningModule):
    def __init__(self, model_schema: ModelSchema):
        super().__init__()
        # Ensure training_probing_training_logger_schema exists in your ModelSchema
        if model_schema.training_probing_training_logger_schema:
            training_probing_schema = model_schema.training_probing_training_logger_schema
        else:
            # Fallback or raise error if schema is critical
            logger.error("training_probing_training_logger_schema is missing in ModelSchema. Using default values.")
            training_probing_schema = TrainingProbingLoggerSchema()  # Use a default

        self.loss_jump_threshold_X = training_probing_schema.loss_jump_threshold_X
        self.loss_avg_window_Y = training_probing_schema.loss_avg_window_Y
        self.recording_window_N = training_probing_schema.recording_window_N

        self.loss_history = deque(maxlen=self.loss_avg_window_Y)
        self.recorded_events = []
        self.batch_buffer = OrderedDict() 
        # self.gradient_buffer = deque(maxlen=self.recording_window_N)
        # self._current_batch_gradients = {}

        # This will be set by _setup_training_probing_logger on rank 0
        self.abnormal_event_save_dir = None


    def _check_for_loss_jump(self, current_loss: torch.Tensor, global_step: int, batch: Any):
        self.loss_history.append(current_loss.item())

        if len(self.loss_history) < self.loss_avg_window_Y:
            return

        avg_loss = sum(self.loss_history) / len(self.loss_history)

        if current_loss.item() > self.loss_jump_threshold_X * avg_loss:
            # Only log this warning on rank 0
            if self.trainer.is_global_zero:
                logger.warning(
                    f"[RANK {self.trainer.global_rank}] Loss jump detected at step {global_step}: "
                    f"Current loss = {current_loss.item():.4f}, "
                    f"Average loss over last {self.loss_avg_window_Y} steps = {avg_loss:.4f}"
                )
            self._record_abnormal_event(current_loss, global_step, avg_loss, batch)

    def _record_abnormal_event(self, current_loss: torch.Tensor, global_step: int, avg_loss: float, batch: ModelInputData):
        # This entire method's file I/O and direct logging should only happen on rank 0
        if self.trainer.is_global_zero:
            lr = self.lr_schedulers().get_last_lr()[0] if self.lr_schedulers() else "N/A"

            event_info = {
                "timestamp": datetime.now().isoformat(),
                "epoch": self.current_epoch,
                "global_step": global_step,
                "current_loss": current_loss.item(),
                "average_loss_Y_steps": avg_loss,
                "learning_rate": lr,
                "gradient_file": None,
                "batch_files": [],  # <--- FIX: Initialize 'batch_files' as an empty list
            }

            # gradients_to_save = {name: grad.cpu() for name, grad in self._current_batch_gradients.items() if grad is not None}

            # if gradients_to_save:
            #     # Ensure abnormal_event_save_dir is set (it will be by _setup_training_probing_logger on rank 0)
            #     if self.abnormal_event_save_dir:
            #         filename = f"global_step_{global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            #         filepath = os.path.join(self.abnormal_event_save_dir, filename)
            #         try:
            #             torch.save(gradients_to_save, filepath)
            #             event_info["gradient_file"] = filepath
            #             logger.info(f"[RANK {self.trainer.global_rank}] Gradients saved to {filepath}")
            #         except Exception as e:
            #             logger.error(f"[RANK {self.trainer.global_rank}] Failed to save gradients to {filepath}: {e}", exc_info=True)
            #     else:
            #         logger.warning(
            #             f"[RANK {self.trainer.global_rank}] abnormal_event_save_dir not set. Cannot save gradients for step {global_step}."
            #         )
            # else:
            #     logger.warning(f"[RANK {self.trainer.global_rank}] No gradients to save at step {global_step}.")
            if self.batch_buffer:
                for file_name, b in self.batch_buffer.items():
                    batch_data = {name: tensor.cpu() for name, tensor in b.model_dump().items() if isinstance(tensor, torch.Tensor)}
                    batch_filepath = os.path.join(self.abnormal_event_save_dir, file_name)
                    try:
                        torch.save(batch_data, batch_filepath)
                        event_info["batch_files"].append(batch_filepath)
                    except Exception as e:
                        logger.error(f"[RANK {self.trainer.global_rank}] Failed to save batch to {batch_filepath}: {e}", exc_info=True)
                logger.info(f"[RANK {self.trainer.global_rank}] Saved {len(event_info['batch_files'])} batch files for abnormal event at step {global_step}")
            else:
                logger.warning(f"[RANK {self.trainer.global_rank}] Batch buffer is empty at step {global_step}.")
            self.recorded_events.append(event_info)
            self.batch_buffer.clear()  # 清空buffer

    def on_train_batch_end(self, outputs: dict, batch: Any, batch_idx: int):
        loss = outputs.get("loss")
        # 生成唯一文件名
        batch_file_name = f"batch_{self.global_step}_{batch_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        # 记录到buffer
        self.batch_buffer[batch_file_name] = batch
        # 控制buffer长度
        while len(self.batch_buffer) > self.recording_window_N:
            self.batch_buffer.popitem(last=False)
        if loss is not None:
            self._check_for_loss_jump(loss, self.global_step, batch)


    # def on_after_backward(self):
    #     # Gradients are collected locally per rank, so this can run on all ranks.
    #     # Saving these gradients to file, however, must be rank 0.
    #     self._current_batch_gradients = {name: param.grad.clone() if param.grad is not None else None for name, param in self.named_parameters()}

    def _setup_training_probing_logger(self):
        # This entire setup process should only happen on rank 0
        if self.trainer.is_global_zero:
            try:
                # self.logger is the PyTorch Lightning logger (e.g., TensorBoardLogger)
                # Its .log_dir is the full path like ./logs/experiment_name/version_X
                self.base_output_dir = self.logger.log_dir
                self.abnormal_event_save_dir = os.path.join(self.base_output_dir, "abnormal_event_records")
                os.makedirs(self.abnormal_event_save_dir, exist_ok=True)

                # Add a specific FileHandler for abnormal events (only on rank 0's logger)
                file_handler = logging.FileHandler(os.path.join(self.abnormal_event_save_dir, "abnormal_event_records.log"))
                file_handler.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)  # Add to the module-level logger
                logger.info(
                    f"[RANK {self.trainer.global_rank}] Training probing logger initialized. Abnormal event records will be saved to: {self.abnormal_event_save_dir}"
                )
            except AttributeError:
                # This might happen if self.logger is not yet initialized, or has no log_dir
                logger.warning(
                    f"[RANK {self.trainer.global_rank}] Could not find 'self.logger.log_dir'. "
                    "Falling back to local 'abnormal_event_records' directory for rank 0. "
                    "Please ensure your Trainer setup correctly initializes loggers."
                )
                # Fallback path for rank 0 if log_dir is not available
                self.abnormal_event_save_dir = "abnormal_event_records"
                os.makedirs(self.abnormal_event_save_dir, exist_ok=True)
        else:
            # For non-rank 0 processes, we don't set up file logging or special directories
            # The 'abnormal_event_save_dir' will remain None for non-rank 0,
            # and _record_abnormal_event will handle this by skipping file saves.
            pass

    def on_fit_start(self):
        # Call the setup function, which internally handles rank 0 logic
        self._setup_training_probing_logger()
        return super().on_fit_start()
    
    def configure_optimizers(self):
        max_lr = 1e-4
        MAX_STEPS =50000
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]