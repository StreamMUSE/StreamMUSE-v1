import torch
import pytorch_lightning as L
from collections import deque
import logging
import os
from datetime import datetime
from schema.model_schema import ModelSchema
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasePyTorchLightningModel(L.LightningModule):
    def __init__(self, model_schema: ModelSchema):
        super().__init__()
        training_probing_schema = model_schema.training_probing_training_logger_schema
        
        self.loss_jump_threshold_X = training_probing_schema.loss_jump_threshold_X
        self.loss_avg_window_Y = training_probing_schema.loss_avg_window_Y
        self.recording_window_N = training_probing_schema.recording_window_N

        self.loss_history = deque(maxlen=self.loss_avg_window_Y)
        self.recorded_events = [] # Stores info about recorded events
        self.gradient_buffer = deque(maxlen=self.recording_window_N) # Stores gradients for N steps
        self._current_batch_gradients = {} # Temporary storage for gradients of current batch

    def _check_for_loss_jump(self, current_loss: torch.Tensor, global_step: int):
        self.loss_history.append(current_loss.item())

        if len(self.loss_history) < self.loss_avg_window_Y:
            # Not enough history to calculate average
            return

        avg_loss = sum(self.loss_history) / len(self.loss_history)

        if current_loss.item() > self.loss_jump_threshold_X * avg_loss:
            logger.warning(
                f"Loss jump detected at step {global_step}: "
                f"Current loss = {current_loss.item():.4f}, "
                f"Average loss over last {self.loss_avg_window_Y} steps = {avg_loss:.4f}"
            )
            self._record_abnormal_event(current_loss, global_step, avg_loss)

    def _record_abnormal_event(self, current_loss: torch.Tensor, global_step: int, avg_loss: float):
        # Get current learning rate
        lr = self.lr_schedulers().get_last_lr()[0] if self.lr_schedulers() else 'N/A'

        # Log simple information
        event_info = {
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "global_step": global_step,
            "current_loss": current_loss.item(),
            "average_loss_Y_steps": avg_loss,
            "learning_rate": lr,
            "gradient_file": None # Placeholder, will be filled after saving
        }
        
        # Save gradients to a .pt file
        gradients_to_save = {name: grad.cpu() for name, grad in self._current_batch_gradients.items() if grad is not None}
        if gradients_to_save:
            save_dir = "abnormal_event_records"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"gradients_step_{global_step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            filepath = os.path.join(save_dir, filename)
            torch.save(gradients_to_save, filepath)
            event_info["gradient_file"] = filepath
            logger.info(f"Gradients saved to {filepath}")
        else:
            logger.warning(f"No gradients to save at step {global_step}.")

        self.recorded_events.append(event_info)
        # Optionally, you might want to limit the size of recorded_events if it grows too large
        # For now, we'll let it grow, assuming events are rare.

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # outputs can be a tensor or a dict, depending on what training_step returns
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            logger.warning("Could not extract loss from training_step outputs. Skipping loss jump check.")
            return

        self._check_for_loss_jump(loss, self.global_step)
        
        # Clear gradients for the next batch
        self._current_batch_gradients = {}

    def on_after_backward(self):
        # Capture gradients after backward pass
        self._current_batch_gradients = {
            name: param.grad.clone() if param.grad is not None else None
            for name, param in self.named_parameters()
        }
