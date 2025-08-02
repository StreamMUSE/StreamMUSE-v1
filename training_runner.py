import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
# from lightning.pytorch.utilities.seed import seed_everything
from schema.project_schema import ProjectSchema
from datamodules.remi_json_datamodule import MelAccRemiJsonDataModule
import os
import shutil
import logging
import sys
import re  # Added for version parsing if needed, but the current approach uses config.version directly
from typing import List, Union, Any, Optional  # Import necessary types


# --- Initial Logging Setup (for early messages and console output) ---
# This basicConfig applies to ALL processes.
# We will dynamically adjust handlers/levels for non-rank 0 processes later.
# Setting level to INFO initially so all ranks can print some startup messages.
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to allow initial messages from all ranks
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # All processes print to console initially
    ],
)

# Get a logger for this module
logger = logging.getLogger(__name__)
# The first few messages might come from all ranks if this runs before DDP is fully set up.
logger.info("Application started. Initial logging setup complete.")

temp_dir = '/home/ubuntu/stanleyz/StreamMUSE/my-temp-space'
os.environ['TMPDIR'] = temp_dir
os.environ['TMP'] = temp_dir  
os.environ['TEMP'] = temp_dir
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(temp_dir, 'pytorch_cache')
os.environ['TORCH_HOME'] = os.path.join(temp_dir, 'torch')
os.environ['HF_HOME'] = os.path.join(temp_dir, 'huggingface')
# 确保所有目录存在
for dir_path in [temp_dir, 
                 os.environ['PYTORCH_KERNEL_CACHE_PATH'],
                 os.environ['TORCH_HOME'],
                 os.environ['HF_HOME']]:
    os.makedirs(dir_path, exist_ok=True)

class ProjectRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            self.config = ProjectSchema.from_yaml(config_path)
        except Exception as e:
            logger.critical(f"Fatal error: Failed to load configuration from {config_path}: {e}", exc_info=True)
            sys.exit(1)

        # Store the version from config. This will be passed to PL loggers.
        self.logger_version = str(self.config.version)  # Ensure it's a string, e.g., "1.0.0"

        self.datamodule = None
        self.model = None
        self.loggers = None  # Renamed to avoid confusion with Trainer's .logger
        self.trainer = None

    def setup_datamodule(self):
        try:
            if self.config.dataset.tokenizer == "REMI":
                self.datamodule = MelAccRemiJsonDataModule(
                    config=self.config.dataset,
                )
            elif self.config.dataset.tokenizer == "XinYue's":
                from datamodules.old_pt_datamodule import OldPtDataModule  # Import if needed

                self.datamodule = OldPtDataModule(
                    config=self.config.dataset,
                )
            elif self.config.dataset.tokenizer == "New XinYue's":
                from datamodules.new_pt_datamodule import NewPtDataModule  # Import if needed
                self.datamodule = NewPtDataModule(
                    config=self.config.dataset,
                )
            else:
                logger.warning(
                    f"Unsupported tokenizer type: {self.config.dataset.tokenizer}. Please ensure this is intentional or define a new tokenizer type."
                )
                raise ValueError(f"Unsupported tokenizer type: {self.config.dataset.tokenizer}")
        except Exception as e:
            logger.critical(f"Fatal error setting up datamodule with tokenizer '{self.config.dataset.tokenizer}': {e}", exc_info=True)
            raise

    def setup_model(self):
        try:
            # Assuming OldM2ATransformer and REMIRoformer are subclasses of BasePyTorchLightningModel
            # and accept model_schema as argument.
            if self.config.model.model_type == "Old-M2A-Transformer":
                from models.old_m2a_transformer import OldM2ATransformer  # Import if needed

                self.model = OldM2ATransformer(
                    model_schema=self.config.model,
                )
            elif self.config.model.model_type == "REMI-RoFormer":
                from models.remi_roformer import REMIRoformer  # Import if needed

                self.model = REMIRoformer(
                    model_schema=self.config.model,
                )
            elif self.config.model.model_type == "New-M2A-Transformer":
                from models.new_m2a_transformer import NewM2ATransformer

                self.model = NewM2ATransformer(
                    model_schema=self.config.model,
                )
            elif self.config.model.model_type == "Old-M2A-Transformer-Nomask":
                from models.old_m2a_nomask_transformer import OldM2ANomaskTransformer

                self.model = OldM2ANomaskTransformer(
                    model_schema=self.config.model,
                )
            
            else:
                logger.warning(f"Unsupported model type: {self.config.model.model_type}. Check your config or implement the model.")
                raise ValueError(f"Unsupported model type: {self.config.model.model_type}")
        except Exception as e:
            logger.critical(f"Fatal error setting up model of type '{self.config.model.model_type}': {e}", exc_info=True)
            raise

    def setup_loggers(self):
        from schema.project_schema import CSVLoggerSchema, WandbLoggerSchema, TensorBoardLoggerSchema

        loggers = []
        try:
            for name, logger_config in self.config.loggers.items():
                _logger_config = logger_config.model_dump()
                _logger_config.__delitem__("type")
                if isinstance(logger_config, TensorBoardLoggerSchema):
                    loggers.append(TensorBoardLogger(**_logger_config))
                elif isinstance(logger_config, WandbLoggerSchema):
                    loggers.append(WandbLogger(**_logger_config))
                elif isinstance(logger_config, CSVLoggerSchema):
                    loggers.append(CSVLogger(**_logger_config))
                else:
                    logger.warning(f"Unknown logger type: {type(logger_config)}. This logger will be skipped.")
                print(name, _logger_config) # This line was in your original code
            self.loggers = loggers if len(loggers) > 0 else [TensorBoardLogger("logs", name="default")]
            if not self.loggers:
                logger.warning("No loggers were configured or recognized. Defaulting to TensorBoardLogger.")
        except Exception as e:
            logger.error(f"Error setting up PyTorch Lightning loggers: {e}", exc_info=True)
            # Fallback if logger setup fails, but try to continue with a basic logger
            self.loggers = [TensorBoardLogger("logs", name="default_error_fallback")]
            logger.info("Proceeding with a default TensorBoardLogger due to an error in logger setup.")


    def setup_trainer(self):
        try:
            # ModelCheckpoint dirpath needs to be built with the actual logger's log_dir
            # which is only fully resolved after the Trainer initializes the logger.
            # However, ModelCheckpoint's dirpath can accept a string that will be resolved.
            # A common pattern is to make checkpoint dir relative to the main log_dir.
            # So, `self.loggers[0].log_dir` will be the actual path like `./logs/my_exp/version_1.0.0`
            # But that's only available AFTER Trainer initializes it.
            # For dirpath in ModelCheckpoint, you can use a string that PL will resolve or a relative path.
            # A good practice is to make it relative to the main logger's `save_dir/name`
            # and let PL append the version automatically, or directly use the resolved log_dir
            # after the trainer object exists.

            # The most robust way is to set dirpath AFTER trainer.fit starts,
            # for example, via a callback that dynamically sets it.
            # But for now, let's use the assumed path based on `save_dir/name/version`.

            base_log_path = os.path.join(
                self.loggers[0].save_dir,
                self.loggers[0].name,
                f"version_{self.loggers[0].version}",  # This is the full versioned path
            )

            # callbacks_list = [
            #     ModelCheckpoint(
            #         every_n_train_steps=1000,
            #         save_top_k=5,
            #         monitor="val_loss",
            #         mode="min",
            #         dirpath=os.path.join(base_log_path, "checkpoints"),  # Checkpoints within versioned dir
            #         filename="{epoch:02d}-{val_loss:.2f}",
            #     ),
            # ]

            # # **self.config.trainer.model_dump() might include callbacks from config.
            # # It's better to exclude it if you manage callbacks explicitly like this.
            # trainer_config_dict = self.config.trainer.model_dump(exclude={"callbacks"})

            self.trainer = Trainer(
                precision="bf16-mixed",  # data precision
                logger=self.loggers,
                val_check_interval=100,
                limit_val_batches=10,
                log_every_n_steps=1,
                **self.config.trainer.model_dump(),
                callbacks=[
                    ModelCheckpoint(
                        every_n_train_steps=200,
                        save_top_k=5,
                        monitor="val_loss",
                        mode="min",
                        dirpath=f"{self.loggers[0].log_dir}/ckpt",  # Ensure this path is correct
                        filename="{epoch:02d}-{val_loss:.2f}",
                    ),
                ],
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0,  # Example value, adjust as needed
                # strategy="ddp",  # Crucial for distributed training
                strategy="ddp_find_unused_parameters_true",  # Use this for DDP with unused parameters
            )
        except Exception as e:
            logger.critical(f"Fatal error setting up PyTorch Lightning Trainer: {e}", exc_info=True)
            raise

    def _configure_rank_logging(self):
        """
        Configures logging for the current DDP rank.
        Only rank 0 will have a FileHandler and its StreamHandler.
        Non-rank 0 processes will have their StreamHandler removed to suppress console output.
        """
        if self.trainer.is_global_zero:
            # Rank 0: Add a FileHandler for logging to file
            exp_log_dir = self.trainer.logger.log_dir  # Get the actual log directory
            logger.info(f"[RANK {self.trainer.global_rank}] Experiment log directory: {exp_log_dir}")
            os.makedirs(exp_log_dir, exist_ok=True)
            log_file_path = os.path.join(exp_log_dir, "experiment_log.log")

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)  # Set desired logging level for the file
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)  # Add to the specific logger used in this module
            logger.info(f"[RANK {self.trainer.global_rank}] All subsequent logs from rank 0 will also be saved to: {log_file_path}")

            # Set the logger level for rank 0 (optional, can be INFO or DEBUG)
            logger.setLevel(logging.INFO)
        else:
            # Non-rank 0: Remove all existing StreamHandlers to suppress console output
            # Keep other handlers if any, but ensure they don't write to shared resources.
            # Also, set their logging level higher to suppress less critical messages.
            logger.setLevel(logging.ERROR)  # Suppress INFO/WARNING on non-rank 0

            # Remove StreamHandlers to prevent console output
            for handler in list(logger.handlers):  # Iterate over a copy to avoid modification issues
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)
            logger.propagate = False  # Prevent logs from propagating to root logger after handlers are removed
            # Any specific non-rank0 file logging would need to be added here for unique files.
            # For "only rank 0 leaves log", we don't add any.

            # A final check to ensure nothing slips to console after removal
            # This line won't appear if StreamHandlers are successfully removed
            logger.info(f"[RANK {self.trainer.global_rank}] Non-rank 0 logging setup complete (console suppressed).")

    def copy_config(self, trainer: Trainer):
        """Copies the configuration file to the experiment log directory.
        This must ONLY be called by the global_rank 0 process.
        """
        # This check is technically redundant if called within `if self.trainer.is_global_zero:` block,
        # but good for modularity if it were called independently.
        if trainer.is_global_zero:
            try:
                log_dir = trainer.logger.log_dir
                os.makedirs(log_dir, exist_ok=True)
                config_copy_path = os.path.join(log_dir, os.path.basename(self.config_path))
                shutil.copy(self.config_path, config_copy_path)
                logger.info(f"Configuration copied to {config_copy_path}")
            except Exception as e:
                logger.error(f"Failed to copy configuration file from {self.config_path} to {config_copy_path}: {e}", exc_info=True)

    def run_experiment(self):
        # Initial message from all ranks
        logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Starting experiment setup...")

        try:
            self.setup_datamodule()
            self.setup_model()
            self.setup_loggers()  # Loggers instantiated, but log_dir not final yet
            self.setup_trainer()  # Trainer instantiated. self.trainer.is_global_zero is now reliable.

            # --- Configure logging handlers based on rank ---
            self._configure_rank_logging()

            # --- Critical: File IO operations ONLY on rank 0 ---
            # These methods internally also check for is_global_zero for safety,
            # but calling them within this main rank 0 guard ensures clear flow.
            if self.trainer.is_global_zero:
                # `self.add_file_logger_to_exp_dir()` is now `_configure_rank_logging()`'s responsibility
                # But if you want to explicitly call it here, it will be fine as it's guarded.
                self.copy_config(self.trainer)

            # This log will only appear in rank 0's console and file log (if configured)
            # if _configure_rank_logging successfully removed handlers for non-rank 0.
            logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment setup complete. Starting training...")
        except Exception as e:
            logger.critical(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Fatal error during experiment setup: {e}", exc_info=True)
            sys.exit(1)

        try:
            self.trainer.fit(self.model, datamodule=self.datamodule)
            # This log will only appear in rank 0's console and file log
            logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment training finished successfully!")
        except Exception as e:
            # Errors from non-rank 0 will still be printed if `basicConfig` is not completely removed
            # from them. However, `logger.critical` is still level-checked.
            logger.error(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] An error occurred during model training: {e}", exc_info=True)
            # raise # Uncomment to re-raise the exception and stop execution

        # This log will only appear in rank 0's console and file log
        logger.info(f"[{os.environ.get('GLOBAL_RANK', 'N/A')}] Experiment run process completed.")


if __name__ == "__main__":

    # torch.cuda.memory._record_memory_history() # start memory snapshot

    # Example usage
    # runner = ProjectRunner(config_path="schema/yaml/old_m2a_transformer_aria_deduped_skyline_top2_0.5B-1.4.yaml")
    runner = ProjectRunner(config_path="logs/old_m2a_aria/1.0.2/old_m2a_transformer_aria_deduped_skyline_top2_0.5B-1.4.yaml")  # Use your specific config

    try:
        runner.run_experiment()
    except Exception as e:
        # This catches any unhandled exceptions that might slip past internal try-excepts
        # and ensures they are logged before program exit.
        logger.critical(f"Unhandled exception occurred during experiment execution: {e}", exc_info=True)
        sys.exit(1)
