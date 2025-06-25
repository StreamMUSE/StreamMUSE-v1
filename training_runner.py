from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch.utilities.seed import seed_everything
from schema.project_schema import ProjectSchema
from datamodules.remi_json_datamodule import MelAccRemiJsonDataModule
import os
import shutil
import logging
import sys

# --- Initial Logging Setup (for early messages and console output) ---
# This logger will catch messages before the experiment directory is known
# It will print to console and can optionally save to a temporary log file
# or a general log file if you need.
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Always log to console
        # You could add a FileHandler here for general logs outside specific exp dirs
        # logging.FileHandler("general_early_logs.log")
    ],
)

# Get a logger for this module
logger = logging.getLogger(__name__)
logger.info("Application started. Initial logging setup complete.")


class ProjectRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            self.config = ProjectSchema.from_yaml(config_path)
        except Exception as e:
            logger.critical(f"Fatal error: Failed to load configuration from {config_path}: {e}", exc_info=True)
            sys.exit(1)  # Exit if config loading is critical

        self.datamodule = None
        self.model = None
        self.loggers = None
        self.trainer = None

    def setup_datamodule(self):
        try:
            if self.config.dataset.tokenizer == "REMI":
                self.datamodule = MelAccRemiJsonDataModule(
                    config=self.config.dataset,
                )
            elif self.config.dataset.tokenizer == "XinYue's":
                from datamodules.old_pt_datamodule import OldPtDataModule

                self.datamodule = OldPtDataModule(
                    config=self.config.dataset,
                )
            else:
                logger.warning(
                    f"Unsupported tokenizer type: {self.config.dataset.tokenizer}. Please ensure this is intentional or define a new tokenizer type."
                )
                raise ValueError(f"Unsupported tokenizer type: {self.config.dataset.tokenizer}")
        except Exception as e:
            logger.critical(f"Fatal error setting up datamodule with tokenizer '{self.config.dataset.tokenizer}': {e}", exc_info=True)
            raise  # Re-raise to stop if datamodule setup fails

    def setup_model(self):
        try:
            if self.config.model.model_type == "Old-M2A-Transformer":
                from models.old_m2a_transformer import OldM2ATransformer

                self.model = OldM2ATransformer(
                    model_schema=self.config.model,
                )
            elif self.config.model.model_type == "REMI-Roformer":
                from models.remi_roformer import REMIRoformer

                self.model = REMIRoformer(
                    model_schema=self.config.model,
                )
            else:
                logger.warning(f"Unsupported model type: {self.config.model.model_type}. Check your config or implement the model.")
                raise ValueError(f"Unsupported model type: {self.config.model.model_type}")
        except Exception as e:
            logger.critical(f"Fatal error setting up model of type '{self.config.model.model_type}': {e}", exc_info=True)
            raise  # Re-raise to stop if model setup fails

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
            # Example check for a specific callback
            # if not any(isinstance(cb, ModelCheckpoint) for cb in self.config.trainer.callbacks):
            #     logger.warning("ModelCheckpoint callback not found in trainer configuration. Model checkpoints will not be saved automatically.")

            self.trainer = Trainer(
                precision="bf16-mixed", # data precision
                logger=self.loggers,
                val_check_interval=500,
                log_every_n_steps=50,
                **self.config.trainer.model_dump(),
                callbacks=[
                    ModelCheckpoint(
                        every_n_train_steps=1000,
                        save_top_k=5,
                        monitor="val_loss",
                        mode="min",
                        dirpath=f"{self.loggers[0].log_dir}/ckpt",  # Ensure this path is correct
                        filename="{epoch:02d}-{val_loss:.2f}",
                    ),
                ],
                gradient_clip_algorithm="norm",
                gradient_clip_val=1.0,  # Example value, adjust as needed
                check_val_every_n_epoch=10,
                strategy="ddp",
            )
        except Exception as e:
            logger.critical(f"Fatal error setting up PyTorch Lightning Trainer: {e}", exc_info=True)
            raise  # Re-raise to stop if trainer setup fails

    def add_file_logger_to_exp_dir(self):
        """Adds a FileHandler to the main logger to save logs within the experiment directory."""
        if not self.loggers or not hasattr(self.loggers[0], "log_dir"):
            logger.warning("Cannot add file logger: Experiment directory not available from PyTorch Lightning loggers.")
            return

        exp_log_dir = self.loggers[0].log_dir
        os.makedirs(exp_log_dir, exist_ok=True)  # Ensure the directory exists
        log_file_path = os.path.join(exp_log_dir, "experiment_log.log")

        # Create a FileHandler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)  # Set the desired logging level for the file
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the handler to the root logger or your specific logger
        # It's better to add it to the specific logger you're using (__name__)
        # To avoid duplicate logs, you might want to remove existing stream handlers if logging ONLY to file.
        # But for this case, keeping console + file is usually desired.
        logger.addHandler(file_handler)
        logger.info(f"All subsequent logs will also be saved to: {log_file_path}")

    def copy_config(self, trainer: Trainer):
        try:
            log_dir = trainer.logger.log_dir
            os.makedirs(log_dir, exist_ok=True)
            config_path = os.path.join(log_dir, os.path.basename(self.config_path))
            shutil.copy(self.config_path, config_path)
            logger.info(f"Configuration copied to {config_path}")
        except Exception as e:
            logger.error(f"Failed to copy configuration file from {self.config_path} to {config_path}: {e}", exc_info=True)
            # You might decide if failing to copy config should stop the run

    def run_experiment(self):
        logger.info("Starting experiment setup...")
        # seed_everything(self.config.seed)  # Set seed for reproducibility
        try:
            self.setup_datamodule()
            self.setup_model()
            self.setup_loggers()
            self.setup_trainer()  # trainer and loggers are ready here
            self.add_file_logger_to_exp_dir()
            self.copy_config(self.trainer)
            logger.info("Experiment setup complete. Starting training...")
        except Exception as e:
            logger.critical(f"Fatal error during experiment setup: {e}", exc_info=True)
            sys.exit(1)  # Exit if setup fails critically

        try:
            self.trainer.fit(self.model, datamodule=self.datamodule)
            logger.info("Experiment training finished successfully!")
        except Exception as e:
            logger.error(f"An error occurred during model training: {e}", exc_info=True)
            # Decide if you want to re-raise or handle gracefully based on desired behavior
            # raise # Uncomment to re-raise the exception and stop execution

        logger.info("Experiment run process completed.")


if __name__ == "__main__":
    # Example usage
    runner = ProjectRunner(config_path="schema/yaml/old_m2a_transformer_aria_skyline_v0-1.2.yaml")
    # runner = ProjectRunner(config_path="schema/yaml/remi_roformer_pop909-1.0.yaml") # Use your specific config

    try:
        runner.run_experiment()
    except Exception as e:
        # This catches any unhandled exceptions that might slip past internal try-excepts
        # and ensures they are logged before program exit.
        logger.critical(f"Unhandled exception occurred during experiment execution: {e}", exc_info=True)
        sys.exit(1)
