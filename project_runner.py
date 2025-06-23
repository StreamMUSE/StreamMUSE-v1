from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger  # or your preferred logger
from pytorch_lightning.callbacks import ModelCheckpoint  # or other callbacks you need
from schema.project_schema import ProjectSchema  # Assuming you have a config schema
from datamodules.remi_json_datamodule import MelAccRemiJsonDataModule
import os
import shutil


class ProjectRunner:
    def __init__(self, config_path: str):
        # Load configuration using your schema
        self.config_path = config_path
        self.config = ProjectSchema.from_yaml(config_path)  # Assuming schema handles YAML loading

        self.datamodule = None
        self.model = None
        self.loggers = None
        self.trainer = None

    def setup_datamodule(self):
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
            raise ValueError(f"Unsupported tokenizer type: {self.config.dataset.tokenizer}")

    def setup_model(self):
        if self.config.model.model_type == "M2A-Transformer":
            from models.old_m2a_transformer import OldM2ATransformer

            self.model = OldM2ATransformer(
                model_schema=self.config.model,
            )
        elif self.config.model.model_type == "REMI-RoFormer":
            from models.remi_roformer import REMIRoformer

            self.model = REMIRoformer(
                model_schema=self.config.model,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.model_type}")

    def setup_loggers(self):
        from schema.project_schema import CSVLoggerSchema, WandbLoggerSchema, TensorBoardLoggerSchema

        loggers = []
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
                raise ValueError(f"Unsupported logger type: {type(logger_config)}")
        self.loggers = loggers if len(loggers) > 0 else [TensorBoardLogger("logs", name="default")]

    def setup_trainer(self):
        self.trainer = Trainer(
            logger=self.loggers,
            **self.config.trainer.model_dump(),  # Unpack trainer config
            # ... other trainer params from config
            callbacks=[
                # Add callbacks based on config, e.g., ModelCheckpoint
                ModelCheckpoint(
                    every_n_epochs=1,
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    dirpath=f"{self.loggers[0].log_dir}/ckpt",
                    filename="{epoch:02d}-{val_loss:.2f}",
                ),
            ],
        )

    def copy_config(self, trainer: Trainer):
        log_dir = trainer.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        config_path = os.path.join(log_dir, self.config_path.split("/")[-1])
        shutil.copy(self.config_path, config_path)
        print(f"Configuration copied to {config_path}")

    def run_experiment(self):
        self.setup_datamodule()
        self.setup_model()
        self.setup_loggers()
        self.setup_trainer()

        self.copy_config(self.trainer)
        self.trainer.fit(self.model, datamodule=self.datamodule)

        print("Experiment finished!")


if __name__ == "__main__":
    # Example usage in your main.py
    # You would typically parse config_path from CLI arguments
    # runner = ProjectRunner(config_path="schema/yaml/remi_roformer_pop909-1.0.yaml")
    runner = ProjectRunner(config_path="schema/yaml/old_m2a_transformer_pop909-1.0.yaml")

    runner.run_experiment()
