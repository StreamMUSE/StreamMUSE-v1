from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger  # or your preferred logger
from pytorch_lightning.callbacks import ModelCheckpoint  # or other callbacks you need
from schema.project_schema import ProjectSchema  # Assuming you have a config schema
from datamodule.remi_json_dataset import Pop909DataModule
from m2a_transformer_refactor import M2ATransformer  # Your PL model
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
        self.datamodule = Pop909DataModule(
            config=self.config.dataset,
        )

    def setup_model(self):
        self.model = M2ATransformer(
            model_schema=self.config.model,
        )

    def setup_loggers(self):
        from schema.project_schema import CSVLoggerSchema, WandbLoggerSchema, TensorBoardLoggerSchema

        loggers = []
        for name,logger_config in self.config.loggers.items():
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
                ModelCheckpoint(every_n_epochs=1, save_top_k=1, monitor="val_loss", mode="min",dirpath=self.loggers[0].log_dir, filename="{epoch:02d}-{val_loss:.2f}"),
            ],
        )
    
    def scopy_config(self,trainer: Trainer):
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

        self.scopy_config(self.trainer)
        self.trainer.fit(self.model,datamodule= self.datamodule)

        # if self.config.evaluation.run_test:
        #     self.trainer.test(self.model, self.datamodule)

        # Optionally save final results or metrics
        print("Experiment finished!")


if __name__ == "__main__":
    # Example usage in your main.py
    # You would typically parse config_path from CLI arguments
    runner = ProjectRunner(config_path="schema/yaml/m2a-1.yaml")
    runner.run_experiment()
