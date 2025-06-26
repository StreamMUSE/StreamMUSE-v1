import torch
from roformer import RoFormerForCausalLM
import pytorch_lightning as pl
from schema.model_schema import ModelSchema
from schema.model_io_schema import ModelInputData, ModelOutputData
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics import MetricCollection
import os

class REMIRoformer(pl.LightningModule):
    def __init__(self, model_schema: ModelSchema) -> None:
        super(REMIRoformer, self).__init__()
        self.model_schema = model_schema
        self.model = RoFormerForCausalLM(config=model_schema.network_schema)
        self.tokenizer = ...
        num_classes = self.model_schema.network_schema.vocab_size
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=num_classes, ignore_index=0),
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=0),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, model_input: ModelInputData) -> torch.Tensor:
        input_ids = model_input.mel_data
        input_attention_mask = input_ids != 0
        x = self.model.forward(input_ids=input_ids, attention_mask=input_attention_mask, return_dict=True)
        return x.logits

    def _evaluation_step(self, batch: ModelInputData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一步评估，返回用于计算损失和指标的张量。
        """
        logits = self.forward(batch)
        loss = self._cal_loss(logits, batch.mel_data)
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch.mel_data[:, 1:].contiguous()
        return loss, shifted_logits, shifted_labels

    def training_step(self, batch: ModelInputData, batch_idx: int) -> torch.Tensor:
        batch = self._move_to_device(batch)
        logits = self.forward(batch)
        loss = self._cal_loss(logits, batch.mel_data)
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = batch.mel_data[:, 1:].contiguous()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.mel_data.size(0))

        preds = shifted_logits.view(-1, self.model_schema.network_schema.vocab_size)
        target = shifted_labels.view(-1)
        self.train_metrics.update(preds, target, batch_size=batch.mel_data.size(0))

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch.mel_data.size(0))

        return loss

    def validation_step(self, batch: ModelInputData, batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.
        """
        batch = self._move_to_device(batch)
        loss, shifted_logits, shifted_labels = self._evaluation_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.mel_data.size(0))

        preds = shifted_logits.view(-1, self.model_schema.network_schema.vocab_size)
        target = shifted_labels.view(-1)
        self.val_metrics.update(preds, target, batch_size=batch.mel_data.size(0))

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, batch_size=batch.mel_data.size(0))

        self._save_validation_data(batch, preds, target)
        
        return loss
    
    def _save_validation_data(self, batch: ModelInputData, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Save validation data to disk after validation epoch ends.
        """
        val_save_dir = os.path.join(self.loggers[0].save_dir, self.loggers[0].name)
        os.makedirs(val_save_dir, exist_ok=True)
        torch.save(
            {
                "mel_data": batch.mel_data,
                "acc_data": batch.acc_data,
                "preds": preds,
                "target": target,
            },
            f=f"{val_save_dir}/val_epoch.pt",
        )

    def test_step(self, batch: ModelInputData, batch_idx: int) -> ModelOutputData:
        batch = self._move_to_device(batch)
        loss, shifted_logits, shifted_labels = self._evaluation_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch.mel_data.size(0))
        preds = shifted_logits.view(-1, self.model_schema.network_schema.vocab_size)
        target = shifted_labels.view(-1)
        self.test_metrics.update(preds, target, batch_size=batch.mel_data.size(0))

        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, batch_size=batch.mel_data.size(0))

        # 执行测试阶段的独特任务：生成序列
        generated = self.model.generate(
            input_ids=batch.mel_data,
            attention_mask=(batch.mel_data != 0),
            max_length=512,
            return_dict_in_generate=True,
        )

        return ModelOutputData(
            logits=shifted_logits,
            predicted_ids=generated.sequences,
        )

    def _cal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        logits = logits[:, :-1, :].contiguous().view(-1, self.model_schema.network_schema.vocab_size)
        labels = labels[:, 1:].contiguous().view(-1)
        loss = loss_fct(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.model_schema.optimizer_schema.params)
        return optimizer

    def _move_to_device(self, batch: ModelInputData) -> ModelInputData:
        batch.mel_data = batch.mel_data.to(self.device)
        batch.acc_data = batch.acc_data.to(self.device)
        return batch
