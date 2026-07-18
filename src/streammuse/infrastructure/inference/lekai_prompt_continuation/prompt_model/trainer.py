"""
Transformer模型训练器
支持分布式训练、混合精度、梯度累积等特性
训练日志会保存在TensorBoard中，使用命令查看: tensorboard --logdir=<log_dir>
"""
import os
import shutil
import time
import torch
from datetime import datetime
from typing import Optional
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

class TransformerTrainer:
    """Transformer模型训练器

    封装了模型训练的完整流程，包括：
    - 优化器和学习率调度器初始化
    - 分布式训练支持
    - 训练和评估循环
    - 模型检查点保存
    - TensorBoard日志记录
    """

    def __init__(self, config, model, train_dataloader, test_dataloader=None, resume_path=None):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.completed_epochs = 0

        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        self.accelerator = self._initialize_accelerator()
        self._prepare_training_components()
        self.accelerator.init_trackers(self.config.tensorboard_log_name)
        self.test_interval_steps = self._calculate_test_interval()

        # 断点续训
        if resume_path:
            self._load_training_state(resume_path)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """配置优化器（bias / LayerNorm 不施加 weight decay）"""
        no_decay_keywords = {"bias", "layernorm", "layer_norm", "norm"}
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 0.1},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.99),
        )

    def _setup_lr_scheduler(self):
        """配置学习率调度器"""
        num_training_steps = int(
            (len(self.train_dataloader) * self.config.num_epochs * 1.5)
            / self.config.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_training_steps * self.config.lr_warmup_ratio)
        return get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _initialize_accelerator(self) -> Accelerator:
        """初始化Accelerator用于分布式训练和混合精度"""
        ddp_kwargs = DistributedDataParallelKwargs()
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=self.config.tensorboard_log_dir,
            device_placement=True,
            kwargs_handlers=[ddp_kwargs]
        )

        # 创建日志目录
        if accelerator.is_main_process and self.config.tensorboard_log_dir:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

        return accelerator

    def _prepare_training_components(self):
        """准备训练组件（处理分布式、混合精度等）"""
        components = [self.model, self.optimizer, self.train_dataloader, self.lr_scheduler]

        if self.test_dataloader is not None:
            components.insert(3, self.test_dataloader)
            prepared = self.accelerator.prepare(*components)
            self.model, self.optimizer, self.train_dataloader, self.test_dataloader, self.lr_scheduler = prepared
        else:
            prepared = self.accelerator.prepare(*components)
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = prepared

    def _calculate_test_interval(self) -> Optional[int]:
        """计算测试评估的间隔步数"""
        if self.config.use_test_set and self.test_dataloader is not None:
            interval = int(len(self.train_dataloader) * self.config.test_frequency)
            print(f"测试间隔: 每 {interval} 步测试一次 (约{self.config.test_frequency}个epoch)")
            return interval
        return None

    def train(self):
        """执行完整的训练流程"""
        for epoch in range(self.completed_epochs, self.config.num_epochs):
            self._train_one_epoch(epoch)
            self._save_checkpoint_if_needed(epoch)
            # 每个 epoch 结束保存可恢复的训练状态
            self.completed_epochs = epoch + 1
            self._save_training_state()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.config.log)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_start_step = self.global_step

        # 吞吐量计数器
        tokens_since_log = 0
        time_since_log = time.time()

        for batch in self.train_dataloader:
            loss = self._training_step(batch)
            tokens_since_log += batch["input_ids"].numel()

            # 记录日志（含吞吐量）
            if self.global_step % 10 == 0:
                elapsed = time.time() - time_since_log
                tokens_per_sec = tokens_since_log / max(elapsed, 1e-6)
                self._log_training_metrics(loss, tokens_per_sec)
                tokens_since_log = 0
                time_since_log = time.time()

            if self._should_evaluate(epoch_start_step):
                self._evaluate_test()

            if self._should_save_checkpoint():
                self._save_checkpoint(f"steps_{self.global_step}")

            if self.config.log:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    lr=self.lr_scheduler.get_last_lr()[0]
                )
            self.global_step += 1

    def _training_step(self, batch) -> torch.Tensor:
        """执行一个训练步骤

        Args:
            batch: 输入的批次数据

        Returns:
            当前批次的损失值
        """
        with self.accelerator.accumulate(self.model):
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                position_ids=batch.get("position_ids"),
            )
            loss = outputs.loss

            # 反向传播
            self.accelerator.backward(loss)

            # 梯度裁剪
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            # 优化器步进
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss

    def _log_training_metrics(self, loss: torch.Tensor, tokens_per_sec: float = 0):
        """记录训练指标到TensorBoard"""
        metrics = {
            "train/loss": loss.detach().item(),
            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "train/tokens_per_sec": tokens_per_sec,
            "train/step": self.global_step,
        }
        self.accelerator.log(metrics, step=self.global_step)

    def _should_evaluate(self, epoch_start_step: int) -> bool:
        """判断是否需要进行测试评估"""
        if self.test_interval_steps is None:
            return False

        return (
            self.global_step % self.test_interval_steps == 0
            and self.global_step > epoch_start_step
        )

    def _evaluate_test(self):
        """在测试集上评估模型"""
        if self.test_dataloader is None:
            return

        self.accelerator.wait_for_everyone()
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        if self.accelerator.is_main_process:
            print(f"\n开始测试集评估 (step {self.global_step})...")

        with torch.no_grad():
            eval_bar = tqdm(
                self.test_dataloader,
                desc="测试中",
                disable=not self.accelerator.is_main_process
            )
            for batch in eval_bar:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch.get("position_ids"),
                )
                gathered_losses = self.accelerator.gather_for_metrics(
                    outputs.loss.detach().reshape(1)
                )
                total_loss += gathered_losses.sum().item()
                num_batches += gathered_losses.numel()

        # 计算平均损失和困惑度
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # 记录测试指标
        if self.config.test_save_results and self.accelerator.is_main_process:
            test_metrics = {
                "test/loss": avg_loss,
                "test/perplexity": perplexity,
                "test/step": self.global_step,
            }
            self.accelerator.log(test_metrics, step=self.global_step)

        if self.accelerator.is_main_process:
            print(f"测试集 - 损失: {avg_loss:.4f}, 困惑度: {perplexity:.4f}")

            if avg_loss < self.best_eval_loss:
                self.best_eval_loss = avg_loss
                self._save_checkpoint("best_model")
                print(f"best model 已更新: loss={self.best_eval_loss:.4f}")

        self.model.train()
        self.accelerator.wait_for_everyone()

    def _should_save_checkpoint(self) -> bool:
        """判断是否需要保存检查点（按步数）"""
        if self.config.save_steps is None or self.config.save_steps <= 0:
            return False

        return (
            self.global_step % self.config.save_steps == 0
            and self.accelerator.is_main_process
        )

    def _save_checkpoint_if_needed(self, epoch: int):
        """根据epoch判断是否需要保存检查点"""
        if not self.accelerator.is_main_process:
            return

        should_save = (
            (epoch + 1) % self.config.save_model_epochs == 0
            or epoch == self.config.num_epochs - 1
        )

        if should_save:
            self._save_checkpoint(f"epoch_{epoch}")

    def _save_checkpoint(self, prefix: str):
        """保存模型权重（用于推理）"""
        with torch.no_grad():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            # 兼容 torch.compile: 取出原始模型
            if hasattr(unwrapped_model, '_orig_mod'):
                unwrapped_model = unwrapped_model._orig_mod
            if prefix == "best_model":
                save_path = os.path.join(self.config.output_dir, prefix)
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)
            else:
                timestamp = datetime.now().strftime("%m%d_%H%M")
                save_path = f"{self.config.output_dir}/{prefix}_{timestamp}"
            unwrapped_model.save_pretrained(save_path)
            print(f"模型已保存至: {save_path}")

    def _save_training_state(self):
        """保存完整训练状态（可断点续训）"""
        save_path = os.path.join(self.config.output_dir, "latest")
        # Accelerate 保存 model / optimizer / scheduler / scaler / rng
        self.accelerator.save_state(save_path)
        if self.accelerator.is_main_process:
            torch.save({
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'completed_epochs': self.completed_epochs,
            }, os.path.join(save_path, 'training_meta.pt'))
            print(f"训练状态已保存至: {save_path}")

    def _load_training_state(self, load_path):
        """从检查点恢复完整训练状态"""
        self.accelerator.load_state(load_path)
        meta_path = os.path.join(load_path, 'training_meta.pt')
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location='cpu')
            self.global_step = meta['global_step']
            self.best_eval_loss = meta['best_eval_loss']
            self.completed_epochs = meta['completed_epochs']
        print(f"已恢复训练状态: epoch={self.completed_epochs}, step={self.global_step}, best_loss={self.best_eval_loss:.4f}")
