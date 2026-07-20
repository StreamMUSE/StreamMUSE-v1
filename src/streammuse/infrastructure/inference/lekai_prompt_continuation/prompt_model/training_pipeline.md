# Training Pipeline Architecture

```
train.py::main()
│
├─ TrainingConfig()                          # config.py
│  train_batch_size = 2
│  gradient_accumulation_steps = 16
│  mixed_precision = "fp16"
│  learning_rate = 1e-4
│  num_epochs = 40
│  ...
│
├─ ModelConfig()                             # config.py
│  vocab_size = 268
│  hidden_size = 768, layers = 16, heads = 6
│  train_cutoff_len = 1024
│  ...
│
├─ create_llama_config(model_config,         # config.py
│      attn_implementation="flash_attention_2")
│  └─► LlamaConfig(...)
│
│
│ ┌─────────────────────────────────────────────────────────────────┐
│ │                    DATA PIPELINE                                │
│ │                                                                 │
│ │  create_datasets(train_config, model_config)      # train.py   │
│ │  │                                                              │
│ │  ├─► PianoDataset(truncate=False, mode='train')                │
│ │  │   │                                 # PianoDataset.py       │
│ │  │   │  .npz files (192K)                                      │
│ │  │   │       │                                                  │
│ │  │   │  __getitem__(idx)                                       │
│ │  │   │       │                                                  │
│ │  │   │       ├─ np.load(file.npz)                              │
│ │  │   │       │    metadata: {time_signature_idx, bpm, ...}     │
│ │  │   │       │    measure_0..N: (4, 88, t)  piano roll         │
│ │  │   │       │                                                  │
│ │  │   │       ├─ pitch_shift (70% prob, -5~+5)                  │
│ │  │   │       │                                                  │
│ │  │   │       └─ tokenizer.build_training_sequence()            │
│ │  │   │            │                    # my_tokenizer.py       │
│ │  │   │            ├─ encode_time_sig(ts_idx) → ts_token        │
│ │  │   │            ├─ encode_bpm(bpm) → bpm_token               │
│ │  │   │            ├─ per measure:                               │
│ │  │   │            │   per beat:                                 │
│ │  │   │            │     piano_roll → patch_tokens (三进制)      │
│ │  │   │            │     → compress (相对位置编码)               │
│ │  │   │            │                                             │
│ │  │   │            └─► [BOS TS BPM bar beat TRK_MEL mel...      │
│ │  │   │                 TRK_ACC acc... beat... bar... EOS]       │
│ │  │   │                                                          │
│ │  │   │  returns: {input_ids: (L,), labels: (L,)}              │
│ │  │   │           L = 变长, 536~5920 tokens                     │
│ │  │   │                                                          │
│ │  │                                                              │
│ │  └─► PianoDataset(truncate=True, mode='test')                  │
│ │                                                                 │
│ │                                                                 │
│ │  create_dataloaders(...)                          # train.py   │
│ │  │                                                              │
│ │  │  ┌─ TRAIN ──────────────────────────────────────────────┐   │
│ │  │  │                                                      │   │
│ │  │  │  StreamingDataset(base=train_dataset)                │   │
│ │  │  │  │                          # PianoDataset.py        │   │
│ │  │  │  │                                                   │   │
│ │  │  │  │  Concatenate all sequences into continuous stream │   │
│ │  │  │  │  ┌──────────────────────────────────────────────┐ │   │
│ │  │  │  │  │seq1 EOS BOS seq2 EOS BOS seq3 EOS BOS seq4..│ │   │
│ │  │  │  │  └──────────────────────────────────────────────┘ │   │
│ │  │  │  │       │                                           │   │
│ │  │  │  │  Slice into fixed-length chunks (1024 tokens)     │   │
│ │  │  │  │  ┌──────────┐┌──────────┐┌──────────┐            │   │
│ │  │  │  │  │ chunk 0  ││ chunk 1  ││ chunk 2  │ ...        │   │
│ │  │  │  │  │ 1024 tok ││ 1024 tok ││ 1024 tok │            │   │
│ │  │  │  │  └──────────┘└──────────┘└──────────┘            │   │
│ │  │  │  │       │                                           │   │
│ │  │  │  │  Track sequence boundaries per chunk:             │   │
│ │  │  │  │                                                   │   │
│ │  │  │  │  yield {                                          │   │
│ │  │  │  │    input_ids:      (1024,)  token 序列            │   │
│ │  │  │  │    labels:         (1024,)  = input_ids           │   │
│ │  │  │  │    attention_mask: (1024,)  序列ID [1,1,..,2,2,..]│   │
│ │  │  │  │    position_ids:   (1024,)  每段从0 [0,1,..,0,1..]│  │
│ │  │  │  │  }                                                │   │
│ │  │  │  │                                                   │   │
│ │  │  │  │  100% token utilization, zero padding             │   │
│ │  │  │  │                                                   │   │
│ │  │  │  └─► DataLoader(batch_size=2, num_workers=8)         │   │
│ │  │  │      └─► batch: each tensor (2, 1024)                │   │
│ │  │  │                                                      │   │
│ │  │  └──────────────────────────────────────────────────────┘   │
│ │  │                                                              │
│ │  │  ┌─ TEST ───────────────────────────────────────────────┐   │
│ │  │  │                                                      │   │
│ │  │  │  PaddingCollator(max_seq_len=1024)                   │   │
│ │  │  │  │                          # PianoDataset.py        │   │
│ │  │  │  │  Pad to max length in batch                       │   │
│ │  │  │  │  attention_mask: 1/0 (valid/pad)                  │   │
│ │  │  │  │                                                   │   │
│ │  │  │  └─► DataLoader(batch_size=4, shuffle=False)         │   │
│ │  │  │      └─► batch: (4, dynamic_len)                     │   │
│ │  │  │                                                      │   │
│ │  │  └──────────────────────────────────────────────────────┘   │
│ │                                                                 │
│ └─────────────────────────────────────────────────────────────────┘
│
│
│ ┌─────────────────────────────────────────────────────────────────┐
│ │                       MODEL                                     │
│ │                                                                 │
│ │  initialize_model(llama_config)                   # train.py   │
│ │  │                                                              │
│ │  └─► PianoLLaMA(config)                           # model.py   │
│ │      │  _supports_flash_attn = True                            │
│ │      │                                                          │
│ │      └─ self.model = LlamaForCausalLM(config)                  │
│ │         │  attn: FlashAttention2 (训练)                         │
│ │         │  hidden=768, layers=16, heads=6                       │
│ │         │  vocab=268, max_pos=8192                              │
│ │         │  params: 151M                                         │
│ │         │                                                       │
│ │         forward(input_ids, attention_mask, position_ids, labels)│
│ │         │                                                       │
│ │         │  attention_mask = 序列ID → cu_seqlens                │
│ │         │  → FlashAttention varlen (跨序列隔离)                │
│ │         │                                                       │
│ │         │  position_ids 每段从0 → RoPE 独立编码                │
│ │         │                                                       │
│ │         └─► CausalLMOutput(loss, logits)                        │
│ │             loss: CrossEntropy(mean), ignore -100               │
│ │                                                                 │
│ └─────────────────────────────────────────────────────────────────┘
│
│
│ ┌─────────────────────────────────────────────────────────────────┐
│ │                      TRAINER                                    │
│ │                                                                 │
│ │  TransformerTrainer(config, model, dataloader, resume_path)    │
│ │  │                                            # trainer.py     │
│ │  │                                                              │
│ │  ├─ optimizer: AdamW(lr=1e-4, betas=(0.9,0.99))               │
│ │  │    param groups:                                             │
│ │  │      decay (114 params):    weight_decay=0.1                │
│ │  │      no_decay (33 params):  weight_decay=0.0  (norm/bias)   │
│ │  │                                                              │
│ │  ├─ lr_scheduler: cosine + warmup (2% warmup ratio)            │
│ │  │                                                              │
│ │  ├─ accelerator: Accelerate                                    │
│ │  │    mixed_precision = "fp16"                                  │
│ │  │    gradient_accumulation_steps = 16                          │
│ │  │    log_with = "tensorboard"                                  │
│ │  │                                                              │
│ │  │                                                              │
│ │  │  train() loop:                                               │
│ │  │  │                                                           │
│ │  │  for epoch in range(completed_epochs, num_epochs):           │
│ │  │  │                                                           │
│ │  │  │  for batch in train_dataloader:                           │
│ │  │  │  │                                                        │
│ │  │  │  │  ┌─ _training_step(batch) ────────────────────────┐   │
│ │  │  │  │  │                                                │   │
│ │  │  │  │  │  with accelerator.accumulate(model):           │   │
│ │  │  │  │  │    outputs = model(                            │   │
│ │  │  │  │  │      input_ids      = batch["input_ids"],      │   │
│ │  │  │  │  │      labels         = batch["labels"],         │   │
│ │  │  │  │  │      attention_mask = batch["attention_mask"],  │   │
│ │  │  │  │  │      position_ids   = batch["position_ids"],   │   │
│ │  │  │  │  │    )                                           │   │
│ │  │  │  │  │    loss = outputs.loss                         │   │
│ │  │  │  │  │    accelerator.backward(loss)                  │   │
│ │  │  │  │  │    clip_grad_norm_(1.0)                        │   │
│ │  │  │  │  │    optimizer.step()                            │   │
│ │  │  │  │  │    lr_scheduler.step()                         │   │
│ │  │  │  │  │                                                │   │
│ │  │  │  │  └────────────────────────────────────────────────┘   │
│ │  │  │  │                                                        │
│ │  │  │  │  every 10 steps:                                       │
│ │  │  │  │    log train/loss, train/lr, train/tokens_per_sec      │
│ │  │  │  │                                                        │
│ │  │  │  │  every test_interval steps:                            │
│ │  │  │  │    _evaluate_test() → log test/loss, test/perplexity   │
│ │  │  │  │    if best → _save_checkpoint("best_model")            │
│ │  │  │  │                                                        │
│ │  │  │  end epoch                                                │
│ │  │  │  │                                                        │
│ │  │  │  ├─ _save_checkpoint_if_needed(epoch)                     │
│ │  │  │  │    unwrap DDP + torch.compile → save_pretrained()      │
│ │  │  │  │                                                        │
│ │  │  │  └─ _save_training_state()                                │
│ │  │  │      accelerator.save_state("checkpoints/latest/")        │
│ │  │  │      + training_meta.pt (global_step, best_loss, epoch)   │
│ │  │  │                                                           │
│ │  │                                                              │
│ │  │  resume: _load_training_state("checkpoints/latest/")         │
│ │  │    accelerator.load_state() + training_meta.pt               │
│ │  │    → skip completed epochs                                   │
│ │  │                                                              │
│ └─────────────────────────────────────────────────────────────────┘
│
│
│  Outputs:
│    checkpoints/best_model/          model weights (safetensors)
│    checkpoints/epoch_N_MMDD_HHMM/   model weights per epoch
│    checkpoints/latest/              full state (resume)
│    logs/                            TensorBoard logs
```
