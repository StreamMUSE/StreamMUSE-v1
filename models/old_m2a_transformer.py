import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerConfig, RoFormerEncoder
from schema.model_io_schema import M2AModelInputData
from schema.model_schema import OldM2ATransformerSchema
import pytorch_lightning as L
from typing import Optional


TRAIN_LENGTH = 192
MAX_STEPS = 1000000

# Indicator: 0
# pitch+duration*2: 3200 (25*128)
N_NORMAL_TOKENS = 3202
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


class OldM2ATransformer(L.LightningModule):
    def __init__(self, model_schema: OldM2ATransformerSchema):
        super().__init__()
        large = model_schema.large
        self.hidden_size = model_schema.hidden_size
        self.num_layers = model_schema.num_layers
        self.num_attention_heads = model_schema.num_attention_heads
        self.intermediate_size = model_schema.intermediate_size
        self.local_model_num_layers = model_schema.local_model_num_layers
        self.local_model_num_attention_heads = model_schema.local_model_num_attention_heads
        self.local_model_intermediate_size = model_schema.local_model_intermediate_size
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)
        # self.token_type_embeddings.weight.requires_grad_(True)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)
        # self.type_classifier = nn.Linear(self.hidden_size, 2)
        # self.type_classifier.weight.requires_grad_(False)

        # self.type_scale = nn.Parameter(torch.tensor(1.0))
        # self.mix_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.type_proj = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.ReLU()
        # )

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x, token_type_ids):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)

        # prepend SOS:
        x = torch.cat([torch.full((x.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x.device), x], dim=-1)  # now [B*seq_len, subseq_len+1]

        mask = x != PAD_TOKEN  # [B*seq_len, subseq_len+1]
        word_emb = self.local_embedding(x)  # → [B*seq_len, subseq_len+1, H]

        type_emb = self.token_type_embeddings(token_type_ids)  # [B*seq_len, subseq_len+1, H]
        type_emb = type_emb.view(batch_size * seq_len, word_emb.shape[1], -1)

        emb = word_emb + type_emb
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]

        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        # print(emb.shape) #3840 8 512
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)

        for i in range(max_subseq_len):
            h_ = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            if temperature == 0:
                p = F.one_hot(self.final_decoder(h_).argmax(dim=-1), N_TOKENS).float()
            else:
                p = F.softmax(self.final_decoder(h_[:, -1]) / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = PAD_TOKEN
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break

            # 5a) now append the embedding (always ACCOMPANIMENT), so token_type_ids = 1
            emb = torch.cat([emb, self.local_embedding(y_next) + self.token_type_embeddings(torch.ones_like(y_next))], dim=1)

        return y

    def global_sampling(self, x, x_mel_gt=None, max_seq_len=384, temperature=1.0):
        batch_size, seq_len, subseq_len = x.shape
        _, seq_len_gt, _ = x_mel_gt.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, _ = self.local_encode(x, token_type_ids)
        h_mel, _ = self.local_encode(
            x_mel_gt, torch.zeros(*x_mel_gt.shape[:-1], x_mel_gt.shape[-1] + 1, device=x_mel_gt.device, dtype=x_mel_gt.dtype)
        )
        h = h.view(batch_size, seq_len, -1)
        h_mel = h_mel.view(batch_size, seq_len_gt, -1)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt != None:
            # print('with gt!')
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    # print('Sampling', i, '/', max_seq_len)
                    ...
                if i % 2 == 0:
                    h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                    y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                    y.append(y_next)
                    b, s, l = y_next.unsqueeze(1).shape
                    token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
                    h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
                else:
                    # token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                    h_prev_mel = h_mel[:, i // 2, :].unsqueeze(1)  # [B, 1, H]
                    h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                    y.append(x_mel_gt[:, i // 2, :])
        else:
            for i in range(0, max_seq_len):
                # if i % 10 == 0:
                #     print('Sampling', i, '/', max_seq_len)
                h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                y.append(y_next)
                b, s, l = y_next.unsqueeze(1).shape
                if i % 2 == 0:
                    token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
                else:
                    token_type_ids = torch.zeros((b, s, l + 1), dtype=torch.long, device=y_next.device)
                h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
        return y

    def global_sampling_from_scratch(self, x_mel: torch.LongTensor, temperature: float = 1.0, max_seq_len=384):
        B, S, L = x_mel.shape
        device = x_mel.device

        # Build program IDs = 0 for all melody tokens
        # token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, L]
        h_mel, _ = self.local_encode(x_mel, torch.zeros((B, S, L + 1), device=device, dtype=x_mel.dtype))
        h_mel = h_mel.view(B, S, self.hidden_size)  # [B, S, H]

        # Prepare SOS for global
        sos = self.global_sos.view(1, 1, -1).repeat(B, 1, 1)  # [B, 1, H]

        # Will store generated accompaniment frames
        y = []  # each entry: [B, L]

        # Start with just [SOS]
        h = sos  # [B, 1, H]
        for t in range(max_seq_len):
            # if t % 10 == 0:
            #         print('Sampling', t, '/', max_seq_len)
            if t > 0:
                # Append previous melody summary before generating new accompaniment
                h_prev_mel = h_mel[:, t - 1, :].unsqueeze(1)  # [B, 1, H]
                h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                y.append(x_mel[:, t - 1, :])

            h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
            y_next = self.local_sampling(h_out[:, -1], max_subseq_len=L, temperature=temperature)
            y.append(y_next)
            b, s, l = y_next.unsqueeze(1).shape
            token_type_ids = torch.ones((b, s, l + 1), dtype=torch.long, device=y_next.device)
            h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids=token_type_ids)[0].unsqueeze(1)], dim=1)
        return y  # list of S tensors [B, L]

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if self._future_mask.size(0) == 0 or (not self._future_mask.device == tensor.device) or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(torch.zeros([dim, dim])), 1)
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, x):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        torch.cuda.memory._record_memory_history() # tool for GPU memory
        batch_size, seq_len, subseq_len = x.shape  # 10*384*8
        assert seq_len % 2 == 0, "Expected even number of frames (2*S interleaved)."

        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, emb = self.local_encode(x, token_type_ids)
        h = h.view(batch_size, seq_len, -1)  # 这是每一帧的summary

        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

        # print(h.shape)
        h = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[
            0
        ]  ##all the sos of every timestep (considering other timestep)
        return self.local_decode(h, emb)

    def preprocess(
        self,
        x: torch.LongTensor,  # melody
        pitch_shift: torch.LongTensor,
        y: Optional[torch.LongTensor] = None,  # accompaniment
    ):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // 3, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255  # pitch is 255 that need to be pad
        eos_indices = x[:, :, :, 0] == 254  # program is 254
        is_not_drum = x[:, :, :, 0] != 127
        x_processed[:, :, :, 0] = 0  # program 不变
        x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN

        if y == None:
            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2)
        else:
            batch_size_y, seq_length_y, subseq_length_y = y.shape
            y = y.long().view(batch_size_y, seq_length_y, subseq_length_y // 3, 3)
            y_processed = torch.zeros(batch_size_y, seq_length_y, subseq_length_y // 3, 2, dtype=torch.long, device=y.device)
            pad_indices_y = y[:, :, :, 1] == 255  # pitch is 255 that need to be pad
            eos_indices_y = y[:, :, :, 0] == 254  # program is 254
            is_not_drum_y = y[:, :, :, 0] != 127
            y_processed[:, :, :, 0] = 1  # program 不变
            y_processed[:, :, :, 1] = y[:, :, :, 1] + (y[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum_y
            y_processed[pad_indices_y] = PAD_TOKEN
            y_processed[:, :, :, 0][eos_indices_y] = EOS_TOKEN

            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2), y_processed.view(
                batch_size_y, seq_length_y, subseq_length_y // 3 * 2
            )

    def loss(self, x_mel, x_acc, pitch_shift):
        # x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y = x_acc)
        x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y=x_acc)
        batch_size, seq_len, subseq_len = x_mel.shape  # 10*384*8
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)

        x_target = x.clone()
        # build a mask: True at every odd timestep
        idx = torch.arange(seq_len * 2, device=x.device)
        mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)  # [1, 2*S, 1]
        mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)  # [B, 2*S, L]
        x_target[mel_mask] = PAD_TOKEN

        y = self(x)

        return F.cross_entropy(y.view(-1, N_TOKENS), x_target.view(-1), ignore_index=PAD_TOKEN)

    def training_step(self, batch: M2AModelInputData, batch_idx):
        batch = self._move_to_device(batch)
        x_mel, x_acc, pitch_shift = batch.mel_data, batch.acc_data, batch.pitch_shift
        loss = self.loss(x_mel, x_acc, pitch_shift)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log("training/lr", scheduler.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch: M2AModelInputData, batch_idx):
        batch = self._move_to_device(batch)
        x_mel, x_acc, pitch_shift = batch.mel_data, batch.acc_data, batch.pitch_shift
        loss = self.loss(x_mel, x_acc, pitch_shift)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        max_lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]

    def _move_to_device(self, batch: M2AModelInputData) -> M2AModelInputData:
        """
        Move the batch data to the appropriate device.
        Args:
            batch (M2AModelInputData): The input batch data.
        Returns:
            M2AModelInputData: The batch data moved to the appropriate device.
        """
        return M2AModelInputData(
            mel_data=batch.mel_data.to(self.device),
            acc_data=batch.acc_data.to(self.device),
            pitch_shift=batch.pitch_shift.to(self.device),
        )
