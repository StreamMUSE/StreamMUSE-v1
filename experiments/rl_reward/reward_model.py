"""Twin pianoroll encoder for melody-accompaniment matching reward.

ReaLchords-style self-supervised match scorer: each part (melody /
accompaniment) encodes to an L2-normalized embedding; reward = cosine
similarity. Trained with symmetric InfoNCE (in-batch mismatches as
negatives) plus a discriminative head (real pair vs corrupted pair).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 128
ROLL_BEATS = 32
TICKS_PER_BEAT = 24
ROLL_TICKS = ROLL_BEATS * TICKS_PER_BEAT  # 768


class PartEncoder(nn.Module):
    """(B, 1, 88, T) binary roll -> (B, EMB_DIM) L2-normalized embedding."""

    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.conv_pitch = nn.Sequential(
            nn.Conv2d(1, 32, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, (22, 1)),  # collapse remaining pitch axis
            nn.ReLU(inplace=True),
        )
        self.time_down = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, ROLL_TICKS // 8, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.out = nn.Linear(dim, dim)

    def forward(self, roll):
        h = self.conv_pitch(roll)          # (B, dim, 1, T/4)
        h = self.time_down(h.squeeze(2))   # (B, dim, T/8)
        h = h.transpose(1, 2)              # (B, T/8, dim)
        h = self.transformer(h + self.pos[:, :h.size(1)])
        h = h.mean(dim=1)
        return F.normalize(self.out(h), dim=-1)


class MatchRewardModel(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.mel_encoder = PartEncoder(dim)
        self.acc_encoder = PartEncoder(dim)
        self.discriminator = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def embed(self, mel_roll, acc_roll):
        return self.mel_encoder(mel_roll), self.acc_encoder(acc_roll)

    def reward(self, mel_roll, acc_roll):
        """Cosine reward at eval time; no gradient through batch stats (use eval mode)."""
        mel, acc = self.embed(mel_roll, acc_roll)
        return (mel * acc).sum(-1)

    def losses(self, mel, acc, acc_neg=None, temperature=0.1):
        """Symmetric InfoNCE + optional same-song hard-negative contrast +
        discriminator BCE (in-batch cross-song fakes and hard negatives)."""
        mel_e, acc_e = self.embed(mel, acc)
        logits = mel_e @ acc_e.t() / temperature
        target = torch.arange(logits.size(0), device=logits.device)
        nce = 0.5 * (F.cross_entropy(logits, target)
                     + F.cross_entropy(logits.t(), target))

        b = mel_e.size(0)
        roll = torch.randperm(b, device=mel_e.device)
        # avoid fixed points in the corruption permutation
        while True:
            clash = roll == torch.arange(b, device=roll.device)
            if not clash.any():
                break
            roll[clash] = roll.roll(1)[clash]
        fake = acc_e[roll]
        real_feat = torch.cat([mel_e, acc_e, mel_e * acc_e], dim=-1)
        fake_feat = torch.cat([mel_e, fake, mel_e * fake], dim=-1)

        hard_acc = 1.0
        if acc_neg is not None:
            neg_e = self.acc_encoder(acc_neg)
            pos_s = (mel_e * acc_e).sum(-1) / temperature
            neg_s = (mel_e * neg_e).sum(-1) / temperature
            nce = nce + F.cross_entropy(
                torch.stack([pos_s, neg_s], dim=1),
                torch.zeros(b, dtype=torch.long, device=mel_e.device))
            hard_acc = (pos_s > neg_s).float().mean().item()
            hard_feat = torch.cat([mel_e, neg_e, mel_e * neg_e], dim=-1)
            fake_feat = torch.cat([fake_feat, hard_feat], dim=0)

        real_score = self.discriminator(real_feat)
        fake_score = self.discriminator(fake_feat)
        disc = F.binary_cross_entropy_with_logits(
            real_score, torch.ones_like(real_score)) + \
            F.binary_cross_entropy_with_logits(
                fake_score, torch.zeros_like(fake_score))
        return nce, disc, logits, hard_acc
