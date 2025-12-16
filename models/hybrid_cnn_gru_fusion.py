
from typing import Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class FeatureAttention(nn.Module):

    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, seq_len, feature_dim)
        scores = self.fc2(self.relu(self.fc1(x)))  # (B, seq_len, feature_dim)
        weights = self.softmax(scores)  # across features
        return weights * x


class HybridCNNGRUFusion(nn.Module):

    def __init__(self, input_features: int, cnn_channels: int = 32, gru_hidden: int = 64, attn_heads: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=cnn_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.feature_attn = FeatureAttention(cnn_channels)
        self.gru = nn.GRU(cnn_channels, gru_hidden, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=gru_hidden, num_heads=attn_heads, batch_first=True)
        self.fc_out = nn.Linear(gru_hidden, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        x_t = x.transpose(1, 2)
        conv_out = self.relu(self.conv1(x_t))
        conv_out = conv_out.transpose(1, 2)
        feat = self.feature_attn(conv_out)
        gru_out, _ = self.gru(feat)
        attn_out, attn_weights = self.temporal_attn(gru_out, gru_out, gru_out)
        out = self.fc_out(attn_out[:, -1, :])
        return out.squeeze(-1), attn_weights.detach()


class HybridModule(pl.LightningModule):
    def __init__(self, input_features: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = HybridCNNGRUFusion(input_features)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
