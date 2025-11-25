import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.static_dim = int(static_dim)
        self.lstm_hidden = int(lstm_hidden)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = bool(use_attention)


        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


        if self.use_attention:
            attn_in = self.lstm_hidden * self.num_directions
            self.attn_layer = nn.Sequential(
                nn.Linear(attn_in, attn_in),
                nn.Tanh(),
                nn.Linear(attn_in, 1, bias=False),
            )
        else:
            self.attn_layer = None


        self.fc = None
        self.to(self.device)

    def _build_fc(self, combined_dim: int):

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(self.device)

    def _apply_attention(self, seq_out: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        scores = self.attn_layer(seq_out)  
        weights = F.softmax(scores, dim=1)  
        context = torch.sum(weights * seq_out, dim=1)  
        return context, weights.squeeze(-1)

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor = None):


        seq_out, (h_n, c_n) = self.lstm(seq_x)  

        if self.attn_layer is not None:
            context, attn_weights = self._apply_attention(seq_out)
        else:

            if self.bidirectional:

                last_layer = -1
                forward_h = h_n[last_layer - 1] 
                backward_h = h_n[last_layer]   
                context = torch.cat([forward_h, backward_h], dim=1) 
            else:
             
                context = h_n[-1]
            attn_weights = None


        if static_x is not None and static_x.ndim == 2 and static_x.shape[1] > 0:
            combined = torch.cat([context, static_x], dim=1)
        else:
            combined = context


        if self.fc is None:
            self._build_fc(combined.size(1))

        out = self.fc(self.dropout_layer(combined) if hasattr(self, "dropout_layer") else combined)
  
        out = out.view(-1)

        return (out, attn_weights) if attn_weights is not None else out



setattr(LSTMModel, "dropout_layer", nn.Dropout(0.2))

 
 