import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output):

        attn_weights = F.softmax(self.attn(gru_output), dim=1)  
        context = torch.sum(attn_weights * gru_output, dim=1)  
        return context, attn_weights.squeeze(-1)


class CNN_GRU_Attn(nn.Module):
    def __init__(self, input_dim, cnn_channels=32, hidden_size=64, num_layers=1, attn_dim=32):
        
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )


        self.attn = AttentionLayer(hidden_size)


        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, seq, static=None):
      
      
        x = seq.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)  

        gru_out, _ = self.gru(cnn_out)  

        context, attn_weights = self.attn(gru_out)  

        output = self.fc(context).squeeze(-1)  

        return output, attn_weights
