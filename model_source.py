import torch
import torch.nn as nn


class AudioRNN(nn.Module):
    """Bidirectional GRU with mean+max pooling for keyword detection."""

    def __init__(self, input_size=20, hidden_size=77, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2 * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        return self.fc(torch.cat([mean_pool, max_pool], dim=1))
