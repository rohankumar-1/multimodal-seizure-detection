import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Inception-style Conv Block
# -----------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.5, maxpool=False, avgpool=False, batchnorm=True):
        super().__init__()

        self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=2, stride=stride, padding="same")
        self.conv4 = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=stride, padding="same")
        self.conv8 = nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=stride, padding="same")

        self.batchnorm = nn.BatchNorm1d(out_ch * 3) if batchnorm else None
        self.dropout = nn.Dropout(dropout)

        self.maxpool = nn.MaxPool1d(2, padding=1) if maxpool else None
        self.avgpool = nn.AvgPool1d(2, padding=1) if avgpool else None

    def forward(self, x):
        c0 = F.relu(self.conv2(x))
        c1 = F.relu(self.conv4(x))
        c2 = F.relu(self.conv8(x))

        x = torch.cat([c0, c1, c2], dim=1)

        if self.maxpool:
            x = self.maxpool(x)
        elif self.avgpool:
            x = self.avgpool(x)

        if self.batchnorm:
            x = self.batchnorm(x)

        x = self.dropout(x)
        return x


# -----------------------------
# Residual GRU stack
# -----------------------------
class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, l2=0.0):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru3 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.gru4 = nn.GRU(hidden_size * 3, hidden_size, batch_first=True)

        # Note: PyTorch L2 is handled via optimizer weight_decay

    def forward(self, x):
        # x: (B, C, T) → GRU wants (B, T, C)
        x = x.transpose(1, 2)

        g1_out, _ = self.gru1(x)
        g2_out, _ = self.gru2(g1_out)

        g12 = torch.cat([g1_out, g2_out], dim=-1)
        g3_out, _ = self.gru3(g12)

        g123 = torch.cat([g1_out, g2_out, g3_out], dim=-1)
        g4_out, _ = self.gru4(g123)

        # final state only
        return g4_out[:, -1, :]


# -----------------------------
# Main Model
# -----------------------------
class ChronoNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_ch = config.CH
        filters = 32
        stride = 2 if config.strided else 1

        # 3 inception blocks
        self.block1 = InceptionBlock(self.in_ch, filters, stride=stride, 
                                     dropout=config.cnn_drop,
                                     maxpool=config.maxpool,
                                     avgpool=config.avgpool,
                                     batchnorm=config.batchnorm)

        self.block2 = InceptionBlock(filters * 3, filters, stride=stride,
                                     dropout=config.cnn_drop,
                                     maxpool=config.maxpool,
                                     avgpool=config.avgpool,
                                     batchnorm=config.batchnorm)

        self.block3 = InceptionBlock(filters * 3, filters, stride=stride,
                                     dropout=config.cnn_drop,
                                     maxpool=config.maxpool,
                                     avgpool=config.avgpool,
                                     batchnorm=config.batchnorm)

        # residual GRU
        self.gru = ResidualGRU(input_size=filters * 3, hidden_size=32)

        # classifier
        self.fc = nn.Linear(32, 1)

    def forward(self, **kwargs):    
        x = kwargs.get("eeg")

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        g = self.gru(x)
        out = self.fc(g)
        return out