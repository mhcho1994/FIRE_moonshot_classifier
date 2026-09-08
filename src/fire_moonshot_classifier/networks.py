"""Shared neural layers; checkpoint keys are kept compatible with training."""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalAttentionPooling(nn.Module):
    """
    Learnable attention pooling over the temporal dimension.
    Input: (B, C, T) → Output: (B, C)
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)

        out = (x * attn_weights).sum(dim=1)  # (B, C)
        
        return out, attn_weights
    
class FlightFeaturizer(nn.Module):
    """
    1D-CNN on fixed (B, N_FEAT, WIN_LEN) input.
      Conv1(N_FEAT→32, k=7) + MaxPool(2) → (B, 32, 50)
      Conv2(32→64, k=5) + MaxPool(2) → (B, 64, 25)
      Conv3(64→128,k=3)              → (B, 128, 25)
      AdaptiveAvgPool1d(1)           → (B, 128)
    """
    def __init__(self, n_feat=3, cnn_ch=128, attn_hidden=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, cnn_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cnn_ch), nn.ReLU(),
        )
        self.attn_pool = TemporalAttentionPooling(cnn_ch, attn_hidden)
        self.in_features = cnn_ch

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))  # (B, CNN_CH, L')
        h, _ = self.attn_pool(x)                      # Attention Applied Pooling to get (B, CNN_CH)
        return h                                      # (B, CNN_CH)

    def forward_features(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        h, attn_weights = self.attn_pool(f3)          
        return f1, f2, h


class FeatBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.fc(x))


class PrototypeClassifier(nn.Module):
    """
    Distance-based classifier: logit = -dist(x, prototype_k).
    Low max-logit (= large min-distance) → OOD signal.
    """
    def __init__(self, n_classes, in_dim=32):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, in_dim))

    def forward(self, x):
        dist = torch.cdist(x, self.prototypes)  # (B, n_classes)
        return -dist

