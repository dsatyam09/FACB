import torch
import torch.nn as nn
import torchvision.models as models

def safe_softmax(x, dim=-1, eps=1e-6):
    """Applies a numerically stable softmax."""
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / (torch.sum(x_exp, dim=dim, keepdim=True) + eps)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y):
        """Applies cross-attention between two feature embeddings."""
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)

        attn_logits = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
        attention = safe_softmax(attn_logits)
        output = attention @ V

        return output + x  # Residual connection

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.projection(x)

class FACB(nn.Module):
    def __init__(self):
        super(FACB, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()  # Remove final classification layer

        embed_dim = 2048
        self.cross_attention = CrossAttention(embed_dim)
        self.projection_head = ProjectionHead(embed_dim, out_dim=128)

    def forward(self, x, x_fft, y, y_fft):
        x_enc = self.encoder(x)
        x_fft_enc = self.encoder(x_fft)
        y_enc = self.encoder(y)
        y_fft_enc = self.encoder(y_fft)

        # Apply cross attention
        z_x = self.cross_attention(x_enc, x_fft_enc)
        z_y = self.cross_attention(y_enc, y_fft_enc)

        # Projection head
        return self.projection_head(z_x), self.projection_head(z_y)