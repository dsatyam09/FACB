import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_x, z_y):
        batch_size = z_x.shape[0]
        device = z_x.device

        # Concatenate positive pairs
        features = torch.cat([z_x, z_y], dim=0)  # Shape: (2 * batch_size, feature_dim)
        
        # Normalize features (IMPORTANT for contrastive loss)
        features = F.normalize(features, dim=1)

        # Compute similarity matrix using dot product
        similarity_matrix = torch.matmul(features, features.T)  # Shape: (2B, 2B)

        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=device)  # [0, 1, 2, ..., batch_size-1]
        labels = torch.cat([labels, labels], dim=0)  # Repeat for both views
        labels = labels.repeat(2 * batch_size, 1)  # Shape: (2B, 2B)
        labels = (labels.T == labels).float().to(device)  # Binary mask for positives

        # Mask out self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Select positives and negatives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Concatenate positives and negatives for softmax
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  # Positives at index 0

        # Apply NT-Xent loss (Softmax + Cross Entropy)
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss
