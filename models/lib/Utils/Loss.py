import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation for binary and multi-class classification.
        
        Args:
            alpha (float or list): Balancing factor for classes.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): Reduction type: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (not probabilities) of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size).
        Returns:
            Loss value (scalar or tensor depending on reduction).
        """
        probs = F.softmax(inputs, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Standard cross-entropy loss
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of target class
        focal_weight = (1 - pt) ** self.gamma  # Compute focal weight

        if isinstance(self.alpha, (list, tuple)):
            alpha_weight = torch.tensor(self.alpha, device=inputs.device)[targets]
        else:
            alpha_weight = self.alpha

        loss = alpha_weight * focal_weight * ce_loss  # Apply weighting

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Example usage:
if __name__ == "__main__":
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]], dtype=torch.float32)  # Example logits
    targets = torch.tensor([0, 1], dtype=torch.int64)  # True labels
    loss = loss_fn(logits, targets)
    print(loss)
