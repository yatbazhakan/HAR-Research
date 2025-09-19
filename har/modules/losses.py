"""
Loss Functions for HAR Models

This module provides various loss functions for human activity recognition,
including Focal Loss for handling class imbalance and other specialized losses.

Author: HAR Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Focal Loss is designed to down-weight easy examples and focus on hard examples,
    which is particularly useful for imbalanced datasets.
    
    Paper: "Focal Loss for Dense Object Detection" by Lin et al.
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float, optional): Weighting factor for rare class (default: 1.0)
        gamma (float, optional): Focusing parameter (default: 2.0)
        reduction (str, optional): Specifies the reduction to apply to the output.
            'none' | 'mean' | 'sum' (default: 'mean')
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient (default: -100)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction should be 'none', 'mean' or 'sum', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size
                and C is number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N,)
        
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Compute p_t (probability of true class)
        p_t = torch.exp(-ce_loss)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            # Get alpha values for each class
            alpha_t = self.alpha
            if isinstance(alpha_t, (list, tuple)):
                alpha_t = torch.tensor(alpha_t, device=inputs.device, dtype=inputs.dtype)
                alpha_t = alpha_t.gather(0, targets)
            elif isinstance(alpha_t, float):
                alpha_t = alpha_t
            else:
                raise ValueError(f"alpha should be float, list, or tuple, got {type(alpha_t)}")
            
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss with class-specific alpha values.
    
    This version allows different alpha values for different classes,
    which is useful when you have specific class imbalance patterns.
    
    Args:
        alpha (Union[float, list, tuple], optional): Weighting factor for each class.
            If float, same weight for all classes. If list/tuple, different weight
            for each class (default: 1.0)
        gamma (float, optional): Focusing parameter (default: 2.0)
        reduction (str, optional): Specifies the reduction to apply to the output.
            'none' | 'mean' | 'sum' (default: 'mean')
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient (default: -100)
    """
    
    def __init__(
        self,
        alpha: Union[float, list, tuple] = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction should be 'none', 'mean' or 'sum', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted Focal Loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size
                and C is number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N,)
        
        Returns:
            torch.Tensor: Computed weighted focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Compute p_t (probability of true class)
        p_t = torch.exp(-ce_loss)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
                alpha_t = alpha_t.gather(0, targets)
            elif isinstance(self.alpha, float):
                alpha_t = self.alpha
            else:
                raise ValueError(f"alpha should be float, list, or tuple, got {type(self.alpha)}")
            
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing helps prevent overfitting by preventing the model from becoming
    too confident in its predictions.
    
    Args:
        smoothing (float, optional): Label smoothing factor (default: 0.1)
        reduction (str, optional): Specifies the reduction to apply to the output.
            'none' | 'mean' | 'sum' (default: 'mean')
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient (default: -100)
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction should be 'none', 'mean' or 'sum', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Label Smoothing Cross Entropy.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size
                and C is number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N,)
        
        Returns:
            torch.Tensor: Computed label smoothing cross entropy loss
        """
        num_classes = inputs.size(-1)
        
        # Create smoothed targets
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Handle ignore index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            targets = targets * mask.long()
        
        # Create one-hot encoding
        true_dist = torch.zeros_like(log_preds)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        loss = -true_dist * log_preds
        loss = loss.sum(dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def get_loss_function(
    loss_type: str = 'cross_entropy',
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type (str): Type of loss function to create
            - 'cross_entropy': Standard Cross Entropy Loss
            - 'focal': Focal Loss
            - 'weighted_focal': Weighted Focal Loss
            - 'label_smoothing': Label Smoothing Cross Entropy
        **kwargs: Additional arguments passed to the loss function
    
    Returns:
        nn.Module: Loss function instance
    """
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'weighted_focal': WeightedFocalLoss,
        'label_smoothing': LabelSmoothingCrossEntropy,
    }
    
    if loss_type not in loss_functions:
        available = ', '.join(loss_functions.keys())
        raise ValueError(f"Unknown loss type '{loss_type}'. Available: {available}")
    
    return loss_functions[loss_type](**kwargs)


# Example usage and testing functions
def test_focal_loss():
    """Test function to verify Focal Loss implementation."""
    # Create dummy data
    batch_size, num_classes = 32, 8
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    weighted_focal = WeightedFocalLoss(alpha=[1.0] * num_classes, gamma=2.0)
    label_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Compute losses
    ce_value = ce_loss(logits, targets)
    focal_value = focal_loss(logits, targets)
    weighted_focal_value = weighted_focal(logits, targets)
    label_smooth_value = label_smooth(logits, targets)
    
    print(f"Cross Entropy Loss: {ce_value:.4f}")
    print(f"Focal Loss: {focal_value:.4f}")
    print(f"Weighted Focal Loss: {weighted_focal_value:.4f}")
    print(f"Label Smoothing Loss: {label_smooth_value:.4f}")


if __name__ == "__main__":
    test_focal_loss()
