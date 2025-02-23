from typing import Tuple
import torch
from typing import List

## Quantile Loss Functions

def compute_pinball_loss(outputs: torch.Tensor, targets: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Compute the pinball loss for quantile regression.

    Args:
        outputs (torch.Tensor): Model predictions of shape [batch_size, num_channels, horizon, num_quantiles].
        targets (torch.Tensor): Ground truth targets of shape [batch_size, num_channels, horizon].
        quantiles (torch.Tensor): Quantiles to compute the loss for, shape [num_quantiles].

    Returns:
        torch.Tensor: Scalar tensor representing the total pinball loss across all samples.
    """
    # Ensure targets have the correct shape for broadcasting
    targets = targets.unsqueeze(-1)  # [batch_size, num_channels, horizon, 1]
    
    # Calculate the quantile errors
    errors = targets - outputs  # [batch_size, num_channels, horizon, num_quantiles]
    
    # Compute the pinball loss for each quantile
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)  # [batch_size, num_channels, horizon, num_quantiles]
    
    # Average over all dimensions
    return loss.mean()

def compute_q_risk(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    quantiles: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Q-risk for quantile regression.

    Q-risk is a scaled metric that quantifies how well the model performs for each quantile.
    It is computed as 2 * (quantile_loss / target_absolute_sum), where target_absolute_sum
    is the sum of the absolute values of the targets.

    Args:
        outputs (torch.Tensor): Model predictions of shape [batch_size, num_channels, tau, num_quantiles].
        targets (torch.Tensor): Ground truth targets of shape [batch_size, num_channels, tau].
        quantiles (torch.Tensor): Tensor of quantiles, shape [num_quantiles].

    Returns:
        torch.Tensor: A tensor of shape [num_quantiles] representing the Q-risk for each quantile.
    """
    # Ensure targets are broadcastable with outputs
    targets = targets.unsqueeze(-1)  # [batch_size, num_channels, tau, 1]

    # Compute the errors
    errors = targets - outputs  # [batch_size, num_channels, tau, num_quantiles]

    # Compute quantile losses
    quantile_losses = torch.max(quantiles * errors, (quantiles - 1) * errors)  # [batch_size, num_channels, tau, num_quantiles]
    total_quantile_loss = quantile_losses.sum(dim=(0, 1, 2))  # Sum over batch, channels, and horizons

    # Compute the absolute sum of targets
    target_absolute_sum = targets.abs().sum()  # Scalar

    # Avoid division by zero
    if target_absolute_sum == 0:
        # Return zero Q-risk if no valid target values
        return torch.zeros_like(quantiles, device=quantiles.device)

    # Compute Q-risk for each quantile
    q_risk = 2 * (total_quantile_loss / target_absolute_sum)  # [num_quantiles]

    return q_risk


def compute_coverage_percentage(outputs: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> float:
    """
    Compute the percentage of target values that fall within the range defined by the high and low quantile predictions.

    Args:
        outputs (torch.Tensor): Model predictions of shape [batch_size, num_channels, horizon, num_quantiles].
        targets (torch.Tensor): Ground truth targets of shape [batch_size, num_channels, horizon].
        quantiles (List[float]): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        float: The percentage of target values within the high-low quantile range.
    """
    # Get the indices of the low and high quantiles
    low_idx, high_idx = 0, len(quantiles) - 1  # Assuming quantiles are sorted [low, median, high]

    # Extract the high and low quantile predictions
    low_quantile_preds = outputs[..., low_idx]  # Shape: [batch_size, num_channels, horizon]
    high_quantile_preds = outputs[..., high_idx]  # Shape: [batch_size, num_channels, horizon]

    # Compute boolean mask where target falls within (low, high) range
    within_bounds = (targets >= low_quantile_preds) & (targets <= high_quantile_preds)

    # Compute percentage
    coverage_percentage = within_bounds.float().mean().item() * 100  # Convert to percentage

    return coverage_percentage
