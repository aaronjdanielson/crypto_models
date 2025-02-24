import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dist


def plot_predictions(
        predictions, 
        targets=None, 
        quantiles=None, 
        target_date_ranges=None, 
        crypto_to_index=None, 
        target_scaler=None,
        forecast_mode=False):
    """
    Plot the predictions against the true targets for each cryptocurrency with date labels.

    Args:
        predictions (torch.Tensor): Model predictions of shape [num_samples, num_channels, tau, num_quantiles].
        targets (torch.Tensor, optional): Ground truth targets of shape [num_samples, num_channels, tau].
        quantiles (list, optional): List of quantile levels corresponding to the prediction intervals.
        target_date_ranges (numpy array, optional): A single 1D array of numpy.datetime64 for the x-axis.
        crypto_to_index (dict, optional): Mapping from crypto names to channel indices.
        target_scaler (object, optional): Scaler object that includes `inverse_scale_targets` for rescaling.
        forecast_mode (bool): Set to True for forecasting without targets.

    Returns:
        None. Displays the plots.
    """
    if quantiles is None:
        raise ValueError("Quantiles must be provided when plotting predictions.")

    num_channels = predictions.size(1)

    # Ensure target_date_ranges is properly formatted
    if target_date_ranges is not None:
        date_range = np.array(target_date_ranges).flatten().astype('datetime64[D]').astype(str)  # Convert to date-only format
    else:
        date_range = np.arange(predictions.size(2))  # Default to sequential indices if no dates provided

    # Reverse lookup dictionary for channel indices to crypto names
    index_to_crypto = {v: k for k, v in crypto_to_index.items()} if crypto_to_index else {}

    # Apply inverse transformation if scaler is provided
    if target_scaler is not None:
        #crypto_ids = [index_to_crypto[channel] for channel in range(num_channels)]  # Get crypto names for each channel
        predictions = target_scaler(predictions, list(crypto_to_index.keys()))
        if targets is not None:
            targets = target_scaler(targets, list(crypto_to_index.keys()))

    for channel in range(num_channels):
        plt.figure(figsize=(10, 6))

        crypto_name = index_to_crypto.get(channel, f"Crypto {channel + 1}")

        if not forecast_mode and targets is not None:
            plt.plot(date_range, targets[0, channel, :].cpu().numpy(), label="True Values", color="blue")

        for i, q in enumerate(quantiles):
            plt.plot(date_range, predictions[0, channel, :, i].cpu().numpy(), label=f"Quantile {q:.2f}", linestyle="--")

        # Extract lower and upper quantiles
        lower_quantile = predictions[0, channel, :, 0].cpu().numpy()
        upper_quantile = predictions[0, channel, :, -1].cpu().numpy()

        # Plot median prediction line
        #median_idx = len(quantiles) // 2  # Assuming symmetric quantiles
        #median_prediction = predictions[0, channel, :, median_idx].cpu().numpy()
        #plt.plot(date_range, median_prediction, label="Median Prediction", linestyle="-", color="purple")

        # Fill between lower and upper quantiles
        plt.fill_between(date_range, lower_quantile, upper_quantile, color="purple", alpha=0.1, label="Prediction Interval")

        # for i, q in enumerate(quantiles):
        #     plt.plot(date_range, predictions[0, channel, :, i].cpu().numpy(), label=f"Quantile {q:.2f}", linestyle="--")

        plt.title(f"{crypto_name}: Predictions {'(Forecast)' if forecast_mode else 'vs. True Values'}")
        plt.xlabel("Date" if target_date_ranges is not None else "Time Step")
        plt.ylabel("Original Scale Value")
        plt.xticks(rotation=45)  # Rotate dates for better visibility
        plt.legend()
        plt.show()

def compute_market_cap_to_volume_ratio(market_cap: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """
    Compute Market Cap to Volume Ratio.
    
    Args:
        market_cap (torch.Tensor): Shape [batch_size, num_cryptos]
        volume (torch.Tensor): Shape [batch_size, num_cryptos]
    
    Returns:
        torch.Tensor: Ratio of Market Cap to Trading Volume.
    """
    return market_cap / (volume + 1e-8)  # Avoid division by zero


def bollinger_bands(x: torch.Tensor, window: int = 20, k: float = 2.0) -> dict:
    """
    Compute rolling Bollinger Bands from a tensor of shape [batch_size, num_channels, lookback, num_features].
    
    Bollinger Bands are a **volatility-based** technical indicator that consists of:
    - A **Simple Moving Average (SMA)** of closing prices over a given window.
    - An **upper band** at SMA + k * standard deviation.
    - A **lower band** at SMA - k * standard deviation.

    Bollinger Bands help traders detect:
    - **Overbought conditions**: When price is near the upper band.
    - **Oversold conditions**: When price is near the lower band.
    - **Volatility Breakouts**: Widening bands signal increasing volatility.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with features ordered as:
        `['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']`.
        Shape: `[batch_size, num_channels, lookback, num_features]`.
    window : int, default=20
        The window size for the SMA and standard deviation.
    k : float, default=2.0
        The multiplier for the standard deviation.

    Returns
    -------
    bands : dict
        Dictionary containing:
        - `'sma'`: Simple Moving Average (shape `[B, C, T-window+1]`).
        - `'upper_band'`: SMA + k * std (upper Bollinger Band).
        - `'lower_band'`: SMA - k * std (lower Bollinger Band).
        - `'median_band'` (Optional): Alternative formulation using median instead of mean.
    """
    # Extract closing prices; shape: [B, C, T]
    close = x[..., 3]
    B, C, T = close.shape

    # Ensure we have enough time steps
    if T < window:
        raise ValueError(f"Time dimension (T={T}) must be at least as long as the window ({window}).")
    
    # Use unfold to create rolling windows along the time dimension.
    # Resulting shape: [B, C, num_windows, window]
    close_windows = close.unfold(dimension=-1, size=window, step=1)  # shape: [B, C, T-window+1, window]
    
    # Compute SMA (mean) over the window; shape: [B, C, T-window+1]
    sma = close_windows.mean(dim=-1)
    
    # Compute standard deviation over the window; shape: [B, C, T-window+1]
    std = close_windows.std(dim=-1, unbiased=False)

    # Compute Bollinger Bands
    upper_band = sma + k * std
    lower_band = sma - k * std

    # Alternative approach using median instead of mean (more robust to outliers)
    median_band = close_windows.median(dim=-1)[0]

    return {
        'sma': sma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'median_band': median_band  # Alternative method
    }


def statistical_bollinger_bands(x: torch.Tensor, 
                                window: int = 14, 
                                q_lower: float = 0.025, 
                                q_upper: float = 0.975) -> dict:
    """
    Compute statistical Bollinger Bands from a tensor of shape [B, C, T, F].
    
    Instead of just computing a moving average and ±K standard deviations, this
    function estimates a Normal distribution over each rolling window of the
    closing prices (feature index 3) and then computes the lower and upper bands
    as the corresponding quantiles of that distribution.
    
    Parameters:
        x (torch.Tensor): Input tensor with shape [batch_size, num_channels, T, num_features],
                          with features ordered as ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
        window (int): The window size for computing the moving average and standard deviation (default: 20).
        q_lower (float): Lower quantile (default: 0.025 for a 95% confidence interval).
        q_upper (float): Upper quantile (default: 0.975 for a 95% confidence interval).
    
    Returns:
        dict: A dictionary containing:
            - 'sma': Rolling simple moving average (shape [B, C, T-window+1]).
            - 'std': Rolling standard deviation (shape [B, C, T-window+1]).
            - 'lower_band': Lower Bollinger Band (quantile q_lower).
            - 'upper_band': Upper Bollinger Band (quantile q_upper).
            - 'distribution': A Normal distribution object with batch dimensions matching sma.
              (This can be used for further probabilistic calculations.)
    """
    # Extract closing prices; shape: [B, C, T]
    close = x[..., 3]
    B, C, T = close.shape
    if T < window:
        raise ValueError(f"Time dimension T={T} must be at least as long as the window ({window}).")
    
    # Use unfold to extract rolling windows along the time dimension.
    # close_windows will have shape [B, C, num_windows, window]
    close_windows = close.unfold(dimension=-1, size=window, step=1)
    
    # Compute the simple moving average and standard deviation for each window.
    sma = close_windows.mean(dim=-1)      # shape: [B, C, num_windows]
    std = close_windows.std(dim=-1, unbiased=False)  # shape: [B, C, num_windows]
    
    # Create a Normal distribution with the estimated parameters.
    # The distribution supports batch dimensions.
    normal_dist = dist.Normal(sma, std + 1e-8)  # add a small epsilon to avoid zero std
    
    # Compute the lower and upper bands using the inverse CDF (icdf).
    # Create scalar tensors for the quantile levels and let broadcasting do its job.
    ql_tensor = torch.tensor(q_lower, device=x.device, dtype=x.dtype)
    qu_tensor = torch.tensor(q_upper, device=x.device, dtype=x.dtype)
    
    lower_band = normal_dist.icdf(ql_tensor)
    upper_band = normal_dist.icdf(qu_tensor)
    
    return {
        'sma': sma,
        'std': std,
        'lower_band': lower_band,
        'upper_band': upper_band,
        'distribution': normal_dist
    }


def compute_obv(x: torch.Tensor) -> torch.Tensor:
    """
    Compute On Balance Volume (OBV) from a tensor of shape [B, C, T, F].
    
    OBV is computed as:
        OBV[0] = 0
        For t >= 1:
          If close[t] > close[t-1], OBV[t] = OBV[t-1] + volume[t]
          If close[t] < close[t-1], OBV[t] = OBV[t-1] - volume[t]
          Else, OBV[t] = OBV[t-1]
    
    Parameters:
        x (torch.Tensor): Input tensor with features ordered as:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
    
    Returns:
        torch.Tensor: OBV as a tensor of shape [B, C, T].
    """
    # Extract closing prices and volumes
    close = x[..., 3]   # shape: [B, C, T]
    volume = x[..., 4]  # shape: [B, C, T]
    
    B, C, T = close.shape
    
    # Compute differences in closing prices (t=1 to T-1)
    delta = close[..., 1:] - close[..., :-1]  # shape: [B, C, T-1]
    
    # Compute the sign of the price change: +1 if up, -1 if down, 0 if no change.
    # Note: torch.sign returns 0 when the input is 0.
    direction = torch.sign(delta)  # shape: [B, C, T-1]
    
    # Compute the volume contribution: for each time step (starting at t=1), 
    # contribution = sign * volume at time t.
    volume_contrib = direction * volume[..., 1:]  # shape: [B, C, T-1]
    
    # Initialize OBV with zeros for the first time step.
    obv_initial = torch.zeros(B, C, 1, device=x.device, dtype=x.dtype)
    
    # Compute the cumulative sum along the time dimension for t>=1.
    obv_rest = volume_contrib.cumsum(dim=-1)  # shape: [B, C, T-1]
    
    # Concatenate the initial OBV (0) with the cumulative sum.
    obv = torch.cat([obv_initial, obv_rest], dim=-1)  # shape: [B, C, T]
    
    return obv

def explicit_ewa(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute the Exponentially Decayed Weighted Average (EWA) along the last dimension.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [..., T] where T is the time dimension.
        lam (float): Decay factor (lambda) between 0 and 1.
    
    Returns:
        torch.Tensor: EWA computed along the last dimension.
    """
    T = x.shape[-1]
    
    device = x.device
    t_indices = torch.arange(T, device=device, dtype=x.dtype)
    # For x[..., 0]: weight = lam^(T-1); for x[..., T-1]: weight = lam^0 = 1.
    exponents = (T - 1 - t_indices).to(x.dtype)
    weights = lam ** exponents  # Shape: [T]
    
    # Reshape weights for broadcasting: add singleton dimensions for all but the last dim.
    for _ in range(x.dim() - 1):
        weights = weights.unsqueeze(0)
    
    weighted_sum = (x * weights).sum(dim=-1)
    normalization = weights.sum()
    ewa = weighted_sum / normalization
    return ewa

def explicit_ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Compute the explicit Exponential Moving Average (EMA) along the time dimension.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [..., T] where T is the time dimension.
        alpha (float): Smoothing factor between 0 and 1.
    
    Returns:
        torch.Tensor: EMA computed along the last dimension, with the same shape as x 
                      except that the time dimension is aggregated.
    """
    # Get the number of time steps (assumed to be the last dimension)
    T = x.shape[-1]
    
    # Create a vector of weights for each time step.
    # We assume x is indexed as x[..., 0], x[..., 1], ..., x[..., T-1]
    # The weight for x[..., i] is: alpha * (1 - alpha)^(T - i - 1)
    device = x.device
    t_indices = torch.arange(T, device=device, dtype=x.dtype)
    # Reverse the order: for x[..., 0] we use exponent (T-1), ..., for x[..., T-1] exponent 0.
    exponents = (T - 1 - t_indices).to(x.dtype)
    weights = alpha * (1 - alpha) ** exponents  # Shape: [T]
    
    # Reshape weights for broadcasting: [1, 1, ..., T]
    for _ in range(x.dim() - 1):
        weights = weights.unsqueeze(0)
    
    # Compute the weighted sum and normalize by the sum of weights
    weighted_sum = (x * weights).sum(dim=-1)
    normalization = weights.sum()
    ema = weighted_sum / normalization
    return ema

# def explicit_ewa(x: torch.Tensor, lam: float) -> torch.Tensor:
#     """
#     Compute the Exponentially Decayed Weighted Average (EWA) along the last dimension.
    
#     Parameters:
#         x (torch.Tensor): Input tensor of shape [..., T] where T is the time dimension.
#         lam (float): Decay factor (lambda) between 0 and 1.
    
#     Returns:
#         torch.Tensor: EWA computed along the last dimension.
#     """
#     T = x.shape[-1]
    
#     device = x.device
#     t_indices = torch.arange(T, device=device, dtype=x.dtype)
#     # For x[..., 0] weight = lam^(T-1), for x[..., T-1] weight = lam^0 = 1.
#     exponents = (T - 1 - t_indices).to(x.dtype)
#     weights = lam ** exponents  # Shape: [T]
    
#     # Reshape weights for broadcasting: [1, 1, ..., T]
#     for _ in range(x.dim() - 1):
#         weights = weights.unsqueeze(0)
    
#     weighted_sum = (x * weights).sum(dim=-1)
#     normalization = weights.sum()
#     ewa = weighted_sum / normalization
#     return ewa

def rolling_ewa(x: torch.Tensor, lam: float, window: int) -> torch.Tensor:
    """
    Compute a rolling exponentially decayed weighted average over the last dimension.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [..., T].
        lam (float): Decay factor between 0 and 1.
        window (int): Window size over which to compute the EWA.
    
    Returns:
        torch.Tensor: Rolling EWA, shape: [..., T - window + 1].
    """
    # Use unfold to extract rolling windows along the time dimension.
    # x_unfold will have shape: [..., T - window + 1, window]
    x_unfold = x.unfold(dimension=-1, size=window, step=1)
    # Merge all dimensions except the last (window) for vectorized processing.
    original_shape = x_unfold.shape[:-1]  # shape: [..., num_windows]
    x_windows = x_unfold.reshape(-1, window)  # shape: [N, window]
    
    # Apply explicit_ewa on each window.
    ewa_vals = explicit_ewa(x_windows, lam)  # shape: [N]
    # Reshape back to the original shape (without the window dimension)
    result = ewa_vals.reshape(original_shape)
    return result

def compute_macd_rolling(x: torch.Tensor, 
                         lam_fast: float, 
                         lam_slow: float, 
                         lam_signal: float,
                         window_slow: int = 26, 
                         window_signal: int = 9) -> dict:
    """
    Compute a rolling MACD indicator.
    
    For each time step (starting when there are at least `window_slow` observations),
    we compute:
    
        Fast EMA (over the last `window_slow` time steps, using lam_fast)
        Slow EMA (over the last `window_slow` time steps, using lam_slow)
        MACD_line = Fast EMA - Slow EMA
    
    Then we compute a rolling Signal line on the MACD_line using a window of `window_signal`
    and decay factor lam_signal. Finally, the histogram is defined as:
    
        Histogram = (Aligned MACD_line) - Signal_line.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [B, C, T, F], with features ordered as
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
        lam_fast (float): Decay factor for the fast EMA (e.g., for a 12-period EMA, you might use 1 - 2/(12+1)).
        lam_slow (float): Decay factor for the slow EMA (e.g., for a 26-period EMA, use 1 - 2/(26+1)).
        lam_signal (float): Decay factor for the signal line EMA (e.g., for a 9-period EMA).
        window_slow (int): Window size for slow EMA (default is 26).
        window_signal (int): Window size for computing the signal line (default is 9).
    
    Returns:
        dict: Contains three keys:
            'macd_line': Tensor of shape [B, C, T - window_slow + 1]
            'signal_line': Tensor of shape [B, C, (T - window_slow + 1) - window_signal + 1]
            'histogram': Tensor of the same shape as 'signal_line'
    """
    # Extract closing prices: shape [B, C, T]
    close = x[..., 3]
    
    # Compute fast and slow EMAs on a rolling window of length window_slow.
    # We use the same window length for both so that the outputs are aligned.
    fast_ema = rolling_ewa(close, lam_fast, window_slow)  # shape: [B, C, T - window_slow + 1]
    slow_ema = rolling_ewa(close, lam_slow, window_slow)  # shape: [B, C, T - window_slow + 1]
    
    macd_line = fast_ema - slow_ema  # shape: [B, C, T - window_slow + 1]
    
    # Now compute the signal line by applying a rolling EWA on the MACD_line
    signal_line = rolling_ewa(macd_line, lam_signal, window_signal)  
    # signal_line shape: [B, C, (T - window_slow + 1) - window_signal + 1]
    
    # To align the MACD_line with the signal line, we take the last corresponding values.
    macd_line_aligned = macd_line[..., window_signal - 1:]
    
    histogram = macd_line_aligned - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

def compute_macd(x: torch.Tensor, lam_fast: float, lam_slow: float, lam_signal: float) -> dict:
    """
    Compute the MACD indicator over the lookback dimension using explicit exponentially decayed weighted averages.
    
    The MACD is defined as:
        MACD line = EMA_fast - EMA_slow
        Signal line = EMA of MACD line (using lam_signal)
        Histogram = MACD line - Signal line
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [batch_size, num_channels, lookback, num_features].
                          The features are assumed to be ordered as:
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
        lam_fast (float): Decay factor for the fast EMA (e.g. corresponding to a 12-period EMA).
        lam_slow (float): Decay factor for the slow EMA (e.g. corresponding to a 26-period EMA).
        lam_signal (float): Decay factor for the signal line EMA (e.g. corresponding to a 9-period EMA).
    
    Returns:
        dict: A dictionary containing:
            - 'macd_line': The difference between the fast and slow EMAs.
            - 'signal_line': The EWA of the MACD line using lam_signal.
            - 'histogram': The difference between the MACD line and the signal line.
        Each output tensor is of shape [batch_size, num_channels].
    """
    # Extract closing prices: shape [B, C, T]
    close = x[..., 3]
    
    # Compute fast and slow EMAs over the entire lookback
    ema_fast = explicit_ewa(close, lam_fast)  # shape: [B, C]
    ema_slow = explicit_ewa(close, lam_slow)  # shape: [B, C]
    
    macd_line = ema_fast - ema_slow  # shape: [B, C]
    
    # To compute the signal line from the MACD line, we need a time dimension.
    # Here we mimic the idea by assuming that the MACD value over the lookback is our series,
    # so we “simulate” a one-element time series. In practice, for a time series MACD, one would 
    # compute the EMAs recursively at each time step.
    # For demonstration purposes, we'll treat macd_line as a single value per series:
    signal_line = macd_line  # With a single point, the EWA is the value itself.
    
    # Therefore the histogram (MACD line minus signal line) will be zero.
    histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }


def compute_rolling_rsi(x: torch.Tensor, period: int = 14) -> torch.Tensor:
    """
    Compute the Relative Strength Index (RSI) over rolling windows in the data.
    
    The Relative Strength Index (RSI) is a momentum oscillator that measures 
    the magnitude of recent price changes to evaluate overbought or oversold conditions.
    
    RSI is calculated using the formula:
    
        RSI = 100 - (100 / (1 + RS))
    
    where RS (Relative Strength) is defined as:
    
        RS = (Average Gain over period) / (Average Loss over period)
    
    The RSI is bounded between 0 and 100:
    - RSI > 70: Asset is considered overbought.
    - RSI < 30: Asset is considered oversold.
    
    This function computes RSI over **rolling windows** along the lookback dimension, 
    ensuring that we capture RSI dynamics over time instead of just a single fixed period.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape `[batch_size, num_channels, lookback, num_features]`.
        The features are assumed to be ordered as:
        `['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']`.
    period : int, default=14
        The period over which to compute RSI (Wilder’s recommended period is 14).
    
    Returns
    -------
    rsi : torch.Tensor
        The computed rolling RSI for each series. Shape: `[batch_size, num_channels, lookback - period + 1]`.
    """
    # Extract the closing prices; shape: [B, C, L]
    close = x[..., 3]

    # Compute differences between consecutive closing prices; shape: [B, C, L-1]
    delta = close[..., 1:] - close[..., :-1]
    
    # Compute gains (U) and losses (D)
    U = torch.clamp(delta, min=0)  # Positive differences (or 0)
    D = torch.clamp(-delta, min=0) # Negative differences, made positive

    # Use torch.unfold to create rolling windows over the last dimension
    # shape after unfold: [B, C, num_windows, period]
    U_windows = U.unfold(-1, period, 1)
    D_windows = D.unfold(-1, period, 1)

    # Compute Smoothed Moving Average (SMMA) for gains and losses
    # SMMA = (previous_average * (period - 1) + current_value) / period
    smma_U = U_windows.mean(dim=-1)  # shape: [B, C, num_windows]
    smma_D = D_windows.mean(dim=-1)  # shape: [B, C, num_windows]

    # Compute Relative Strength (RS)
    eps = 1e-8  # Small constant to avoid division by zero
    RS = smma_U / (smma_D + eps)

    # Compute RSI
    rsi = 100 - (100 / (1 + RS))

    return rsi



def compute_rsi(x: torch.Tensor, period: int = 14) -> torch.Tensor:
    """
    Compute the Relative Strength Index (RSI) over the lookback dimension.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape [batch_size, num_channels, lookback, num_features].
        The features are assumed to be ordered as:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
    period : int, default=14
        The period over which to compute RSI (Wilder’s recommended period is 14).
    
    Returns
    -------
    rsi : torch.Tensor
        The computed RSI for each series. Shape: [batch_size, num_channels].
    """
    # Extract the closing prices; shape: [B, C, L]
    close = x[..., 3]

    # Compute differences between consecutive closing prices; shape: [B, C, L-1]
    delta = close[..., 1:] - close[..., :-1]
    
    # Compute U (upward changes) and D (downward changes)
    U = torch.clamp(delta, min=0)         # positive differences (or 0)
    D = torch.clamp(-delta, min=0)          # negative differences, made positive

    # We'll compute a weighted average (SMMA) for U and D using exponential weights
    # based on the explicit formula:
    # SMMA = sum_{i=0}^{T-1} [α * (1-α)^(T-1-i) * value_i] / sum_{i=0}^{T-1} [α * (1-α)^(T-1-i)]
    # Here T = number of differences = L-1, and α = 1/period.
    T = U.shape[-1]  # number of differences
    alpha = 1.0 / period

    # Create weights: the most recent difference gets weight (1-α)^0 = 1.
    # We'll assume index 0 corresponds to the oldest difference and index T-1 is the most recent.
    # So weight[i] = α * (1-α)^(T-1 - i)
    device = x.device
    weights = alpha * (1 - alpha) ** torch.arange(T - 1, -1, -1, device=device, dtype=x.dtype)
    # Reshape weights for broadcasting: [1, 1, T]
    weights = weights.view(1, 1, T)
    
    # Compute the SMMA for U and D
    # Multiply each difference by its weight and then sum over time
    smma_U = (U * weights).sum(dim=-1) / weights.sum()
    smma_D = (D * weights).sum(dim=-1) / weights.sum()
    
    # Compute Relative Strength (RS)
    # To avoid division by zero, we add a small epsilon
    eps = 1e-8
    RS = smma_U / (smma_D + eps)
    
    # Compute RSI: RSI = 100 - 100/(1+RS)
    rsi = 100 - 100 / (1 + RS)
    
    return rsi

def compute_market_cap_adjusted_rsi(close_prices, market_caps, period=14):
    """
    Compute Market Cap Adjusted RSI.
    
    Args:
        close_prices (torch.Tensor): Shape [batch_size, num_cryptos, lookback]
        market_caps (torch.Tensor): Shape [batch_size, num_cryptos]
        period (int): Lookback period for RSI.
    
    Returns:
        torch.Tensor: Adjusted RSI per crypto.
    """
    # Compute price differences
    delta = close_prices[..., 1:] - close_prices[..., :-1]
    
    # Compute U (up) and D (down) movements
    U = torch.clamp(delta, min=0)
    D = torch.clamp(-delta, min=0)
    
    # Weight up/down movements by market cap
    market_cap_weights = market_caps / market_caps.sum(dim=-1, keepdim=True)
    U_weighted = U * market_cap_weights.unsqueeze(-1)
    D_weighted = D * market_cap_weights.unsqueeze(-1)

    # Compute moving average (exponential smoothing)
    alpha = 1.0 / period
    smma_U = (U_weighted * alpha).sum(dim=-1)
    smma_D = (D_weighted * alpha).sum(dim=-1)

    # Compute Relative Strength Index (RSI)
    RS = smma_U / (smma_D + 1e-8)  # Avoid division by zero
    rsi = 100 - (100 / (1 + RS))

    return rsi


# def compute_stochastic_oscillator(x: torch.Tensor, period: int = 14) -> torch.Tensor:
#     """
#     Compute the Stochastic Oscillator over the lookback dimension.
    
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor with shape [batch_size, num_channels, lookback, num_features].
#         The features are assumed to be ordered as:
#         ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
#     period : int, default=14
#         The period over which to compute the Stochastic Oscillator.
    
#     Returns
#     -------
#     stochastic_oscillator : torch.Tensor
#         The computed Stochastic Oscillator for each series. Shape: [batch_size, num_channels].
#     """
#     # Extract high and low prices; shape: [B, C, L]
#     high = x[..., 1]
#     low = x[..., 2]
    
#     # Compute the highest high and lowest low over the lookback period
#     highest_high = high.unfold(dimension=-1, size=period, step=1).max(dim=-1).values
#     lowest_low = low.unfold(dimension=-1, size=period, step=1).min(dim=-1).values
    
#     # Compute the Stochastic Oscillator
#     stochastic_oscillator = 100 * (x[..., 3] - lowest_low) / (highest_high - lowest_low)
    
#     return stochastic_oscillator


def compute_stochastic_oscillator(x: torch.Tensor, period: int = 14) -> dict:
    """
    Compute the Stochastic Oscillator (%K and %D) over rolling windows.
    
    The Stochastic Oscillator measures the position of the closing price relative 
    to its recent high-low range, helping identify overbought and oversold conditions.
    
    It is calculated as:
    
        %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        %D = Moving Average of %K
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape `[batch_size, num_channels, lookback, num_features]`.
        The features are assumed to be ordered as:
        `['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']`.
    period : int, default=14
        The period over which to compute the oscillator.
    
    Returns
    -------
    stochastic : dict
        Dictionary containing:
        - `'percent_k'` (Tensor): %K line, shape `[batch_size, num_channels, lookback - period + 1]`
        - `'percent_d'` (Tensor): 3-period moving average of %K
    """
    high = x[..., 1]  # High prices
    low = x[..., 2]   # Low prices
    close = x[..., 3] # Closing prices

    # Compute rolling min and max
    lowest_low = low.unfold(-1, period, 1).min(dim=-1)[0]
    highest_high = high.unfold(-1, period, 1).max(dim=-1)[0]

    # Compute %K (Stochastic Oscillator)
    percent_k = 100 * (close[..., period-1:] - lowest_low) / (highest_high - lowest_low + 1e-8)

    # Compute %D (3-period SMA of %K)
    percent_d = percent_k.unfold(-1, 3, 1).mean(dim=-1)

    return {'percent_k': percent_k, 'percent_d': percent_d}

def stochastic_oscillator(x: torch.Tensor, window: int = 14, d_period: int = 3) -> dict:
    """
    Compute the stochastic oscillator for a given input tensor.
    
    The oscillator is defined as:
        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %D = SMA(%K) over d_period windows (optional)
    
    Parameters:
        x (torch.Tensor): Input tensor with shape [B, C, T, F] where features are ordered as:
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
        window (int): Lookback period to compute the highest high and lowest low (default: 14).
        d_period (int): Window size for computing the moving average of %K (default: 3).
    
    Returns:
        dict: A dictionary containing:
            - 'percent_k': The %K oscillator, shape [B, C, T - window + 1].
            - 'percent_d': The %D oscillator (moving average of %K), shape [B, C, T - window - d_period + 2].
    """
    # Extract the required prices
    high = x[..., 1]   # shape: [B, C, T]
    low = x[..., 2]    # shape: [B, C, T]
    close = x[..., 3]  # shape: [B, C, T]
    
    B, C, T = close.shape
    if T < window:
        raise ValueError(f"Time dimension T={T} must be at least as long as the window ({window}).")
    
    # Use unfold to extract rolling windows for high, low, and close
    # The resulting shape will be [B, C, num_windows, window] where num_windows = T - window + 1.
    high_windows = high.unfold(dimension=-1, size=window, step=1)  # shape: [B, C, T-window+1, window]
    low_windows = low.unfold(dimension=-1, size=window, step=1)    # shape: [B, C, T-window+1, window]
    close_windows = close.unfold(dimension=-1, size=window, step=1)  # shape: [B, C, T-window+1, window]
    
    # Compute the highest high and lowest low for each window.
    highest_high, _ = high_windows.max(dim=-1)  # shape: [B, C, T-window+1]
    lowest_low, _ = low_windows.min(dim=-1)       # shape: [B, C, T-window+1]
    
    # For each window, we want the closing price corresponding to the last time step in that window.
    # This can be obtained by taking the last element of each unfolded window.
    close_last = close_windows[..., -1]  # shape: [B, C, T-window+1]
    
    # Compute %K: avoid division by zero by adding a small epsilon.
    eps = 1e-8
    percent_k = 100 * (close_last - lowest_low) / (highest_high - lowest_low + eps)  # shape: [B, C, T-window+1]
    
    # Optionally compute %D as the simple moving average of %K over d_period windows.
    # We'll use unfold again on the %K time dimension.
    if percent_k.shape[-1] < d_period:
        raise ValueError(f"Not enough %K values ({percent_k.shape[-1]}) to compute a moving average of period {d_period}.")
    
    percent_k_windows = percent_k.unfold(dimension=-1, size=d_period, step=1)  # shape: [B, C, (T-window+1)-d_period+1, d_period]
    percent_d = percent_k_windows.mean(dim=-1)  # shape: [B, C, T-window-d_period+2]
    
    return {
        'percent_k': percent_k,
        'percent_d': percent_d
    }


def compute_moving_average(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    """
    Compute the simple moving average over the lookback dimension.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape [batch_size, num_channels, lookback, num_features].
        The features are assumed to be ordered as:
        ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'].
    window : int, default=20
        The window size over which to compute the moving average.
    
    Returns
    -------
    moving_average : torch.Tensor
        The computed moving average for each series. Shape: [batch_size, num_channels].
    """
    # Extract closing prices; shape: [B, C, L]
    close = x[..., 3]
    
    # Compute the simple moving average using a rolling window
    moving_average = close.unfold(dimension=-1, size=window, step=1).mean(dim=-1)
    
    return moving_average



# Example usage:
if __name__ == "__main__":

     # Assume batch_size=2, num_channels=3, lookback=60, num_features=6
    batch_size, num_channels, lookback, num_features = 64, 13, 56, 6

    # Create dummy data for demonstration (simulated price data)
    x = torch.rand(batch_size, num_channels, lookback, num_features)
    
    # Compute Bollinger Bands with a 20-day window and multiplier 2.
    bb = bollinger_bands(x, window=14, k=2.0)
    print("Bollinger Bands SMA shape:", bb['sma'].shape)
    print("Upper Band shape:", bb['upper_band'].shape)
    print("Lower Band shape:", bb['lower_band'].shape)
    
    # Compute OBV.
    obv = compute_obv(x)
    print("OBV shape:", obv.shape)
    
    # Choose decay factors corresponding roughly to standard EMA periods.
    # A common EMA smoothing factor is: α = 2/(N+1) so that lam = 1 - α.
    # For a 12-period EMA: α ~ 2/13 ≈ 0.1538 so lam_fast ≈ 0.8462.
    # For a 26-period EMA: α ~ 2/27 ≈ 0.0741 so lam_slow ≈ 0.9259.
    # For a 9-period EMA on the MACD: α ~ 2/10 = 0.2 so lam_signal ≈ 0.8.
    lam_fast = 0.8462
    lam_slow = 0.9259
    lam_signal = 0.8
    
    macd_dict = compute_macd_rolling(x, lam_fast, lam_slow, lam_signal,
                                     window_slow=26, window_signal=9)
    
    print("MACD Line shape:", macd_dict['macd_line'].shape)
    print("Signal Line shape:", macd_dict['signal_line'].shape)
    print("Histogram shape:", macd_dict['histogram'].shape)

    # Compute Rolling RSI
    rsi = compute_rolling_rsi(x, period=14)
    print("Rolling RSI shape:", rsi.shape)

    # Compute Stochastic Oscillator
    stochastic = compute_stochastic_oscillator(x, period=14)
    print("Stochastic %K shape:", stochastic['percent_k'].shape)
    print("Stochastic %D shape:", stochastic['percent_d'].shape)


    

    print(1640/60)