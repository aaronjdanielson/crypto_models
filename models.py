import torch
import copy
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
#from modules import ExpDecayLinearRegressionSlope, ChannelTimeFusion, DynamicHorizonScaler


from dataclasses import dataclass
from typing import List

@dataclass
class DataProps:
    num_historical_features: int
    num_time_features: int
    num_channels: int
    lookback: int
    horizon: int

@dataclass
class ModelConfig:
    output_quantiles: List[float]

@dataclass
class Config:
    data_props: DataProps
    model: ModelConfig


class BaselineModel(nn.Module):
    """
    BaselineModel for data from CryptoDataset.

    Parameters:
    ----------
    config: dict
        Configuration dictionary with the model and data properties.
    
    Returns: Dict
        Dictionary with the predicted quantiles, historical weights, future weights, and attention scores.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # Data properties
        self.num_historical_features = config.data_props.num_historical_features
        self.num_time_features = config.data_props.num_time_features
        self.num_channels = config.data_props.num_channels
        self.lookback = config.data_props.lookback
        self.horizon = config.data_props.horizon

        # Model properties
        self.output_quantiles = config.model.output_quantiles
        self.num_outputs = len(self.output_quantiles)

        # Learnable logits for price mixture (ensuring sum-to-1 constraint via softmax)
        self.price_mixture_logits = nn.Parameter(torch.randn(self.num_channels, self.horizon, self.num_outputs, 4))
        #self.price_mixture_logits = nn.Parameter(torch.randn(self.num_channels, self.num_outputs, 4))

           
    def forward(self, batch):
        """
        Forward pass using historical and future data from the batch.
        """
        historical_data = batch['historical_data']  # Shape: [batch_size, num_channel, lookback, num_historical_features]
        # time_features = batch['time_features']  # Shape: [batch_size, lookback + horizon, num_future_features]
        # close_prices = historical_data[:, :, :, 3]
        # high_prices = historical_data[:, :, :, 1]
        # low_prices = historical_data[:, :, :, 2]
        # open_prices = historical_data[:, :, :, 0]

        # **Step 2: Extract Volume & Market Cap Features (Full Lookback)**
        # volume_features = historical_data[:, :, :, 4]
        # market_cap_features = historical_data[:, :, :, 5]

        ## get batch_size
        batch_size = historical_data.size(0)
        ## get num_channels
        # num_channels = historical_data.size(1)
        # ## get lookback
        # lookback = historical_data.size(2)
        
        
        # **Extract Key Price Signals**
        # Compute overall mean price including Open, High, Low, and Close
        overall_price_mean = historical_data[:, :, :, :4].mean(dim=3).mean(dim=2, keepdim=True)
        overall_price_mean = overall_price_mean.unsqueeze(3).expand(-1, -1, self.horizon, self.num_outputs)
        close_price_means = historical_data[:, :, :, 3].mean(dim=2, keepdim=True).unsqueeze(3).expand(-1, -1, self.horizon, self.num_outputs)
        most_recent_prices = historical_data[:, :, -1, 3].unsqueeze(2).unsqueeze(3).expand(-1, -1, self.horizon, self.num_outputs)
        raw_high_prices = historical_data[:, :, :, 1].max(dim=2).values.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.horizon, self.num_outputs)
        raw_low_prices = historical_data[:, :, :, 2].min(dim=2).values.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.horizon, self.num_outputs)
        ## method 3:
        # **Define Learnable Logits**
        logits = self.price_mixture_logits  # [num_channels, num_outputs, horizon, 4]

        # **Apply Softmax to Get Simplex-Consistent Weights**
        weights = torch.softmax(logits, dim=-1)  # [num_channels, num_outputs, 4] - 4 components
        print(f"weights: {weights.size()}")

        # **Split Weights into Components and Expand**
        # w_close, w_recent, w_high, w_low = weights.split(1, dim=-1)  # Keeps tensor structure
        # w_close = w_close.squeeze(-1).unsqueeze(0).unsqueeze(2).expand(-1, -1, self.horizon, -1)
        # w_recent = w_recent.squeeze(-1).unsqueeze(0).unsqueeze(2).expand(-1, -1, self.horizon, -1)
        # w_high = w_high.squeeze(-1).unsqueeze(0).unsqueeze(2).expand(-1, -1, self.horizon, -1)
        # w_low = w_low.squeeze(-1).unsqueeze(0).unsqueeze(2).expand(-1, -1, self.horizon, -1)

        weights = weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_channels, num_outputs, horizon, 4]
        w_close, w_recent, w_high, w_low = weights.split(1, dim=-1)  # Still [batch_size, num_channels, num_outputs, horizon, 1]

        # Remove last dimension without affecting batch structure
        w_close = w_close.squeeze(-1)
        w_recent = w_recent.squeeze(-1)
        w_high = w_high.squeeze(-1)
        w_low = w_low.squeeze(-1)

        print(w_close.size())
        print(close_price_means.size())


        # **Compute Final Weighted Forecast**
        weighted_prices = (
            w_close * close_price_means +
            w_recent * most_recent_prices +
            w_high * raw_high_prices +
            w_low * raw_low_prices
        )
          


        return weighted_prices

 