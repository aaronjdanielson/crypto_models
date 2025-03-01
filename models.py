import torch
import copy
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional, Any
from functions import compute_volume_change, compute_rolling_cmf, compute_rolling_vwma, compute_rolling_mfi, statistical_bollinger_bands, compute_obv, compute_rolling_rsi, compute_stochastic_oscillator
from modules import MLPBlock
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

            num_historical_features: int
                Number of historical features per channel.
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

            num_time_features: int
                Number of time features per time step.
                ['day_of_week', 'month', 'day_of_month', 'quarter', 'year_since_earliest_date']

            num_channels: int
                Number of channels in the dataset.

            lookback: int
                Number of historical time steps to consider.

            horizon: int
                Number of future time steps to predict.

            output_quantiles: List[float]
                List of quantiles to predict.

    Returns: Tensor
        Tensor with the predicted quantiles.
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

        ## get batch_size
        batch_size = historical_data.size(0)
        
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
        logits = self.price_mixture_logits  # [num_channels, horizon, num_outputs, 4]

        # **Apply Softmax to Get Simplex-Consistent Weights**
        weights = torch.softmax(logits, dim=-1)  # [num_channels, num_outputs, 4] - 4 components
        print(f"weights: {weights.size()}")

        weights = weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_channels, horizon, num_outputs, 4]
        w_close, w_recent, w_high, w_low = weights.split(1, dim=-1)  

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



class CommonMetricsModel(nn.Module):
    """
    CommonMetricsModel for data from CryptoDataset.

    Parameters:
    ----------
    config: dict
        Configuration dictionary with the model and data properties.

            num_historical_features: int
                Number of historical features per channel.
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

            num_time_features: int
                Number of time features per time step.
                ['day_of_week', 'month', 'day_of_month', 'quarter', 'year_since_earliest_date']

            num_channels: int
                Number of channels in the dataset.

            lookback: int
                Number of historical time steps to consider.

            horizon: int
                Number of future time steps to predict.

            output_quantiles: List[float]
                List of quantiles to predict.

    Returns: Tensor
        Tensor with the predicted quantiles.
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
        self.window = self.horizon

        # Model properties
        self.output_quantiles = config.model.output_quantiles
        self.num_outputs = len(self.output_quantiles)

        # Multi-Layer Perceptron (MLP) for Bollinger Bands
        self.bollinger_bands_mlp = MLPBlock(
            input_size= 2*(7) + 16, output_size= 4 * self.num_outputs, num_layers=3, hidden_size=32, dropout=0.1
        )

        ## create channel embeddings
        self.channel_embeddings = nn.Parameter(torch.randn(self.num_channels, 16))

        ## create a parameter for the quantile adjustment
        self.alpha = nn.Parameter(torch.randn(1, 1, 1, 1))

        # Multi-Layer Perceptron (MLP) for On-Balance Volume (OBV)
        # self.obv_mlp = MLPBlock(
        #     input_size= self.lookback, output_size= 4 * self.num_outputs, num_layers=3, hidden_size=64, dropout=0.1
        # )

        # Multi-Layer Perceptron (MLP) for Relative Strength Index (RSI)
        self.rsi_mlp = MLPBlock(
            input_size= self.lookback, output_size= 4 * self.horizon, num_layers=3, hidden_size=64, dropout=0.1
        )

        # self.volume_change_mlp = MLPBlock(
        #     input_size= self.lookback -5, output_size= self.num_outputs * self.horizon, num_layers=3, hidden_size=64, dropout=0.1
        # )

        self.quantile_adjustment_embeddings = MLPBlock(
            input_size= 7*5 + 16, output_size= self.num_outputs * self.horizon, num_layers=3, hidden_size=32, dropout=0.1
        )
        #nn.Parameter(torch.randn(self.num_channels, self.horizon, self.num_outputs))


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
        volume_features = historical_data[:, :, :, 4]
        market_cap_features = historical_data[:, :, :, 5]

        #print(f"volume_features: {volume_features}")
        #print(f"market_cap_features: {market_cap_features}")
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

        channel_embeddings = self.channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        print(f"channel_embeddings: {channel_embeddings.shape}")

        # **Define Common Metrics**
        # Compute Bollinger Bands
        bollinger_bands = statistical_bollinger_bands(historical_data, window=14)
        #print(f"bollinger_bands: {bollinger_bands['sma'].shape}")
        #print(f"bollinger_bands: {bollinger_bands['std'][:, :, -7:].shape}")
        x_bollinger = torch.cat((bollinger_bands['sma'][:, :, -7:], bollinger_bands['std'][:, :, -7:],channel_embeddings), dim=2)
        print(f"x_bollinger: {x_bollinger.shape}")
        #bollinger_embeddings = self.bollinger_bands_mlp(bollinger_bands['std'])
        #print(f"bollinger_embeddings: {bollinger_embeddings.shape}")
        bollinger_embeddings = self.bollinger_bands_mlp(x_bollinger)
        #print(f"bollinger_embeddings: {bollinger_embeddings.shape}")
        ## reshape bollinger_embeddings to be [batch_size, num_channels, horizon, 4]
        bollinger_embeddings = bollinger_embeddings.view(batch_size, self.num_channels, self.num_outputs, 4)
        #print(f"bollinger_embeddings: {bollinger_embeddings.shape}")
        #expanding bollinger_embeddings to match the shape of logits
        bollinger_embeddings = bollinger_embeddings.unsqueeze(2).expand(-1, -1, self.horizon, -1, -1)
        
        # Compute On-Balance Volume (OBV)
        obv = compute_obv(historical_data)
        print(f"obv: {obv.shape}")
        # get last 7 values of obv
        obv = obv[:, :, -7:]

        # Compute On-Balance Volume (OBV)
        rsi = compute_rolling_rsi(historical_data)
        print(f"rsi: {rsi.shape}")
        # get last 7 values of rsi
        rsi = rsi[:, :, -7:]

        stochastic_oscillator =  compute_stochastic_oscillator(historical_data)
        print(f"stochastic_oscillator: {stochastic_oscillator['percent_d'].shape}")
        ## get last 7 values of percent_d
        stochastic_oscillator = stochastic_oscillator['percent_d'][:, :, -7:]

        #vwma = compute_rolling_vwma(historical_data)
        #print(f"vwma: {vwma.shape}")

        #rsi_embeddings = self.rsi_mlp(torch.cat((obv), dim=2))

        # rsi_embeddings = self.rsi_mlp(obv)
        # print(f"rsi_embeddings: {rsi_embeddings.shape}")
        ## reshape rsi_embeddings to be [batch_size, num_channels, horizon, 4]

        # rsi_embeddings = rsi_embeddings.view(batch_size, self.num_channels, self.horizon, 4)
        # print(f"rsi_embeddings: {rsi_embeddings.shape}")
        # #expanding rsi_embeddings to match the shape of logits
        # rsi_embeddings = rsi_embeddings.unsqueeze(3).expand(-1, -1, -1, self.num_outputs, -1)
        # print(f"rsi_embeddings: {rsi_embeddings.shape}")

        volume_change = compute_volume_change(historical_data[..., 4], period=5)
        print(f"volume_change: {volume_change.shape}")
        ## get last 7 values of volume_change
        volume_change = volume_change[:, :, -7:]
        #breakpoint()

        # volume_change_embeddings = self.volume_change_mlp(volume_change)
        # print(f"volume_change_embeddings: {volume_change_embeddings.shape}")
        # ## reshape volume_change_embeddings to be [batch_size, num_channels, horizon, 4]
        # volume_change_embeddings = volume_change_embeddings.view(batch_size, self.num_channels, self.horizon, self.num_outputs)
        # print(f"volume_change_embeddings: {volume_change_embeddings.shape}")
        # #expanding volume_change_embeddings to match the shape of logits

        # Compute a quantile adjustment factor
        # reshape channel_embeddings to be [batch_size, num_channels, 1, 32]

        #channel_embeddings = self.channel_embeddings.unsqueeze(0).expand(batch_size, -1)
        #channel_embeddings = self.channel_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        #print(f"channel_embeddings: {channel_embeddings.shape}")
        quantile_adjustment_embeddings = self.quantile_adjustment_embeddings(torch.cat((volume_change, rsi, obv, x_bollinger), dim=2))
        print(f"quantile_adjustment_embeddings: {quantile_adjustment_embeddings.shape}")
        ## reshape quantile_adjustment_embeddings to be [batch_size, num_channels, horizon, 4]
        quantile_adjustment_embeddings = quantile_adjustment_embeddings.view(batch_size, self.num_channels, self.horizon, self.num_outputs)

        # **Define Learnable Logits**
        logits = self.price_mixture_logits  # [num_channels, num_outputs, horizon, 4]
        logits = logits.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_channels, num_outputs, horizon, 4]
        #print(f"logits: {logits.shape}")

        # Compute scaling factors using embeddings
        scaling_factors = torch.sigmoid(bollinger_embeddings)# + stochastic_embeddings)

        # Apply to logits before softmax
        logits = self.price_mixture_logits * (1 + scaling_factors)


        # **Apply Softmax to Get Simplex-Consistent Weights**
        weights = torch.softmax(logits, dim=-1)  # [num_channels, num_outputs, 4] - 4 components
        #print(f"weights: {weights.shape}")

        #weights = weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch_size, num_channels, num_outputs, horizon, 4]
        w_close, w_recent, w_high, w_low = weights.split(1, dim=-1)  # Still [batch_size, num_channels, num_outputs, horizon, 1]

        # Remove last dimension without affecting batch structure
        w_close = w_close.squeeze(-1)
        w_recent = w_recent.squeeze(-1)
        w_high = w_high.squeeze(-1)
        w_low = w_low.squeeze(-1)

        #print(w_close.size())
        #print(close_price_means.size())


        # **Compute Final Weighted Forecast**
        weighted_prices = (
            w_close * close_price_means +
            w_recent * most_recent_prices +
            w_high * raw_high_prices +
            w_low * raw_low_prices
        )

        print(f"weighted_prices: {weighted_prices.shape}")

        # Compute a quantile adjustment factor
        quantile_adjustment = torch.tanh(quantile_adjustment_embeddings)

        alpha_signal = torch.sigmoid(self.alpha)

        # Adjust predicted quantiles
        #adjusted_predictions = weighted_prices * (1 + quantile_adjustment * 0.1)  # 10% scaling limit


        adjusted_predictions = weighted_prices  + (quantile_adjustment * alpha_signal)  # 10% scaling limit


          
        return adjusted_predictions