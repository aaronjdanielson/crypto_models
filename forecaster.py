import torch
import matplotlib.pyplot as plt


class MultiHorizonForecaster:
    def __init__(self, model, config, train_config, crypto_to_index, target_scaler=None):
        """
        Wrapper class for a trained TemporalCrossChannelTransformer to handle forecasting and evaluation.

        Args:
            model (torch.nn.Module): Trained TemporalCrossChannelTransformer model.
            config (dict): Training configuration dictionary containing:
                - 'model': Model configuration with quantiles.
                - 'device': Device to run the model on.
            train_config (dict): Training configuration dictionary.
            crypto_to_index (dict): Mapping from crypto names to channel indices.
            target_scaler (callable, optional): Function to inverse scale the targets and predictions.

        """
        self.model = model
        self.train_config = train_config
        self.config = config
        self.quantiles = config.model.output_quantiles
        self.device = train_config['device']
        self.crypto_to_index = crypto_to_index
        self.target_scaler = target_scaler  # Store the scaler

    def predict(self, batch):
        """
        Generate predictions for a batch of data.

        Args:
            batch (dict): Input batch with keys:
                - 'historical_data': Tensor of shape [batch_size, num_channels, lookback, num_features].
                - 'time_features': Tensor of shape [batch_size, tau, num_features].

        Returns:
            torch.Tensor: Predictions of shape [batch_size, num_channels, tau, num_quantiles].
        """
        self.model.eval()
        with torch.no_grad():
            batch = {key: value.to(self.device) for key, value in batch.items()}
            predictions = self.model(batch)
        return predictions

    def evaluate_or_forecast(self, data_loader, target_scaler=None, forecast_mode=False):
        """Use stored scaler if none is explicitly passed."""
        if target_scaler is None:
            target_scaler = self.target_scaler  # Default to instance's scaler
        self.model.eval()
        all_targets, all_predictions = [], []
        historical_date_ranges, target_date_ranges = [], []

        with torch.no_grad():
            for batch in data_loader:
                inputs = {
                    'historical_data': batch['historical_data'].to(self.device),
                    'time_features': batch['known_features'].to(self.device),
                }

                predictions = self.model(inputs)
                all_predictions.append(predictions.cpu())

                if not forecast_mode and 'targets' in batch:
                    targets = batch['targets'].to(self.device)
                    all_targets.append(targets.cpu())
                    historical_date_ranges.extend(batch['historical_date_ranges'])
                    target_date_ranges.extend(batch['target_date_ranges'])

        all_predictions = torch.cat(all_predictions, dim=0)  # [total_batches, num_channels, tau, num_quantiles]
        
        if not forecast_mode and all_targets:
            all_targets = torch.cat(all_targets, dim=0)  # [total_batches, num_channels, tau]
        
        if target_scaler:
            # Inverse transform predictions
            original_pred_shape = all_predictions.shape
            # Reshape to [total_samples * tau * num_quantiles, num_channels]
            pred_reshaped = all_predictions.reshape(-1, original_pred_shape[1])
            pred_inv = target_scaler.inverse_transform(pred_reshaped)
            all_predictions = torch.tensor(pred_inv).reshape(original_pred_shape)
            
            if not forecast_mode and all_targets:
                # Inverse transform targets
                original_target_shape = all_targets.shape
                # Reshape to [total_samples * tau, num_channels]
                target_reshaped = all_targets.reshape(-1, original_target_shape[1])
                target_inv = target_scaler.inverse_transform(target_reshaped)
                all_targets = torch.tensor(target_inv).reshape(original_target_shape)

        self.plot_predictions(
            predictions=all_predictions,
            targets=all_targets if not forecast_mode else None,
            historical_date_ranges=historical_date_ranges,
            target_date_ranges=target_date_ranges if not forecast_mode else None,
            forecast_mode=forecast_mode,
        )
