import torch
from torch import optim
from torch import nn
from dataclasses import dataclass
from typing import List
from torch.utils.data import DataLoader
from dataloader import CryptoDataset, custom_collate_fn
from models import CommonMetricsModel
from loss_functions import compute_pinball_loss, compute_q_risk, compute_coverage_percentage
from forecaster import MultiHorizonForecaster

class CryptoScaler:
    def __init__(self, crypto_stats, crypto_order):
        self.crypto_stats = crypto_stats  # Dict with mean/std per crypto
        self.crypto_order = crypto_order  # List of cryptos in channel order

    def inverse_transform(self, data):
        """Inverse scales data using stored crypto statistics."""
        rescaled = data.clone()
        for channel_idx, crypto in enumerate(self.crypto_order):
            mean = self.crypto_stats[crypto]['mean']
            std = self.crypto_stats[crypto]['std']
            rescaled[:, channel_idx, ...] = rescaled[:, channel_idx, ...] * std + mean
        return rescaled

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


config = Config(
    data_props=DataProps(
        num_historical_features=6,
        num_time_features=5,
        num_channels=13,
        lookback=56,
        horizon=14
    ),
    model=ModelConfig(
        output_quantiles=[0.1, 0.5, 0.9],
    )
)

training_dates = {
        'train_start': '2020-11-11',
        'train_end': '2024-05-31',
        'validation_start': '2024-06-01',
        'validation_end': '2024-12-31',
        'test_start': '2025-01-01',
        'test_end': '2025-01-14',
    }

train_config = {
    'batch_size': 64,
    'num_epochs': 200,
    'learning_rate': 1e-2,  ## use 1e-2 for non-deep prediction
    'weight_decay': 1e-4,  ## use 1e-4 for non-deep prediction
    'early_stopping_patience': 5,
    'miniumum_epochs': 5,
    'device': torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"),
    'training_dates': training_dates,
    'file_path': 'data/combined_data.csv',
    'date_attrs': ['day_of_week', 'month', 'day_of_month', 'quarter', 'year_since_earliest_date'],
    'crypto_data_features': ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'],
}

# Generate dynamic model filename based on configuration

model_type = "baseline"

#model_type += f"_main_method_{config.model.main_method}"

# Construct model filenames
best_model_filename = f"best_model_{model_type}.pth"
final_model_filename = f"final_model_{model_type}.pth"
forecaster_filename = f"forecaster_{model_type}.pth"

def train_model(model_config=config, train_config=train_config):
    device = train_config['device']
    batch_size = train_config['batch_size']
    num_epochs = train_config['num_epochs']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    early_stopping_patience = train_config['early_stopping_patience']

    # Data preparation
    train_dataset = CryptoDataset(
        file_path=train_config['file_path'],
        lookback=model_config.data_props.lookback,
        horizon=model_config.data_props.horizon,
        date_attrs=train_config['date_attrs'],
        crypto_data_features=train_config['crypto_data_features'],
        date_ranges=train_config['training_dates'],
        split="train",
        normalize="zscore",
    )
    validation_dataset = CryptoDataset(
        file_path=train_config['file_path'],
        lookback=model_config.data_props.lookback,
        horizon=model_config.data_props.horizon,
        date_attrs=train_config['date_attrs'],
        crypto_data_features=train_config['crypto_data_features'],
        date_ranges=train_config['training_dates'],
        split="validation",
        normalize="zscore",
        crypto_stats=train_dataset.crypto_stats,
    )

    # Extract scaler parameters
    crypto_order = list(train_dataset.crypto_index_mapping.keys())
    crypto_stats = train_dataset.crypto_stats
    scaler = CryptoScaler(crypto_stats, crypto_order)

    # Retrieve crypto-to-index mapping
    crypto_to_index = train_dataset.crypto_index_mapping
    median_quantile_index = config.model.output_quantiles.index(0.5)  # Index of median quantile (0.5)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4
        )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model = CommonMetricsModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    quantiles = torch.tensor(config.model.output_quantiles, device=device)

    best_dev_loss = float('inf')
    epochs_without_improvement = 0

    best_q_risk = torch.full((len(config.model.output_quantiles),), float('inf'), device=device)  # Initialize best Q-risk

    for epoch in range(num_epochs):
        q_risk_improved = False  # Reset flag
        model.train()
        epoch_loss = 0.0
        for inputs in train_loader:
            batch = {
                'historical_data': inputs['historical_data'].to(device),
                'time_features': inputs['date_features'].to(device),
            }
            targets = inputs['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = compute_pinball_loss(outputs, targets, quantiles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        validation_loss = 0.0
        total_q_risk = torch.zeros(len(config.model.output_quantiles), device=device)  # Initialize Q-risk accumulator
        # Dictionary to store errors for each crypto
        crypto_median_errors = {crypto: [] for crypto in crypto_to_index.keys()}
        with torch.no_grad():
            for inputs in validation_loader:
                batch = {
                    'historical_data': inputs['historical_data'].to(device),
                    'time_features': inputs['date_features'].to(device),
                }
                targets = inputs['targets'].to(device)
                outputs = model(batch)

                # Compute pinball loss
                loss = compute_pinball_loss(outputs, targets, quantiles)
                validation_loss += loss.item()

                # Compute Q-risk
                q_risk = compute_q_risk(outputs, targets, quantiles)  # Returns tensor of shape [num_quantiles]
                total_q_risk += q_risk

                # Extract median quantile predictions and calculate errors
                median_predictions = outputs[..., median_quantile_index]  # [batch_size, num_channels, tau]
                median_rescaled_predictions = validation_dataset.inverse_scale_targets(
                    median_predictions, list(crypto_to_index.keys())
                )
                target_rescaled = validation_dataset.inverse_scale_targets(
                    targets, list(crypto_to_index.keys())
                )
                median_errors = torch.abs(median_rescaled_predictions - target_rescaled)  # [batch_size, num_channels, tau]

                # Aggregate errors per crypto
                for crypto, idx in crypto_to_index.items():
                    crypto_median_errors[crypto].extend(median_errors[:, idx, :].flatten().tolist())


        avg_dev_loss = validation_loss / len(validation_loader)
        avg_q_risk = total_q_risk / len(validation_loader)  # Normalize Q-risk over validation set
        # Compute coverage percentage
        coverage_percentage = compute_coverage_percentage(outputs, targets, config.model.output_quantiles)
        print(f"Coverage Percentage (Within High-Low Quantile Range): {coverage_percentage:.2f}%")

        mean_median_errors = {crypto: torch.tensor(errors).mean().item() if errors else float('nan')
                      for crypto, errors in crypto_median_errors.items()}
        print(f"Validation Loss: {avg_dev_loss:.4f}")
        print(f"Validation Q-Risk: {avg_q_risk}")
        print(f"Median Errors Per Crypto: {mean_median_errors}")

        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_filename)
            print("Validation performance improved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience and epoch >= train_config['miniumum_epochs']:
            print("Early stopping triggered.")
            break

    return model, config, train_config, crypto_to_index, scaler  # Return scaler

if __name__ == "__main__":
    model, config, train_config, crypto_to_index, scaler = train_model()
    # Save the final trained model
    torch.save(model.state_dict(), final_model_filename)
    print(f"Final trained model saved as {final_model_filename}")

    # Initialize forecaster and save it
    forecaster = MultiHorizonForecaster(model, config, train_config, crypto_to_index, target_scaler=scaler)
    torch.save(forecaster, forecaster_filename)
    print(f"Forecaster saved as {forecaster_filename}")





