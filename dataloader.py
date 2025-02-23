import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class CryptoDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 lookback: int, 
                 horizon: int, 
                 date_attrs: List[str], 
                 crypto_data_features: List[str], 
                 date_ranges: Dict[str, str], 
                 split: str = "train", 
                 normalize: Optional[str] = "zscore",
                 crypto_stats: Optional[Dict[str, Dict[str, float]]] = None):
        self.data = pd.read_csv(file_path)
        self.lookback = lookback
        self.horizon = horizon
        self.date_attrs = date_attrs
        self.crypto_data_features = crypto_data_features
        self.normalize = normalize
        self.split = split
        self.date_ranges = date_ranges

        # Convert date column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Start'])

        # Sort by date for sequential time slicing
        self.data = self.data.sort_values(by=['Source', 'Date']).reset_index(drop=True)

        # Calculate known features based on dates
        self.data['day_of_week'] = self.data['Date'].dt.dayofweek
        self.data['day_of_month'] = self.data['Date'].dt.day
        self.data['month'] = self.data['Date'].dt.month
        self.data['quarter'] = self.data['Date'].dt.quarter
        self.data['year_since_earliest_date'] = (self.data['Date'] - self.data['Date'].min()).dt.days / 365

        # Filter data by split using date ranges from the dictionary
        if split == "train":
            #start_date = pd.to_datetime(self.date_ranges['train_start']) + pd.Timedelta(days=self.lookback)
            #end_date = pd.to_datetime(self.date_ranges['train_end']) - pd.Timedelta(days=self.horizon)
            self.data = self.data[
                (self.data['Date'] >= pd.to_datetime(self.date_ranges['train_start'])) & 
                (self.data['Date'] <= pd.to_datetime(self.date_ranges['train_end']))
            ]
        elif split == "validation":
            #start_date = pd.to_datetime(self.date_ranges['validation_start']) + pd.Timedelta(days=self.lookback)
            #end_date = pd.to_datetime(self.date_ranges['validation_end']) - pd.Timedelta(days=self.horizon)
            self.data = self.data[
                (self.data['Date'] >= pd.to_datetime(self.date_ranges['validation_start'])) & 
                (self.data['Date'] <= pd.to_datetime(self.date_ranges['validation_end']))
            ]
        elif split == "test":
            #start_date = pd.to_datetime(self.date_ranges['test_start']) + pd.Timedelta(days=self.lookback)
            #end_date = pd.to_datetime(self.date_ranges['test_end']) - pd.Timedelta(days=self.horizon)
            self.data = self.data[
                (self.data['Date'] >= pd.to_datetime(self.date_ranges['test_start'])) & 
                (self.data['Date'] <= pd.to_datetime(self.date_ranges['test_end']))
            ]
        elif split == "forecast":
            # Use the last `lookback` rows for each cryptocurrency
            self.data = self.data.groupby('Source').tail(lookback)
        else:
            raise ValueError("Invalid split. Choose 'train', 'validation', 'test', or 'forecast'.")

        # Compute or reuse per-crypto normalization statistics
        if normalize:
            if split == "train":
                self.crypto_stats = {}
                for crypto, group in self.data.groupby('Source'):
                    self.crypto_stats[crypto] = {
                        'mean': group[crypto_data_features].mean(),
                        'std': group[crypto_data_features].std(),
                        'min': group[crypto_data_features].min(),
                        'max': group[crypto_data_features].max(),
                    }
                    # Print normalization values for debugging
                    print(f"Crypto: {crypto}")
                    print(f"Mean: {self.crypto_stats[crypto]['mean']}")
                    print(f"Std: {self.crypto_stats[crypto]['std']}")
                    print(f"Min: {self.crypto_stats[crypto]['min']}")
                    print(f"Max: {self.crypto_stats[crypto]['max']}")
            else:
                self.crypto_stats = crypto_stats  # Reuse provided stats
        else:
            self.crypto_stats = None

        # Group data by cryptocurrency source
        self.crypto_groups = {crypto: group for crypto, group in self.data.groupby('Source')}

        # Create a presence table
        self.presence_table = self._create_presence_table()
        
        # Create a crypto index mapping
        self.crypto_index_mapping = {crypto: idx for idx, crypto in enumerate(self.crypto_groups.keys())}
    
    def _create_presence_table(self) -> pd.DataFrame:
        """Creates a table tracking the presence of each crypto by date."""
        presence_table = pd.crosstab(self.data['Date'], self.data['Source']).astype(int).reset_index()
        presence_table = presence_table.sort_values(by='Date').reset_index(drop=True)
        return presence_table
    
    def inverse_scale_targets(self, normalized_values: torch.Tensor, crypto_ids: List[str]) -> torch.Tensor:
        """
        Inverts the normalization applied to the target values.

        Args:
            normalized_values (torch.Tensor): Normalized values of shape [batch_size, num_cryptos, horizon].
            crypto_ids (List[str]): List of crypto names corresponding to each channel.

        Returns:
            torch.Tensor: Denormalized values of the same shape as `normalized_values`.
        """
        denormalized_values = torch.zeros_like(normalized_values)
        for i, crypto in enumerate(crypto_ids):
            if crypto not in self.crypto_stats:
                raise ValueError(f"Missing stats for crypto: {crypto}")

            stats = self.crypto_stats[crypto]
            if self.normalize == "zscore":
                denormalized_values[:, i, :] = normalized_values[:, i, :] * stats['std']['Close'] + stats['mean']['Close']
            elif self.normalize == "minmax":
                min_val = stats['min']['Close']
                max_val = stats['max']['Close']
                denormalized_values[:, i, :] = normalized_values[:, i, :] * (max_val - min_val) + min_val
            else:
                raise ValueError(f"Unsupported normalization method: {self.normalize}")

        return denormalized_values


    def __len__(self):
        if self.split == "forecast":
            return 1  # Forecast split always has one batch
        return max(len(group) for group in self.crypto_groups.values()) - self.lookback - self.horizon + 1

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Retrieves a batch of data for the specified index, including historical and target data,
        for training, validation, testing, or forecasting.

        The method extracts a valid window of `lookback + horizon` rows, ensuring that all
        samples returned are complete and adhere to the specified date ranges. Data is normalized
        if a normalization strategy is specified, and the presence of missing values is indicated
        using masks.

        The set of all possible samples is determined by the set of consecutive dates of length lookback + horizon  
        in the presence table. The method retrieves the data for the index-th sample by extracting the data for the

        Args:
            idx (int): The index of the batch to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the following keys:
                - 'historical_data' (torch.Tensor): Historical feature data of shape 
                  (num_cryptos, lookback, num_features).
                - 'historical_mask' (torch.Tensor): Mask indicating missing values in the 
                  historical data, of shape (num_cryptos, lookback, num_features).
                - 'date_features' (torch.Tensor): Temporal features of shape 
                  (lookback + horizon, num_date_features).
                - 'historical_date_ranges' (List[Tuple]): List of tuples indicating the date ranges
                  for the historical data.

                If the split is not "forecast," the following additional keys are included:
                - 'targets' (torch.Tensor): Target feature data of shape 
                  (num_cryptos, horizon).
                - 'targets_mask' (torch.Tensor): Mask indicating missing values in the target 
                  data, of shape (num_cryptos, horizon).
                - 'target_date_ranges' (List[Tuple]): List of tuples indicating the date ranges 
                  for the target data.

        Raises:
            IndexError: If the provided index is out of range.
        """
        lookback_plus_horizon = self.lookback + self.horizon
        valid_date_ranges = []

        # Identify all valid date ranges of length `lookback + horizon`
        presence_table_values = self.presence_table.drop(columns="Date").values  # Exclude Date column
        for start_idx in range(len(presence_table_values) - lookback_plus_horizon + 1):
            window = presence_table_values[start_idx: start_idx + lookback_plus_horizon]
            if np.all(window.sum(axis=0) == lookback_plus_horizon):  # Ensure full presence
                valid_date_ranges.append(start_idx)

        if idx >= len(valid_date_ranges):
            raise IndexError(f"Index {idx} is out of bounds for valid date ranges.")

        start_idx = valid_date_ranges[idx]
        end_idx = start_idx + lookback_plus_horizon
        date_range = self.presence_table['Date'].iloc[start_idx:end_idx].values

        historical_data_list, historical_mask_list = [], []
        targets_list, targets_mask_list = [], []

        # Process data for each cryptocurrency
        for crypto, group in self.crypto_groups.items():
            group_values = group.reset_index(drop=True)
            group_dates = group_values['Date'].values

            # Filter group data for the current date range
            date_mask = np.isin(group_dates, date_range)
            valid_group_data = group_values[date_mask]
            if len(valid_group_data) != lookback_plus_horizon:
                raise ValueError(f"Incomplete data for crypto {crypto} in the date range.")

            # Split historical and target data
            historical_data = valid_group_data.iloc[:self.lookback][self.crypto_data_features].values
            historical_mask = np.zeros_like(historical_data)
            target_data = valid_group_data.iloc[self.lookback:]['Close'].values  # Only 'Close'
            target_mask = np.zeros_like(target_data)
            #target_data = valid_group_data.iloc[self.lookback:][self.crypto_data_features].values
            #target_mask = np.zeros_like(target_data)

            # Normalize data if applicable
            if self.normalize and crypto in self.crypto_stats:
                stats = self.crypto_stats[crypto]
                if self.normalize == "zscore":
                    historical_data = (historical_data - stats['mean'].values) / stats['std'].values
                    target_data = (target_data - stats['mean']['Close']) / stats['std']['Close']
                    #target_data = (target_data - stats['mean'].values) / stats['std'].values
                elif self.normalize == "minmax":
                    historical_data = (historical_data - stats['min'].values) / (stats['max'].values - stats['min'].values)
                    target_data = (target_data - stats['min']['Close']) / (stats['max']['Close'] - stats['min']['Close'])
                    #target_data = (target_data - stats['min'].values) / (stats['max'].values - stats['min'].values)

            # Append to respective lists
            historical_data_list.append(historical_data)
            historical_mask_list.append(historical_mask)
            if self.split != "forecast":
                targets_list.append(target_data)
                targets_mask_list.append(target_mask)

        # Stack data across all cryptos
        historical_data = np.stack(historical_data_list, axis=0)
        historical_mask = np.stack(historical_mask_list, axis=0)

        # Construct the result dictionary
        result = {
            'historical_data': torch.tensor(historical_data, dtype=torch.float),
            'historical_mask': torch.tensor(historical_mask, dtype=torch.float),
            'date_features': torch.tensor(
                self.data[self.data['Date'].isin(date_range)][self.date_attrs].values[:lookback_plus_horizon],
                dtype=torch.float
            ),
            'historical_date_ranges': list(date_range[:self.lookback]),
        }

        if self.split != "forecast":
            targets = np.stack(targets_list, axis=0)
            targets_mask = np.stack(targets_mask_list, axis=0)
            result.update({
                'targets': torch.tensor(targets, dtype=torch.float),
                'targets_mask': torch.tensor(targets_mask, dtype=torch.float),
                'target_date_ranges': list(date_range[self.lookback:]),
            })
        

        return result

# Custom collate function for handling variable-sized batches
def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Always present in the batch
    historical_data = torch.stack([b['historical_data'] for b in batch])
    historical_mask = torch.stack([b['historical_mask'] for b in batch])
    date_features = torch.stack([b['date_features'] for b in batch])
    historical_date_ranges = [b['historical_date_ranges'] for b in batch]

    # Conditional stacking for non-forecast splits
    if 'targets' in batch[0]:
        targets = torch.stack([b['targets'] for b in batch])
        targets_mask = torch.stack([b['targets_mask'] for b in batch])
        target_date_ranges = [b['target_date_ranges'] for b in batch]

        return {
            'historical_data': historical_data,
            'historical_mask': historical_mask,
            'date_features': date_features,
            'targets': targets,
            'targets_mask': targets_mask,
            'historical_date_ranges': historical_date_ranges,
            'target_date_ranges': target_date_ranges,
        }
    else:
        return {
            'historical_data': historical_data,
            'historical_mask': historical_mask,
            'date_features': date_features,
            'historical_date_ranges': historical_date_ranges,
        }

if __name__ == "__main__":
    # Test CryptoDataset

    training_dates = {
        'train_start': '2020-11-11',
        'train_end': '2024-05-31',
        'validation_start': '2024-06-01',
        'validation_end': '2024-12-31',
        'test_start': '2025-01-01',
        'test_end': '2025-01-14',
    }
    batch_size = 64

    train_dataset = CryptoDataset(
        file_path='data/combined_data.csv',
        lookback=28,
        horizon=14,
        date_attrs=['day_of_week', 'day_of_month', 'month', 'quarter', 'year_since_earliest_date'],
        crypto_data_features=['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'],
        date_ranges=training_dates,
        split="train",
        normalize="minmax",
    )
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate_fn)
    for batch in train_loader:
        print(f"Train Historical Data Shape: {batch['historical_data'].shape}")
        print(f"Train Historical Mask Shape: {batch['historical_mask'].shape}")
        print(f"Train Known Features Shape: {batch['date_features'].shape}")
        print(f"Train Targets Shape: {batch['targets'].shape}")
        print(f"Train Targets Mask Shape: {batch['targets_mask'].shape}")
        #print(f"Train Historical Date Ranges: {batch['historical_date_ranges']}")
        #print(f"Train Target Date Ranges: {batch['target_date_ranges']}")
        print(batch['targets'])
        break