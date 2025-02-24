import torch
import torch.nn as nn


class MLPBlock(nn.Module):
        def __init__(self, input_size, output_size, num_layers=4, hidden_size=64, dropout=0.1 ) -> None:
            """
            Constructs an MLP block with:
            - First layer: input_size -> hidden_size
            - (num_layers - 2) middle layers: hidden_size -> hidden_size
            - Last layer: hidden_size -> output_sizes
            A skip connection is added, projecting input to output_size if needed.
            """
            super().__init__()

            layers = []

                # First layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Middle layers (if any)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            # Last layer
            layers.append(nn.Linear(hidden_size, output_size))
            
            # Combine all layers into a sequential model.
            self.model = nn.Sequential(*layers)
        
            if input_size != output_size:
                self.skip = nn.Linear(input_size, output_size)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            return self.skip(x) + self.model(x)