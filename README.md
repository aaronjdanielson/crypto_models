# Baseline Model - Training Instructions

This repository is organized into a few different files:

- dataloader.py reads from from the data directory and produces batches for training and validation in our training function.
- models.py contains the BaselineModel module.
- loss_functions.py contains the pinball loss function as well as other functions used to assess training performance
- train_baseline_model.py contains the training loop as well as config files required for training.
- forecaster.py is a class for a forecaster object used to produce predictions.

To train the baseline model:

```bash
python3 train_baseline_model.py
```
