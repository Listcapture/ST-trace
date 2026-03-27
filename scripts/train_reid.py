#!/usr/bin/env python
"""
Train ReID with spatio-temporal contrastive learning.

Usage:
    python scripts/train_reid.py --config configs/nlpr_mct.yaml
"""
import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from st_trace.models.reid.transreid import transreid_base
from st_trace.models.reid.st_contrastive import STContrastiveReIDLoss
from st_trace.data.transforms import get_train_transform, get_val_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./experiments/checkpoints')
    parser.add_argument('--log-dir', type=str, default='./experiments/logs')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters from config
    num_classes = config.get('num_classes', 702)
    lambda_st = config.get('lambda_st', 0.3)
    lambda_triplet = config.get('lambda_triplet', 0.5)
    lr = config.get('lr', 3e-4)
    epochs = config.get('epochs', 120)
    batch_size = config.get('batch_size', 64)

    # Create model
    model = transreid_base(num_classes=num_classes)
    model = model.to(device)

    # Create loss
    criterion = STContrastiveReIDLoss(
        num_classes=num_classes,
        feature_dim=768,
        lambda_triplet=lambda_triplet,
        lambda_st=lambda_st,
        temperature=0.07
    )
    criterion = criterion.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Tensorboard logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'reid_{config["dataset"]}'))

    print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print("Starting training...")

    # TODO: Data loading needs to be implemented when dataset is ready
    # Training loop skeleton

    os.makedirs(args.output_dir, exist_ok=True)

    best_map = 0.0
    for epoch in range(epochs):
        model.train()
        # Training step here when data is ready

        scheduler.step()

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'reid_{config["dataset"]}_epoch_{epoch+1}.pth'))

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    main()
