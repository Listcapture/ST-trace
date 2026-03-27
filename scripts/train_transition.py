#!/usr/bin/env python
"""
Train ST-ANBS transition prediction model.

Usage:
    python scripts/train_transition.py --config configs/nlpr_mct.yaml
"""
import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

from st_trace.models.transition_net import TransitionNet


class TransitionDataset(Dataset):
    """Dataset for training transition prediction model."""
    def __init__(self, samples, camera_graph):
        self.samples = samples
        self.camera_graph = camera_graph

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        from_cam = sample['from_camera']
        to_cam = sample['to_camera']
        timestamp = sample['timestamp']

        # Compute edge features
        neighbors = self.camera_graph.get_neighbors(from_cam)
        features = []
        hour = (timestamp / 3600.0) % 24 / 24.0
        day = 0.0

        target = -1
        for n_idx, n in enumerate(neighbors):
            dist = self.camera_graph.get_distance(from_cam, n)
            t_min, t_max = self.camera_graph.get_travel_time_range(from_cam, n)
            dist_norm = dist / 100.0
            t_min_norm = t_min / 600.0
            t_max_norm = t_max / 600.0
            features.append([dist_norm, t_min_norm, t_max_norm, hour])
            if n == to_cam:
                target = n_idx

        return {
            'edge_features': torch.tensor(features, dtype=torch.float32),
            'temporal_context': torch.tensor([hour, day], dtype=torch.float32),
            'target': target
        }


def collate_fn(batch):
    """Custom collate for variable number of neighbors."""
    edge_features = [b['edge_features'] for b in batch]
    temporal_context = torch.stack([b['temporal_context'] for b in batch])
    targets = [b['target'] for b in batch]
    # Pad to max length
    max_n = max(ef.shape[0] for ef in edge_features)
    padded = []
    mask = []
    for ef in edge_features:
        n = ef.shape[0]
        pad = torch.zeros(max_n - n, ef.shape[1])
        padded.append(torch.cat([ef, pad], dim=0))
        m = torch.zeros(max_n, dtype=torch.bool)
        m[:n] = True
        mask.append(m)
    padded = torch.stack(padded)
    mask = torch.stack(mask)
    return padded, temporal_context, targets, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./experiments/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # TODO: Load dataset and training samples
    # This needs full dataset loading to get transitions

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = TransitionNet(
        edge_dim=4,
        temporal_dim=2,
        hidden_dim=256,
        lstm_hidden_dim=128,
        dropout=0.3
    )
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.BCELoss()

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print("Starting training...")

    # Training loop (skeleton - filled when data is available)
    os.makedirs(args.output_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # TODO: iterate over dataloader

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss = {total_loss:.4f}, acc = {correct/total:.4f}")

            # Save checkpoint
            if correct/total > best_acc:
                best_acc = correct/total
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(args.output_dir, f'transition_{config["dataset"]}_best.pth'))

    print(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
