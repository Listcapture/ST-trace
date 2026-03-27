"""
Transition Network for ST-ANBS.

This module implements the neural network that predicts camera transition
probabilities conditioned on:
- Edge features (distance, temporal context)
- Historical trajectory (LSTM encoding)
- Temporal context (hour of day, day of week)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class TransitionNet(nn.Module):
    """
    Transition Network predicts P(next_camera | current_camera, history, temporal context).

    Architecture:
    - Edge embedding: [distance, t_min, t_max, hour_of_day] -> hidden
    - LSTM encodes trajectory history
    - MLP predicts probability
    """
    def __init__(
        self,
        edge_dim: int = 4,
        temporal_dim: int = 2,
        hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Args:
            edge_dim: Dimension of edge features
            temporal_dim: Dimension of temporal context (hour, day)
            hidden_dim: MLP hidden dimension
            lstm_hidden_dim: LSTM hidden dimension for trajectory history
            dropout: Dropout probability
        """
        super().__init__()

        # Edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM for trajectory history encoding
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Final MLP for probability prediction
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

    def forward(
        self,
        edge_features: torch.Tensor,
        temporal_context: torch.Tensor,
        history: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            edge_features: Edge features (distance, t_min, t_max, ...)
                           shape: (batch, num_neighbors, edge_dim)
            temporal_context: Hour of day, day of week
                              shape: (batch, 2)
            history: Historical trajectory encoding
                      shape: (batch, seq_len, hidden_dim)
                      For first step, seq_len = 0

        Returns:
            Transition probabilities for each neighbor: (batch, num_neighbors)
        """
        batch_size, num_neighbors, _ = edge_features.shape

        # Broadcast temporal context to all neighbors
        temporal_expanded = temporal_context.unsqueeze(1).expand(
            -1, num_neighbors, -1
        )

        # Concatenate edge features and temporal context
        x = torch.cat([edge_features, temporal_expanded], dim=-1)

        # Encode each edge
        x_encoded = self.edge_encoder(x)  # (B, N, H)

        if history is None or history.size(1) == 0:
            # No history - first step, use zero LSTM hidden
            h_0 = torch.zeros(
                1, batch_size * num_neighbors, self.lstm_hidden_dim,
                device=x_encoded.device
            )
            c_0 = torch.zeros_like(h_0)
            # Each neighbor is independent
            x_encoded_reshaped = x_encoded.reshape(
                batch_size * num_neighbors, 1, self.hidden_dim
            )
        else:
            # Get final hidden state from LSTM
            _, (h_n, _) = self.lstm(history)
            h_n = h_n[-1]  # (batch, lstm_hidden)
            # Broadcast to neighbors
            h_expanded = h_n.unsqueeze(1).expand(
                batch_size, num_neighbors, self.lstm_hidden_dim
            )
            x_encoded_reshaped = x_encoded

        if history is None or history.size(1) == 0:
            # LSTM over the empty history - just get the hidden
            _, (h_n, _) = self.lstm(x_encoded_reshaped)
            h_final = h_n[-1]  # (B*N, lstm_hidden)
            h_final = h_final.reshape(batch_size, num_neighbors, self.lstm_hidden_dim)
        else:
            h_final = h_expanded

        # Concatenate edge encoding and history encoding
        combined = torch.cat([x_encoded, h_final], dim=-1)

        # Predict probability
        probs = self.mlp(combined).squeeze(-1)  # (B, N)

        return probs

    def encode_history(
        self,
        path_edges: List[torch.Tensor],
        device: torch.device = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a trajectory history using LSTM.

        Args:
            path_edges: List of edge feature tensors along the path
            device: Target device

        Returns:
            (encoded_sequence, (h_n, c_n))
        """
        if not path_edges:
            return None, (None, None)

        # Stack sequence
        sequence = torch.stack(path_edges, dim=1)  # (1, seq_len, hidden_dim)
        output, (h_n, c_n) = self.lstm(sequence)

        return output, (h_n, c_n)

    @staticmethod
    def compute_edge_features(
        from_camera: int,
        to_cameras: List[int],
        camera_graph,
        current_time: float
    ) -> torch.Tensor:
        """
        Compute edge features for all neighbors of current camera.

        Args:
            from_camera: Current camera ID
            to_cameras: List of candidate next cameras
            camera_graph: CameraGraph object
            current_time: Current timestamp

        Returns:
            edge_features: tensor shape (1, num_neighbors, edge_dim)
        """
        features = []
        hour = (current_time / 3600.0) % 24 / 24.0  # normalize to [0,1]
        day = 0.0  # placeholder, can be computed from actual data

        for to_cam in to_cameras:
            dist = camera_graph.get_distance(from_camera, to_cam)
            t_min, t_max = camera_graph.get_travel_time_range(from_camera, to_cam)
            # Normalize
            dist_norm = dist / 100.0  # normalize by 100m
            t_min_norm = t_min / 600.0  # normalize by 10 minutes
            t_max_norm = t_max / 600.0

            features.append([dist_norm, t_min_norm, t_max_norm, hour])

        tensor = torch.tensor(features, dtype=torch.float32)
        return tensor.unsqueeze(0)  # (1, num_neighbors, 4)
