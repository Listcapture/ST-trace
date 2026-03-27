#!/usr/bin/env python
"""
Full evaluation of ST-Trace pipeline on a dataset.

Usage:
    python scripts/evaluate.py --config configs/nlpr_mct.yaml \
        --transition-checkpoint experiments/checkpoints/transition_best.pth \
        --reid-checkpoint experiments/checkpoints/reid_best.pth
"""
import argparse
import yaml
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from st_trace.data.graph import CameraGraph
from st_trace.search.st_anbs import STANBS
from st_trace.models.transition_net import TransitionNet
from st_trace.models.detector import YOLOPersonDetector
from st_trace.tracking.pipeline import CoarseToFineRetrieval
from st_trace.evaluation.metrics import evaluate_full
from st_trace.evaluation.efficiency import EfficiencyTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--transition-checkpoint', type=str, required=True)
    parser.add_argument('--reid-checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='./experiments/results')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    # ... dataset loading code here

    # Get camera graph
    camera_graph = dataset.get_camera_graph()

    # Load transition model
    transition_model = TransitionNet()
    checkpoint = torch.load(args.transition_checkpoint, map_location=device)
    transition_model.load_state_dict(checkpoint['model_state_dict'])
    transition_model.eval()
    transition_model.to(device)

    # Create ST-ANBS
    st_anbs = STANBS(
        camera_graph=camera_graph,
        transition_model=transition_model,
        beam_width=config.get('beam_width', 5),
        max_depth=config.get('max_depth', 6),
        max_duration_min=config.get('max_duration_min', 30),
        gamma=config.get('gamma', 0.9),
        lambda_length=config.get('lambda_length', 0.1),
        device=device
    )

    # Load ReID model
    # ... load ReID model

    # Create detector
    detector = YOLOPersonDetector(device=device)

    # Create retrieval pipeline
    pipeline = CoarseToFineRetrieval(
        camera_graph=camera_graph,
        st_anbs=st_anbs,
        detector=detector,
        reid_model=model,
        similarity_threshold=config.get('similarity_threshold', 0.75),
        alpha_adaptive=config.get('alpha_adaptive', 0.0),
        device=device
    )

    # Get probes
    probes = dataset.get_probes()
    print(f"Evaluating on {len(probes)} probes...")

    # Initialize efficiency tracker
    # Compute total frames: sum over all cameras of video lengths
    total_frames = ...  # computed from dataset
    tracker = EfficiencyTracker(total_frames)
    tracker.start_total()

    # Run evaluation on all probes
    all_results = []
    all_hypotheses = []
    all_gt = []

    for probe in tqdm(probes):
        retrieved, temporal_map = pipeline.retrieve(
            start_camera=probe['start_camera'],
            start_time=probe['start_time'],
            probe_feature=...,  # extract from probe
            video_root=config['video_root'],
            do_iterative_refinement=config.get('iterative_refinement', True)
        )

        # Collect results
        # ... build hypothesis trajectories

        # Count processed frames
        total_processed = sum(
            int((t_max - t_min) * pipeline.keyframe_fps)
            for (t_min, t_max) in temporal_map.values()
        )
        tracker.increment_processed(total_processed)
        all_results.append({
            'probe_id': probe['probe_id'],
            'num_retrieved': len(retrieved),
            'cameras_predicted': len(temporal_map)
        })

    # Compute metrics
    efficiency_metrics = tracker.compute_metrics()
    accuracy_metrics = evaluate_full(all_hypotheses, all_gt)

    # Combine all metrics
    result = {
        'config': config,
        'accuracy': accuracy_metrics,
        'efficiency': efficiency_metrics,
        'num_probes': len(probes)
    }

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f'results_{config["dataset"]}.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Print results
    print("\n=== Final Results ===")
    print(f"Dataset: {config['dataset']}")
    print(f"MOTA: {accuracy_metrics['MOTA']:.2f}")
    print(f"IDF1: {accuracy_metrics['IDF1']:.2f}")
    print(f"MT: {accuracy_metrics['MT']:.2f}%")
    print(f"ML: {accuracy_metrics['ML']:.2f}%")
    print(f"VRR: {efficiency_metrics['VRR']:.2f}%")
    print(f"FPS: {efficiency_metrics['FPS']:.2f}")
    print(f"Total runtime: {efficiency_metrics['total_time_minutes']:.2f} min")
    print(f"\nResults saved to {output_file}")

    tracker.print_breakdown()


if __name__ == '__main__':
    main()
